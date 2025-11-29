"""
Smart Context Agent with Tool Calling.

Single LLM call that uses tools to fetch relevant context.
Replaces the need for separate RetrievalQueryAgent + ContextAssemblerAgent.

Uses gpt-4o-mini with native tool support for efficient context assembly.
"""
import json
from typing import Dict, Any, List, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass

from .base_agent import BaseAgent
from ..core.schemas import Message, AssembledContext, MetadataSchema, DEFAULT_METADATA_SCHEMA
from ..utils.logger import get_logger

if TYPE_CHECKING:
    from ..llm import BaseLLMProvider

logger = get_logger(__name__)


@dataclass
class ContextTools:
    """
    Tools available to the SmartContextAgent.

    These are callbacks that the agent can invoke to fetch data.
    """
    get_recent_messages: Callable[[str, str, int], List[Message]]
    search_history: Callable[[str, Optional[str], List[str], List[str], Optional[str], int], List[Message]]
    get_session_metadata: Callable[[str, str], Dict[str, Any]]


# Tool definitions for OpenAI function calling
CONTEXT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_recent_messages",
            "description": "Get the most recent messages from the current conversation thread. Use this for context about the ongoing conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of recent messages to retrieve (default: 10, max: 50)",
                        "default": 10
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_history",
            "description": "Search historical messages across threads. Use this when the user references past conversations, previous topics, or needs context from earlier interactions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topics to search for (e.g., ['billing', 'refund', 'account'])"
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Categories to filter by (e.g., ['support', 'sales'])"
                    },
                    "intent": {
                        "type": "string",
                        "description": "Intent type to filter by (e.g., 'question', 'complaint', 'feedback')"
                    },
                    "search_current_thread_only": {
                        "type": "boolean",
                        "description": "If true, only search within current thread. If false, search across all user's threads.",
                        "default": False
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of historical messages to retrieve (default: 20)",
                        "default": 20
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_metadata",
            "description": "Get aggregated metadata about the current session (topics discussed, categories, intents). Useful for understanding the overall context of the conversation.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


class SmartContextAgent(BaseAgent):
    """
    Intelligent context assembly agent using tool calling.

    Instead of separate query analysis and context assembly steps,
    this agent uses a single LLM call with tools to:
    1. Understand what context is needed
    2. Fetch relevant data via tool calls
    3. Assemble the final context

    This reduces latency by eliminating one LLM round-trip while
    maintaining intelligent context selection.

    Example:
        >>> from mindcore.llm import create_provider, ProviderType
        >>> provider = create_provider(ProviderType.OPENAI, ...)
        >>>
        >>> tools = ContextTools(
        ...     get_recent_messages=lambda uid, tid, limit: [...],
        ...     search_history=lambda uid, tid, topics, cats, intent, limit: [...],
        ...     get_session_metadata=lambda uid, tid: {...}
        ... )
        >>>
        >>> agent = SmartContextAgent(provider, tools)
        >>> context = agent.process(
        ...     query="What did we discuss about billing last time?",
        ...     user_id="user123",
        ...     thread_id="thread456"
        ... )
    """

    def __init__(
        self,
        llm_provider: "BaseLLMProvider",
        context_tools: ContextTools,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        metadata_schema: Optional[MetadataSchema] = None,
        max_tool_rounds: int = 3
    ):
        """
        Initialize smart context agent.

        Args:
            llm_provider: LLM provider instance (should support tool calling)
            context_tools: Callbacks for fetching context data
            temperature: Temperature for generation (lower for consistency)
            max_tokens: Maximum tokens in response
            metadata_schema: Schema for valid topics/categories/intents
            max_tool_rounds: Maximum rounds of tool calling (default: 3)
        """
        super().__init__(llm_provider, temperature, max_tokens)
        self.tools = context_tools
        self.metadata_schema = metadata_schema or DEFAULT_METADATA_SCHEMA
        self.max_tool_rounds = max_tool_rounds
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create system prompt for context assembly."""
        schema_list = self.metadata_schema.to_prompt_list()

        return f"""You are a context assembly agent. Your task is to gather relevant context for an AI assistant to answer a user's query.

You have access to tools that fetch conversation history and metadata. Use them intelligently:

1. **get_recent_messages**: Get recent messages from the current thread. Use this for ongoing conversation context.

2. **search_history**: Search historical messages by topics, categories, or intent. Use this when:
   - User references past conversations ("last time", "before", "previously")
   - User asks about something that might have been discussed earlier
   - You need broader context beyond the current thread

3. **get_session_metadata**: Get aggregated info about current session topics. Use this to understand what's been discussed.

{schema_list}

After gathering context, respond with a JSON object:
{{
    "relevant_context": "A concise summary of relevant information from the fetched messages",
    "key_points": ["Important point 1", "Important point 2", ...],
    "context_source": "recent|historical|both",
    "confidence": "high|medium|low"
}}

Guidelines:
- Be selective - don't fetch everything, only what's relevant to the query
- If the query is simple and recent context is enough, don't search history
- If the query references past interactions, search history
- Summarize the context concisely - the main AI doesn't need raw messages
- Focus on information that helps answer the user's query"""

    def _execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_id: str,
        thread_id: str
    ) -> str:
        """Execute a tool and return the result as a string."""
        try:
            if tool_name == "get_recent_messages":
                limit = arguments.get("limit", 10)
                messages = self.tools.get_recent_messages(user_id, thread_id, min(limit, 50))
                return self._format_messages(messages)

            elif tool_name == "search_history":
                topics = arguments.get("topics", [])
                categories = arguments.get("categories", [])
                intent = arguments.get("intent")
                current_only = arguments.get("search_current_thread_only", False)
                limit = arguments.get("limit", 20)

                search_thread_id = thread_id if current_only else None
                messages = self.tools.search_history(
                    user_id, search_thread_id, topics, categories, intent, limit
                )
                return self._format_messages(messages)

            elif tool_name == "get_session_metadata":
                metadata = self.tools.get_session_metadata(user_id, thread_id)
                return json.dumps(metadata, indent=2)

            else:
                return f"Unknown tool: {tool_name}"

        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return f"Error executing {tool_name}: {str(e)}"

    def _format_messages(self, messages: List[Message]) -> str:
        """Format messages for the LLM."""
        if not messages:
            return "No messages found."

        formatted = []
        for msg in messages[:30]:  # Limit to prevent context overflow
            meta_info = ""
            if msg.metadata and msg.metadata.topics:
                meta_info = f" [topics: {', '.join(msg.metadata.topics)}]"

            formatted.append(
                f"[{msg.role.value}]{meta_info}: {msg.raw_text[:500]}"
            )

        return "\n".join(formatted)

    def process(
        self,
        query: str,
        user_id: str,
        thread_id: str,
        additional_context: Optional[str] = None
    ) -> AssembledContext:
        """
        Assemble context for a query using tool calling.

        Args:
            query: User's query that needs context
            user_id: User identifier
            thread_id: Thread identifier
            additional_context: Optional additional context to include

        Returns:
            AssembledContext with relevant information
        """
        logger.debug(f"SmartContextAgent processing query: {query[:100]}...")

        # Build initial messages
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        user_content = f"User query: {query}"
        if additional_context:
            user_content += f"\n\nAdditional context: {additional_context}"

        messages.append({"role": "user", "content": user_content})

        # Tool calling loop
        for round_num in range(self.max_tool_rounds):
            try:
                response = self._call_llm_with_tools(messages)

                # Check if LLM wants to call tools
                if response.get("tool_calls"):
                    tool_results = []
                    for tool_call in response["tool_calls"]:
                        try:
                            tool_name = tool_call["function"]["name"]
                            arguments_str = tool_call["function"]["arguments"]
                            # Safe JSON parsing with fallback
                            try:
                                arguments = json.loads(arguments_str) if arguments_str else {}
                            except json.JSONDecodeError as je:
                                logger.warning(f"Failed to parse tool arguments: {je}")
                                arguments = {}

                            logger.debug(f"Tool call: {tool_name}({arguments})")

                            result = self._execute_tool(tool_name, arguments, user_id, thread_id)
                            tool_results.append({
                                "tool_call_id": tool_call["id"],
                                "role": "tool",
                                "content": result
                            })
                        except (KeyError, TypeError) as e:
                            logger.error(f"Malformed tool call: {e}")
                            tool_results.append({
                                "tool_call_id": tool_call.get("id", "unknown"),
                                "role": "tool",
                                "content": f"Error: malformed tool call - {e}"
                            })

                    # Add assistant message with tool calls
                    messages.append({
                        "role": "assistant",
                        "content": response.get("content"),
                        "tool_calls": response["tool_calls"]
                    })

                    # Add tool results
                    messages.extend(tool_results)

                else:
                    # No more tool calls, parse final response
                    return self._parse_final_response(response.get("content", ""), query)

            except Exception as e:
                logger.error(f"Error in tool calling round {round_num}: {e}")
                break

        # Fallback if tool calling fails
        logger.warning("Tool calling exhausted or failed, returning minimal context")
        return AssembledContext(
            assembled_context="Unable to assemble context",
            key_points=[],
            relevant_message_ids=[],
            metadata={"error": "Context assembly failed"}
        )

    def _call_llm_with_tools(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Call the LLM with tool definitions.

        Returns dict with 'content' and optionally 'tool_calls'.
        """
        try:
            # Use the provider's native tool support
            response = self._llm_provider.generate_with_tools(
                messages=messages,
                tools=CONTEXT_TOOLS,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response
        except AttributeError:
            # Provider doesn't support tools, fall back to regular generation
            logger.warning("LLM provider doesn't support tools, using fallback")
            content = self._call_llm(messages)
            return {"content": content, "tool_calls": None}

    def _parse_final_response(self, content: str, query: str) -> AssembledContext:
        """Parse the final LLM response into AssembledContext."""
        try:
            data = self._parse_json_response(content)

            return AssembledContext(
                assembled_context=data.get("relevant_context", ""),
                key_points=data.get("key_points", []),
                relevant_message_ids=[],  # Already summarized by agent
                metadata={
                    "context_source": data.get("context_source", "unknown"),
                    "confidence": data.get("confidence", "medium"),
                    "query": query
                }
            )
        except Exception as e:
            logger.warning(f"Failed to parse context response: {e}")
            # Return content as-is if JSON parsing fails
            return AssembledContext(
                assembled_context=content[:1000] if content else "",
                key_points=[],
                relevant_message_ids=[],
                metadata={"parse_error": str(e)}
            )
