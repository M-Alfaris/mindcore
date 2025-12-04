"""Smart Context Agent with Tool Calling.

Single LLM call that uses tools to fetch relevant context.
Uses VocabularyManager for controlled vocabulary in tool parameters.

This is the PRIMARY context retrieval method for Mindcore.

Handles early conversation scenarios gracefully by detecting when there's
insufficient history and returning appropriate context without unnecessary
LLM calls.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mindcore.core.schemas import AssembledContext, Message, ThreadSummary, UserPreferences
from mindcore.core.vocabulary import VocabularyManager, get_vocabulary
from mindcore.utils.logger import LogCategory, get_logger

from .base_agent import BaseAgent


if TYPE_CHECKING:
    from mindcore.llm import BaseLLMProvider

logger = get_logger(__name__, category=LogCategory.CONTEXT)


# Thresholds for early conversation detection
MIN_MESSAGES_FOR_CONTEXT = 2  # Minimum messages before attempting context assembly
MIN_MESSAGES_FOR_HISTORY_SEARCH = 5  # Minimum messages before searching history


@dataclass
class ContextTools:
    """Tools available to the SmartContextAgent.

    These are callbacks that the agent can invoke to fetch data.
    """

    get_recent_messages: Callable[[str, str, int], list[Message]]
    search_history: Callable[
        [str, str | None, list[str], list[str], str | None, int], list[Message]
    ]
    get_session_metadata: Callable[[str, str], dict[str, Any]]
    # New tools for enhanced context
    get_historical_summaries: Callable[[str, list[str] | None, int], list[ThreadSummary]] | None = (
        None
    )
    get_user_preferences: Callable[[str], UserPreferences] | None = None
    update_user_preference: Callable[[str, str, Any, str], tuple] | None = None
    # External connector tools
    lookup_external_data: Callable[[str, list[str], dict[str, Any]], list[Any]] | None = None


def _build_context_tools(vocabulary: VocabularyManager) -> list[dict[str, Any]]:
    """Build tool definitions with vocabulary-constrained parameters.

    Args:
        vocabulary: VocabularyManager instance for enum constraints.

    Returns:
        List of tool definitions for OpenAI function calling.
    """
    topics = vocabulary.get_topics()
    categories = vocabulary.get_categories()
    intents = vocabulary.get_intents()

    return [
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
                            "default": 10,
                        }
                    },
                    "required": [],
                },
            },
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
                            "items": {
                                "type": "string",
                                "enum": topics,  # Constrained to vocabulary
                            },
                            "description": "Topics to search for (must be from predefined list)",
                        },
                        "categories": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": categories,  # Constrained to vocabulary
                            },
                            "description": "Categories to filter by (must be from predefined list)",
                        },
                        "intent": {
                            "type": "string",
                            "enum": intents,  # Constrained to vocabulary
                            "description": "Intent type to filter by",
                        },
                        "search_current_thread_only": {
                            "type": "boolean",
                            "description": "If true, only search within current thread. If false, search across all user's threads.",
                            "default": False,
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of historical messages to retrieve (default: 20)",
                            "default": 20,
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_session_metadata",
                "description": "Get aggregated metadata about the current session (topics discussed, categories, intents). Useful for understanding the overall context of the conversation.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_historical_summaries",
                "description": "Get summaries of past conversation threads. Use when user references older conversations or you need context from previous sessions that have been summarized.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topics": {
                            "type": "array",
                            "items": {"type": "string", "enum": topics},
                            "description": "Topics to filter summaries by",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of summaries to retrieve (default: 5)",
                            "default": 5,
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_user_preferences",
                "description": "Get user's preferences (language, timezone, interests, goals, communication style). Use to personalize responses.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_user_preference",
                "description": "Update a user's preference when they explicitly request it. Only works for amendable fields like language, timezone, interests, goals, communication_style, preferred_name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "field": {
                            "type": "string",
                            "enum": [
                                "language",
                                "timezone",
                                "communication_style",
                                "interests",
                                "goals",
                                "preferred_name",
                            ],
                            "description": "The preference field to update",
                        },
                        "value": {"description": "The new value for the field"},
                        "action": {
                            "type": "string",
                            "enum": ["set", "add", "remove"],
                            "description": "For list fields (interests, goals): 'add' or 'remove' item. For others: 'set' value.",
                            "default": "set",
                        },
                    },
                    "required": ["field", "value"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_external_data",
                "description": "Fetch data from external systems (orders, billing, etc.) based on topics. Use when user asks about their orders, payments, subscriptions, deliveries, or other business data. This provides READ-ONLY access to external systems.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topics": {
                            "type": "array",
                            "items": {"type": "string", "enum": topics},
                            "description": "Topics to look up (e.g., 'orders', 'billing', 'subscription', 'delivery')",
                        },
                        "context": {
                            "type": "object",
                            "description": "Extracted entities like order_id, invoice_id, dates. Pass relevant IDs mentioned in conversation.",
                        },
                    },
                    "required": ["topics"],
                },
            },
        },
    ]


class SmartContextAgent(BaseAgent):
    """Intelligent context assembly agent using tool calling.

    This is the PRIMARY context retrieval method for Mindcore.

    Instead of separate query analysis and context assembly steps,
    this agent uses a single LLM call with tools to:
    1. Understand what context is needed
    2. Fetch relevant data via tool calls
    3. Assemble the final context

    Uses VocabularyManager to ensure all topic/category/intent parameters
    are constrained to the controlled vocabulary.

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
        vocabulary: VocabularyManager | None = None,
        max_tool_rounds: int = 3,
    ):
        """Initialize smart context agent.

        Args:
            llm_provider: LLM provider instance (should support tool calling)
            context_tools: Callbacks for fetching context data
            temperature: Temperature for generation (lower for consistency)
            max_tokens: Maximum tokens in response
            vocabulary: VocabularyManager for vocabulary constraints. If None, uses global.
            max_tool_rounds: Maximum rounds of tool calling (default: 3)
        """
        super().__init__(llm_provider, temperature, max_tokens)
        self.tools = context_tools
        self.vocabulary = vocabulary or get_vocabulary()
        self.max_tool_rounds = max_tool_rounds

        # Build tool definitions with vocabulary constraints
        self._tool_definitions = _build_context_tools(self.vocabulary)
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create system prompt for context assembly."""
        vocab_prompt = self.vocabulary.to_prompt_list()

        return f"""You are a context assembly agent. Your task is to gather relevant context for an AI assistant to answer a user's query.

You have access to tools that fetch conversation history, summaries, user preferences, and external data. Use them intelligently:

1. **get_recent_messages**: Get recent messages from the current thread. Use this for ongoing conversation context.

2. **search_history**: Search historical messages by topics, categories, or intent. Use this when:
   - User references past conversations ("last time", "before", "previously")
   - User asks about something that might have been discussed earlier
   - You need broader context beyond the current thread

3. **get_session_metadata**: Get aggregated info about current session topics. Use this to understand what's been discussed.

4. **get_historical_summaries**: Get summaries of older conversation threads. Use this when:
   - User references conversations from a while ago
   - You need context from multiple past sessions
   - Recent history search doesn't have enough information

5. **get_user_preferences**: Get user's preferences (language, timezone, interests, etc.). Use this to:
   - Personalize responses based on user preferences
   - Understand user's communication style preference
   - Include relevant context about the user

6. **update_user_preference**: Update user preferences when they explicitly request it. Use this when:
   - User says "remember that I prefer..." or "change my preference to..."
   - User asks to update their language, timezone, interests, or goals
   - ONLY use when user explicitly requests a preference change

7. **lookup_external_data**: Fetch data from external business systems. Use this when:
   - User asks about their orders, deliveries, or shipping status
   - User asks about billing, invoices, payments, or subscriptions
   - User needs information from external systems (CRM, orders, billing, etc.)
   - Pass relevant IDs (order numbers, invoice numbers) in the context parameter
   - This tool provides READ-ONLY access to external systems

VOCABULARY CONSTRAINTS:
{vocab_prompt}

IMPORTANT: When using search_history or lookup_external_data, you MUST use topics and categories from the Available lists above.

After gathering context, respond with a JSON object:
{{
    "relevant_context": "A concise summary of relevant information from the fetched messages",
    "key_points": ["Important point 1", "Important point 2", ...],
    "context_source": "recent|historical|summaries|preferences|external|multiple",
    "confidence": "high|medium|low",
    "user_preferences_applied": true|false,
    "preference_updates": [],
    "external_data_fetched": []
}}

Guidelines:
- Be selective - don't fetch everything, only what's relevant to the query
- If the query is simple and recent context is enough, don't search history
- If the query references past interactions, try summaries first (more efficient), then search_history
- Use get_user_preferences when personalizing responses would help
- Only update_user_preference when user explicitly requests a change
- Use lookup_external_data when user asks about orders, billing, or other business data
- When using lookup_external_data, extract any relevant IDs from the conversation first
- Summarize the context concisely - the main AI doesn't need raw messages
- Focus on information that helps answer the user's query"""

    def _execute_tool(
        self, tool_name: str, arguments: dict[str, Any], user_id: str, thread_id: str
    ) -> str:
        """Execute a tool and return the result as a string."""
        try:
            if tool_name == "get_recent_messages":
                limit = arguments.get("limit", 10)
                messages = self.tools.get_recent_messages(user_id, thread_id, min(limit, 50))
                result = self._format_messages(messages)
                logger.debug(
                    "Tool executed: get_recent_messages", message_count=len(messages), limit=limit
                )
                return result

            if tool_name == "search_history":
                # Validate topics/categories against vocabulary
                raw_topics = arguments.get("topics", [])
                raw_categories = arguments.get("categories", [])
                raw_intent = arguments.get("intent")

                topics = self.vocabulary.validate_topics(raw_topics)
                categories = self.vocabulary.validate_categories(raw_categories)
                intent = self.vocabulary.resolve_intent(raw_intent) if raw_intent else None

                current_only = arguments.get("search_current_thread_only", False)
                limit = arguments.get("limit", 20)

                search_thread_id = thread_id if current_only else None
                messages = self.tools.search_history(
                    user_id, search_thread_id, topics, categories, intent, limit
                )
                result = self._format_messages(messages)
                logger.debug(
                    "Tool executed: search_history",
                    message_count=len(messages),
                    topics=topics,
                    categories=categories,
                    intent=intent,
                )
                return result

            if tool_name == "get_session_metadata":
                metadata = self.tools.get_session_metadata(user_id, thread_id)
                logger.debug("Tool executed: get_session_metadata", has_metadata=bool(metadata))
                return json.dumps(metadata, indent=2)

            if tool_name == "get_historical_summaries":
                if self.tools.get_historical_summaries is None:
                    logger.debug("Tool not available: get_historical_summaries")
                    return "Historical summaries feature is not enabled."
                raw_topics = arguments.get("topics")
                topics = self.vocabulary.validate_topics(raw_topics) if raw_topics else None
                limit = arguments.get("limit", 5)
                summaries = self.tools.get_historical_summaries(user_id, topics, limit)
                logger.debug(
                    "Tool executed: get_historical_summaries",
                    summary_count=len(summaries) if summaries else 0,
                )
                return self._format_summaries(summaries)

            if tool_name == "get_user_preferences":
                if self.tools.get_user_preferences is None:
                    logger.debug("Tool not available: get_user_preferences")
                    return "User preferences feature is not enabled."
                prefs = self.tools.get_user_preferences(user_id)
                logger.debug(
                    "Tool executed: get_user_preferences", has_preferences=prefs is not None
                )
                return self._format_preferences(prefs)

            if tool_name == "update_user_preference":
                if self.tools.update_user_preference is None:
                    logger.debug("Tool not available: update_user_preference")
                    return "User preference updates are not enabled."
                field = arguments.get("field")
                value = arguments.get("value")
                action = arguments.get("action", "set")
                success, message = self.tools.update_user_preference(user_id, field, value, action)
                logger.debug(
                    "Tool executed: update_user_preference",
                    field=field,
                    action=action,
                    success=success,
                )
                return json.dumps({"success": success, "message": message})

            if tool_name == "lookup_external_data":
                if self.tools.lookup_external_data is None:
                    logger.debug("Tool not available: lookup_external_data")
                    return "External data lookup is not enabled. Configure connectors to enable."
                raw_topics = arguments.get("topics", [])
                topics = self.vocabulary.validate_topics(raw_topics)
                context = arguments.get("context", {})
                results = self.tools.lookup_external_data(user_id, topics, context)
                logger.debug(
                    "Tool executed: lookup_external_data",
                    result_count=len(results) if results else 0,
                    topics=topics,
                )
                return self._format_external_results(results)

            logger.warning("Unknown tool requested", tool_name=tool_name)
            return f"Unknown tool: {tool_name}"

        except Exception as e:
            logger.exception(
                "Tool execution error",
                tool=tool_name,
                error=str(e),
                error_type=type(e).__name__,
                arguments=arguments,
            )
            return f"Error executing {tool_name}: {e!s}"

    def _format_messages(self, messages: list[Message]) -> str:
        """Format messages for the LLM."""
        if not messages:
            return "No messages found."

        formatted = []
        for msg in messages[:30]:  # Limit to prevent context overflow
            meta_info = ""
            if msg.metadata and msg.metadata.topics:
                meta_info = f" [topics: {', '.join(msg.metadata.topics)}]"

            formatted.append(f"[{msg.role.value}]{meta_info}: {msg.raw_text[:500]}")

        return "\n".join(formatted)

    def _format_summaries(self, summaries: list[ThreadSummary]) -> str:
        """Format thread summaries for the LLM."""
        if not summaries:
            return "No historical summaries found."

        formatted = []
        for summary in summaries[:10]:  # Limit to prevent context overflow
            date_str = ""
            if summary.last_message_at:
                date_str = f" (last activity: {summary.last_message_at.strftime('%Y-%m-%d')})"

            topics_str = f" [topics: {', '.join(summary.topics)}]" if summary.topics else ""

            formatted.append(
                f"Thread {summary.thread_id}{date_str}{topics_str}:\n"
                f"  Summary: {summary.summary[:500]}\n"
                f"  Key facts: {'; '.join(summary.key_facts[:3]) if summary.key_facts else 'None'}"
            )

        return "\n\n".join(formatted)

    def _format_preferences(self, prefs: UserPreferences) -> str:
        """Format user preferences for the LLM."""
        if not prefs:
            return "No user preferences found."

        parts = [f"User Preferences for {prefs.user_id}:"]

        if prefs.preferred_name:
            parts.append(f"  Preferred name: {prefs.preferred_name}")
        parts.append(f"  Language: {prefs.language}")
        parts.append(f"  Timezone: {prefs.timezone}")
        parts.append(f"  Communication style: {prefs.communication_style}")

        if prefs.interests:
            parts.append(f"  Interests: {', '.join(prefs.interests)}")
        if prefs.goals:
            parts.append(f"  Goals: {', '.join(prefs.goals)}")
        if prefs.custom_context:
            for key, value in prefs.custom_context.items():
                parts.append(f"  {key}: {value}")

        return "\n".join(parts)

    def _format_external_results(self, results: list[Any]) -> str:
        """Format external connector results for the LLM."""
        if not results:
            return "No external data found."

        formatted = []
        for result in results:
            # Handle ConnectorResult objects
            if hasattr(result, "to_context_string"):
                formatted.append(result.to_context_string())
            elif hasattr(result, "source") and hasattr(result, "data"):
                # Duck typing for ConnectorResult-like objects
                if result.error:
                    formatted.append(f"[{result.source}] Error: {result.error}")
                elif result.data:
                    formatted.append(f"[{result.source}] Data retrieved:")
                    data_str = json.dumps(result.data, indent=2, default=str)
                    # Truncate if too long
                    if len(data_str) > 1000:
                        data_str = data_str[:1000] + "..."
                    formatted.append(data_str)
                else:
                    formatted.append(f"[{result.source}] No data found.")
            elif isinstance(result, dict):
                formatted.append(json.dumps(result, indent=2, default=str))
            else:
                formatted.append(str(result))

        return "\n\n".join(formatted)

    def process(
        self, query: str, user_id: str, thread_id: str, additional_context: str | None = None
    ) -> AssembledContext:
        """Assemble context for a query using tool calling.

        Args:
            query: User's query that needs context
            user_id: User identifier
            thread_id: Thread identifier
            additional_context: Optional additional context to include

        Returns:
            AssembledContext with relevant information
        """
        logger.debug(
            "SmartContextAgent processing query",
            query_preview=query[:100],
            user_id=user_id,
            thread_id=thread_id,
        )

        # Check conversation state before expensive LLM calls
        conversation_state = self._assess_conversation_state(user_id, thread_id)

        if conversation_state["is_early_conversation"]:
            return self._handle_early_conversation(
                query=query,
                user_id=user_id,
                thread_id=thread_id,
                state=conversation_state,
                additional_context=additional_context,
            )

        # Build initial messages for tool calling
        messages = [{"role": "system", "content": self.system_prompt}]

        user_content = f"User query: {query}"
        if additional_context:
            user_content += f"\n\nAdditional context: {additional_context}"

        # Add conversation state hint to help LLM make better tool choices
        if conversation_state["message_count"] < MIN_MESSAGES_FOR_HISTORY_SEARCH:
            user_content += (
                f"\n\nNote: This is a relatively new conversation with only "
                f"{conversation_state['message_count']} messages. "
                f"Focus on recent context rather than searching history."
            )

        messages.append({"role": "user", "content": user_content})

        # Tool calling loop
        for round_num in range(self.max_tool_rounds):
            try:
                logger.debug(
                    "Tool calling round started",
                    round=round_num + 1,
                    max_rounds=self.max_tool_rounds,
                )

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
                                logger.warning(
                                    "Failed to parse tool arguments", tool=tool_name, error=str(je)
                                )
                                arguments = {}

                            logger.debug("Executing tool", tool=tool_name, arguments=arguments)

                            result = self._execute_tool(tool_name, arguments, user_id, thread_id)
                            tool_results.append(
                                {"tool_call_id": tool_call["id"], "role": "tool", "content": result}
                            )
                        except (KeyError, TypeError) as e:
                            logger.exception(
                                "Malformed tool call", error=str(e), tool_call=tool_call
                            )
                            tool_results.append(
                                {
                                    "tool_call_id": tool_call.get("id", "unknown"),
                                    "role": "tool",
                                    "content": f"Error: malformed tool call - {e}",
                                }
                            )

                    # Add assistant message with tool calls
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response.get("content"),
                            "tool_calls": response["tool_calls"],
                        }
                    )

                    # Add tool results
                    messages.extend(tool_results)

                else:
                    # No more tool calls, parse final response
                    logger.debug(
                        "Tool calling complete, parsing response", rounds_used=round_num + 1
                    )
                    return self._parse_final_response(response.get("content", ""), query)

            except Exception as e:
                logger.exception(
                    "Error in tool calling round",
                    round=round_num + 1,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                break

        # Fallback if tool calling fails - provide helpful context
        return self._create_fallback_context(
            query=query,
            reason="tool_calling_exhausted",
            message_count=conversation_state.get("message_count", 0),
        )

    def _assess_conversation_state(self, user_id: str, thread_id: str) -> dict[str, Any]:
        """Assess the current conversation state to determine context strategy.

        Returns dict with:
        - is_early_conversation: True if not enough history for full context
        - message_count: Number of messages in thread
        - has_enriched_messages: Whether messages have been enriched
        - recent_messages: List of recent messages (for early conversation handling)
        """
        try:
            # Get recent messages to assess state
            recent_messages = self.tools.get_recent_messages(user_id, thread_id, 10)
            message_count = len(recent_messages)

            # Check if messages have been enriched (have topics)
            enriched_count = sum(
                1
                for msg in recent_messages
                if msg.metadata and msg.metadata.topics and len(msg.metadata.topics) > 0
            )
            has_enriched_messages = enriched_count > 0

            is_early = message_count < MIN_MESSAGES_FOR_CONTEXT

            if is_early:
                logger.info(
                    "Early conversation detected",
                    message_count=message_count,
                    min_required=MIN_MESSAGES_FOR_CONTEXT,
                    enriched_count=enriched_count,
                )

            return {
                "is_early_conversation": is_early,
                "message_count": message_count,
                "has_enriched_messages": has_enriched_messages,
                "enriched_count": enriched_count,
                "recent_messages": recent_messages,
            }

        except Exception as e:
            logger.exception(
                "Failed to assess conversation state",
                error=str(e),
                user_id=user_id,
                thread_id=thread_id,
            )
            # Return safe defaults - assume early conversation
            return {
                "is_early_conversation": True,
                "message_count": 0,
                "has_enriched_messages": False,
                "enriched_count": 0,
                "recent_messages": [],
                "error": str(e),
            }

    def _handle_early_conversation(
        self,
        query: str,
        user_id: str,
        thread_id: str,
        state: dict[str, Any],
        additional_context: str | None = None,
    ) -> AssembledContext:
        """Handle context assembly for early conversations without LLM tool calling.

        For new conversations with minimal history, we skip the expensive LLM
        tool-calling loop and directly return available context.
        """
        message_count = state.get("message_count", 0)
        recent_messages = state.get("recent_messages", [])

        logger.info(
            "Handling early conversation context",
            message_count=message_count,
            query_preview=query[:50],
        )

        # Build context from available messages
        context_parts = []

        if message_count == 0:
            context_parts.append(
                "This is the start of a new conversation. No prior context available."
            )
        else:
            context_parts.append(
                f"This is an early conversation with {message_count} message(s). "
                f"Limited historical context is available."
            )

            # Include recent messages as context
            if recent_messages:
                context_parts.append("\nRecent messages:")
                for msg in recent_messages[-5:]:  # Last 5 messages
                    role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                    text_preview = msg.raw_text[:200] if msg.raw_text else ""
                    context_parts.append(f"  [{role}]: {text_preview}")

        if additional_context:
            context_parts.append(f"\nAdditional context: {additional_context}")

        assembled = "\n".join(context_parts)

        return AssembledContext(
            assembled_context=assembled,
            key_points=[
                f"New conversation ({message_count} messages)",
                "Context will improve as conversation progresses",
            ],
            relevant_message_ids=[msg.message_id for msg in recent_messages],
            metadata={
                "context_source": "early_conversation",
                "confidence": "low" if message_count == 0 else "medium",
                "message_count": message_count,
                "is_early_conversation": True,
                "query": query,
            },
        )

    def _create_fallback_context(
        self, query: str, reason: str, message_count: int = 0
    ) -> AssembledContext:
        """Create a fallback context when tool calling fails or is exhausted.

        Provides a helpful response instead of a generic error.
        """
        if reason == "tool_calling_exhausted":
            if message_count < MIN_MESSAGES_FOR_HISTORY_SEARCH:
                # Not enough history - this is expected
                logger.info(
                    "Context assembly completed with limited history",
                    message_count=message_count,
                    reason=reason,
                )
                return AssembledContext(
                    assembled_context=(
                        f"Limited conversation history ({message_count} messages). "
                        f"Context will improve as the conversation continues."
                    ),
                    key_points=["Conversation history is still building"],
                    relevant_message_ids=[],
                    metadata={
                        "context_source": "fallback",
                        "confidence": "low",
                        "reason": reason,
                        "message_count": message_count,
                    },
                )
            # Unexpected failure with sufficient history
            logger.warning(
                "Tool calling exhausted with sufficient history",
                message_count=message_count,
                reason=reason,
            )
            return AssembledContext(
                assembled_context="Context assembly encountered an issue. Proceeding with available information.",
                key_points=[],
                relevant_message_ids=[],
                metadata={
                    "context_source": "fallback",
                    "confidence": "low",
                    "reason": reason,
                    "error": "Tool calling exhausted without producing final response",
                },
            )
        logger.error("Context assembly failed", reason=reason, message_count=message_count)
        return AssembledContext(
            assembled_context="Unable to assemble context due to an error.",
            key_points=[],
            relevant_message_ids=[],
            metadata={"context_source": "error", "confidence": "none", "error": reason},
        )

    def _call_llm_with_tools(self, messages: list[dict]) -> dict[str, Any]:
        """Call the LLM with tool definitions.

        Returns dict with 'content' and optionally 'tool_calls'.
        """
        try:
            # Use the provider's native tool support
            return self._llm_provider.generate_with_tools(
                messages=messages,
                tools=self._tool_definitions,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
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
                    "query": query,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to parse context response: {e}")
            # Return content as-is if JSON parsing fails
            return AssembledContext(
                assembled_context=content[:1000] if content else "",
                key_points=[],
                relevant_message_ids=[],
                metadata={"parse_error": str(e)},
            )

    def refresh_vocabulary(self) -> None:
        """Refresh vocabulary and rebuild tool definitions."""
        self.vocabulary.refresh_from_external()
        self._tool_definitions = _build_context_tools(self.vocabulary)
        self.system_prompt = self._create_system_prompt()
        logger.info("SmartContextAgent vocabulary and tools refreshed")
