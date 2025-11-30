"""
Summarization Agent for compressing threads into summaries.

Generates concise summaries from conversation threads, extracting
key facts, topics, entities, and sentiment for efficient retrieval.
"""
import uuid
from typing import List, Optional, TYPE_CHECKING
from datetime import datetime, timezone

from .base_agent import BaseAgent
from ..core.schemas import (
    Message,
    ThreadSummary,
    MetadataSchema,
    DEFAULT_METADATA_SCHEMA
)
from ..utils.logger import get_logger

if TYPE_CHECKING:
    from ..llm import BaseLLMProvider

logger = get_logger(__name__)


SUMMARIZATION_PROMPT = """Summarize this conversation thread. Extract the following information:

1. **Summary**: A concise summary of the conversation (2-3 sentences max)
2. **Key Facts**: Important facts mentioned (bullet points, max 5)
3. **Topics**: Main topics discussed (from the allowed list)
4. **Categories**: Categories this conversation falls into (from the allowed list)
5. **Entities**: Important entities mentioned (order IDs, dates, product names, user requests)
6. **Sentiment**: Overall sentiment of the conversation (positive/negative/neutral/mixed)

{schema_list}

Conversation ({message_count} messages):
{messages}

Respond ONLY with valid JSON in this exact format:
{{
    "summary": "Concise summary of the conversation...",
    "key_facts": ["fact 1", "fact 2"],
    "topics": ["topic1", "topic2"],
    "categories": ["category1"],
    "entities": {{
        "order_ids": ["#12345"],
        "dates": ["2024-03-15"],
        "products": ["Product Name"],
        "other": ["any other important entities"]
    }},
    "overall_sentiment": "neutral"
}}"""


class SummarizationAgent(BaseAgent):
    """
    Agent for compressing conversation threads into summaries.

    Uses LLM to generate intelligent summaries that preserve key information
    while reducing storage and speeding up context retrieval.

    Example:
        >>> from mindcore.llm import create_provider, ProviderType
        >>> provider = create_provider(ProviderType.OPENAI, api_key="sk-...")
        >>>
        >>> agent = SummarizationAgent(provider)
        >>> summary = agent.summarize_thread(
        ...     messages=[...],
        ...     thread_id="thread123",
        ...     user_id="user456"
        ... )
        >>> print(summary.summary)
        "User inquired about order #12345 delivery status..."
    """

    def __init__(
        self,
        llm_provider: "BaseLLMProvider",
        temperature: float = 0.2,
        max_tokens: int = 1500,
        metadata_schema: Optional[MetadataSchema] = None
    ):
        """
        Initialize summarization agent.

        Args:
            llm_provider: LLM provider instance
            temperature: Temperature for generation (lower = more consistent)
            max_tokens: Maximum tokens in response
            metadata_schema: Schema for valid topics/categories
        """
        super().__init__(llm_provider, temperature, max_tokens)
        self.metadata_schema = metadata_schema or DEFAULT_METADATA_SCHEMA

    def process(
        self,
        messages: List[Message],
        thread_id: str,
        user_id: str,
        session_id: Optional[str] = None
    ) -> ThreadSummary:
        """
        Process messages and generate a thread summary.

        Args:
            messages: List of Message objects to summarize
            thread_id: Thread identifier
            user_id: User identifier
            session_id: Optional session identifier

        Returns:
            ThreadSummary object with generated summary
        """
        return self.summarize_thread(messages, thread_id, user_id, session_id)

    def summarize_thread(
        self,
        messages: List[Message],
        thread_id: str,
        user_id: str,
        session_id: Optional[str] = None
    ) -> ThreadSummary:
        """
        Generate a summary from a list of messages.

        Extracts:
        - Overall summary (1-3 paragraphs)
        - Key facts (bullet points)
        - Aggregated topics/categories
        - Important entities (order IDs, dates, names)
        - Overall sentiment

        Args:
            messages: List of Message objects to summarize
            thread_id: Thread identifier
            user_id: User identifier
            session_id: Optional session identifier

        Returns:
            ThreadSummary object with all extracted information
        """
        if not messages:
            logger.warning(f"No messages to summarize for thread {thread_id}")
            return self._create_empty_summary(thread_id, user_id, session_id)

        logger.info(f"Summarizing thread {thread_id} with {len(messages)} messages")

        # Sort messages by time
        sorted_messages = sorted(
            messages,
            key=lambda m: m.created_at or datetime.min.replace(tzinfo=timezone.utc)
        )

        # Format messages for the prompt
        formatted_messages = self._format_messages(sorted_messages)

        # Get schema list for the prompt
        schema_list = self.metadata_schema.to_prompt_list()

        # Build the prompt
        prompt = SUMMARIZATION_PROMPT.format(
            schema_list=schema_list,
            message_count=len(messages),
            messages=formatted_messages
        )

        # Call LLM
        try:
            llm_messages = [
                {"role": "system", "content": "You are a conversation summarization assistant. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ]

            response = self._call_llm(llm_messages, json_mode=True)
            data = self._parse_json_response(response)

            # Validate and filter topics/categories
            topics = self.metadata_schema.validate_topics(
                data.get("topics", [])
            )
            categories = self.metadata_schema.validate_categories(
                data.get("categories", [])
            )

            # Get time bounds
            first_msg = sorted_messages[0]
            last_msg = sorted_messages[-1]

            # Create summary
            summary = ThreadSummary(
                summary_id=str(uuid.uuid4()),
                user_id=user_id,
                thread_id=thread_id,
                session_id=session_id,
                summary=data.get("summary", ""),
                key_facts=data.get("key_facts", [])[:5],  # Limit to 5
                topics=topics,
                categories=categories,
                overall_sentiment=data.get("overall_sentiment", "neutral"),
                message_count=len(messages),
                first_message_at=first_msg.created_at,
                last_message_at=last_msg.created_at,
                entities=data.get("entities", {}),
                messages_deleted=False
            )

            logger.info(
                f"Generated summary for thread {thread_id}: "
                f"{len(summary.summary)} chars, {len(topics)} topics"
            )
            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary for thread {thread_id}: {e}")
            # Return a minimal summary on error
            return self._create_fallback_summary(
                messages, thread_id, user_id, session_id, str(e)
            )

    def _format_messages(self, messages: List[Message]) -> str:
        """Format messages for the LLM prompt."""
        formatted = []
        for msg in messages[:100]:  # Limit to prevent context overflow
            timestamp = ""
            if msg.created_at:
                timestamp = f" ({msg.created_at.strftime('%Y-%m-%d %H:%M')})"

            # Include relevant metadata hints
            meta_hints = []
            if msg.metadata and msg.metadata.topics:
                meta_hints.append(f"topics: {', '.join(msg.metadata.topics[:3])}")
            if msg.metadata and msg.metadata.intent:
                meta_hints.append(f"intent: {msg.metadata.intent}")

            meta_str = f" [{'; '.join(meta_hints)}]" if meta_hints else ""

            formatted.append(
                f"[{msg.role.value}]{timestamp}{meta_str}: {msg.raw_text[:1000]}"
            )

        return "\n".join(formatted)

    def _create_empty_summary(
        self,
        thread_id: str,
        user_id: str,
        session_id: Optional[str]
    ) -> ThreadSummary:
        """Create an empty summary for threads with no messages."""
        return ThreadSummary(
            summary_id=str(uuid.uuid4()),
            user_id=user_id,
            thread_id=thread_id,
            session_id=session_id,
            summary="No messages in this thread.",
            key_facts=[],
            topics=[],
            categories=[],
            overall_sentiment="neutral",
            message_count=0,
            messages_deleted=False
        )

    def _create_fallback_summary(
        self,
        messages: List[Message],
        thread_id: str,
        user_id: str,
        session_id: Optional[str],
        error: str
    ) -> ThreadSummary:
        """Create a fallback summary when LLM fails."""
        # Aggregate topics from message metadata
        all_topics = []
        all_categories = []
        for msg in messages:
            if msg.metadata:
                all_topics.extend(msg.metadata.topics)
                all_categories.extend(msg.metadata.categories)

        # Get unique topics/categories
        topics = list(set(all_topics))[:5]
        categories = list(set(all_categories))[:3]

        # Create basic summary from first/last messages
        first_text = messages[0].raw_text[:200] if messages else ""
        summary = f"Conversation with {len(messages)} messages. Started with: {first_text}..."

        sorted_msgs = sorted(
            messages,
            key=lambda m: m.created_at or datetime.min.replace(tzinfo=timezone.utc)
        )

        return ThreadSummary(
            summary_id=str(uuid.uuid4()),
            user_id=user_id,
            thread_id=thread_id,
            session_id=session_id,
            summary=summary,
            key_facts=[f"Summarization error: {error[:100]}"],
            topics=topics,
            categories=categories,
            overall_sentiment="neutral",
            message_count=len(messages),
            first_message_at=sorted_msgs[0].created_at if sorted_msgs else None,
            last_message_at=sorted_msgs[-1].created_at if sorted_msgs else None,
            entities={},
            messages_deleted=False
        )

    def summarize_multiple_threads(
        self,
        threads_data: List[dict],
        db_manager
    ) -> List[ThreadSummary]:
        """
        Summarize multiple threads.

        Args:
            threads_data: List of dicts with thread_id, user_id keys
            db_manager: Database manager to fetch messages

        Returns:
            List of ThreadSummary objects
        """
        summaries = []
        for thread_info in threads_data:
            thread_id = thread_info['thread_id']
            user_id = thread_info['user_id']

            # Fetch messages
            messages = db_manager.fetch_recent_messages(
                user_id=user_id,
                thread_id=thread_id,
                limit=200  # Get more messages for summarization
            )

            if messages:
                summary = self.summarize_thread(
                    messages=messages,
                    thread_id=thread_id,
                    user_id=user_id
                )
                summaries.append(summary)

        return summaries
