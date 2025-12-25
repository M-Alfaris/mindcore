"""Usage Detection and Automatic Reinforcement for FLR.

Analyzes LLM responses to detect which retrieved memories were actually used,
then automatically applies reinforcement signals. This creates an implicit
feedback loop that improves retrieval quality without explicit user feedback.

Flow:
    1. FLR retrieves memories for query
    2. LLM generates response using some memories
    3. UsageDetector analyzes response to find used memories
    4. Automatic reinforcement applied:
       - Used memories: +0.5 to +0.8 signal
       - Retrieved but unused: -0.1 to 0 signal
    5. Metadata effectiveness updated
    6. Future queries optimized

Example:
    detector = UsageDetector()

    # After LLM responds
    usage_result = detector.detect_usage(
        llm_response="Based on your order #12345, it shipped yesterday...",
        retrieved_memories=recall_result.memories,
    )

    # Auto-reinforce based on usage
    detector.auto_reinforce(usage_result, flr)

    # Get query optimization hints
    hints = detector.get_query_optimization_hints()
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mindcore.v2.flr.recall import FLR, Memory, RecallResult


@dataclass
class MemoryUsage:
    """Tracks whether and how a memory was used in LLM response."""

    memory_id: str
    memory_content: str
    memory_topics: list[str]
    memory_categories: list[str]

    # Usage detection results
    was_used: bool = False
    usage_confidence: float = 0.0  # 0-1, how confident we are it was used
    usage_type: str = "none"  # "direct_quote", "paraphrase", "entity_match", "topic_match"

    # Evidence
    matched_content: str = ""  # What in the response matched
    matched_entities: list[str] = field(default_factory=list)
    content_overlap_score: float = 0.0

    # Reinforcement to apply
    suggested_signal: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "was_used": self.was_used,
            "usage_confidence": self.usage_confidence,
            "usage_type": self.usage_type,
            "matched_content": self.matched_content[:100] if self.matched_content else "",
            "matched_entities": self.matched_entities,
            "content_overlap_score": self.content_overlap_score,
            "suggested_signal": self.suggested_signal,
        }


@dataclass
class UsageDetectionResult:
    """Result of analyzing LLM response for memory usage."""

    llm_response: str
    total_memories: int
    used_memories: list[MemoryUsage]
    unused_memories: list[MemoryUsage]

    # Aggregate stats
    usage_rate: float = 0.0  # What fraction of retrieved were used
    avg_usage_confidence: float = 0.0

    # Query context
    query_topics: list[str] = field(default_factory=list)
    session_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if self.total_memories > 0:
            self.usage_rate = len(self.used_memories) / self.total_memories
        if self.used_memories:
            self.avg_usage_confidence = sum(
                m.usage_confidence for m in self.used_memories
            ) / len(self.used_memories)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_memories": self.total_memories,
            "used_count": len(self.used_memories),
            "unused_count": len(self.unused_memories),
            "usage_rate": self.usage_rate,
            "avg_usage_confidence": self.avg_usage_confidence,
            "used_memories": [m.to_dict() for m in self.used_memories],
            "query_topics": self.query_topics,
            "timestamp": self.timestamp.isoformat(),
        }


class UsageDetector:
    """Detects which memories were used in LLM response.

    Uses multiple signals to determine usage:
    1. Content overlap (word/phrase matching)
    2. Entity matching (names, numbers, identifiers)
    3. Semantic indicators (topic relevance)
    4. Direct quotes or paraphrases

    Example:
        detector = UsageDetector()

        result = detector.detect_usage(
            llm_response="Your order #12345 shipped on Dec 20th.",
            retrieved_memories=recall_result.memories,
            query_topics=["orders", "shipping"],
        )

        print(f"Used {len(result.used_memories)} of {result.total_memories}")
    """

    def __init__(
        self,
        min_overlap_threshold: float = 0.15,
        entity_match_weight: float = 0.4,
        content_match_weight: float = 0.4,
        topic_match_weight: float = 0.2,
        usage_confidence_threshold: float = 0.3,
    ):
        """Initialize detector.

        Args:
            min_overlap_threshold: Minimum word overlap to consider
            entity_match_weight: Weight for entity matches in confidence
            content_match_weight: Weight for content matches
            topic_match_weight: Weight for topic matches
            usage_confidence_threshold: Minimum confidence to mark as "used"
        """
        self.min_overlap_threshold = min_overlap_threshold
        self.entity_match_weight = entity_match_weight
        self.content_match_weight = content_match_weight
        self.topic_match_weight = topic_match_weight
        self.usage_confidence_threshold = usage_confidence_threshold

        # Tracking for optimization
        self._usage_history: list[UsageDetectionResult] = []
        self._topic_usage_stats: dict[str, dict[str, int]] = {}  # topic -> {used, total}

    def detect_usage(
        self,
        llm_response: str,
        retrieved_memories: list[Memory],
        query_topics: list[str] | None = None,
        session_id: str | None = None,
    ) -> UsageDetectionResult:
        """Detect which memories were used in the LLM response.

        Args:
            llm_response: The LLM's generated response
            retrieved_memories: Memories that were retrieved for context
            query_topics: Topics used in the query
            session_id: Current session ID

        Returns:
            UsageDetectionResult with analysis
        """
        query_topics = query_topics or []
        used = []
        unused = []

        response_lower = llm_response.lower()
        response_words = set(self._tokenize(llm_response))

        for memory in retrieved_memories:
            usage = self._analyze_memory_usage(
                memory=memory,
                response=llm_response,
                response_lower=response_lower,
                response_words=response_words,
                query_topics=query_topics,
            )

            if usage.was_used:
                used.append(usage)
            else:
                unused.append(usage)

            # Update topic stats
            self._update_topic_stats(memory.topics, usage.was_used)

        result = UsageDetectionResult(
            llm_response=llm_response,
            total_memories=len(retrieved_memories),
            used_memories=used,
            unused_memories=unused,
            query_topics=query_topics,
            session_id=session_id,
        )

        # Store for history
        self._usage_history.append(result)
        if len(self._usage_history) > 1000:
            self._usage_history = self._usage_history[-1000:]

        return result

    def _analyze_memory_usage(
        self,
        memory: Memory,
        response: str,
        response_lower: str,
        response_words: set[str],
        query_topics: list[str],
    ) -> MemoryUsage:
        """Analyze if a single memory was used."""
        usage = MemoryUsage(
            memory_id=memory.memory_id,
            memory_content=memory.content,
            memory_topics=memory.topics,
            memory_categories=memory.categories,
        )

        confidence_components = []

        # 1. Content overlap analysis
        memory_words = set(self._tokenize(memory.content))
        if memory_words:
            overlap = len(response_words & memory_words)
            overlap_score = overlap / len(memory_words)
            usage.content_overlap_score = overlap_score

            if overlap_score >= self.min_overlap_threshold:
                confidence_components.append(
                    ("content", min(1.0, overlap_score * 2) * self.content_match_weight)
                )

        # 2. Entity matching (numbers, identifiers, names)
        entities = self._extract_entities(memory.content)
        matched_entities = []
        for entity in entities:
            if entity.lower() in response_lower:
                matched_entities.append(entity)

        usage.matched_entities = matched_entities
        if entities:
            entity_match_rate = len(matched_entities) / len(entities)
            if entity_match_rate > 0:
                confidence_components.append(
                    ("entity", entity_match_rate * self.entity_match_weight)
                )

        # 3. Direct quote detection
        direct_quote = self._find_direct_quote(memory.content, response)
        if direct_quote:
            usage.matched_content = direct_quote
            usage.usage_type = "direct_quote"
            confidence_components.append(("quote", 0.9))  # High confidence for quotes

        # 4. Topic relevance (memory topics mentioned in response)
        topic_matches = 0
        for topic in memory.topics:
            # Check if topic or related words appear in response
            if topic.lower() in response_lower:
                topic_matches += 1
            # Check for topic-related patterns
            elif self._topic_appears_in_response(topic, response_lower):
                topic_matches += 0.5

        if memory.topics and topic_matches > 0:
            topic_score = topic_matches / len(memory.topics)
            confidence_components.append(
                ("topic", min(1.0, topic_score) * self.topic_match_weight)
            )

        # Calculate final confidence
        if confidence_components:
            usage.usage_confidence = sum(c[1] for c in confidence_components)
            usage.usage_confidence = min(1.0, usage.usage_confidence)

            # Determine usage type if not already set
            if not usage.usage_type or usage.usage_type == "none":
                primary = max(confidence_components, key=lambda x: x[1])
                usage.usage_type = primary[0] + "_match"

        # Determine if used based on threshold
        usage.was_used = usage.usage_confidence >= self.usage_confidence_threshold

        # Calculate suggested signal
        if usage.was_used:
            # Positive signal scaled by confidence
            usage.suggested_signal = 0.3 + (usage.usage_confidence * 0.5)  # 0.3 to 0.8
        else:
            # Slight negative for retrieved but unused
            # Higher negative if we were confident in retrieval but LLM didn't use it
            usage.suggested_signal = -0.05  # Small negative

        return usage

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for word overlap."""
        # Remove punctuation and split
        words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        # Filter stop words and short words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                      'from', 'as', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'between', 'under', 'again', 'further',
                      'then', 'once', 'here', 'there', 'when', 'where', 'why',
                      'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                      'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                      'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                      'because', 'until', 'while', 'this', 'that', 'these', 'those',
                      'it', 'its', 'i', 'you', 'he', 'she', 'we', 'they', 'my',
                      'your', 'his', 'her', 'our', 'their', 'what', 'which', 'who'}
        return [w for w in words if w not in stop_words and len(w) > 2]

    def _extract_entities(self, text: str) -> list[str]:
        """Extract entities (numbers, identifiers, proper nouns)."""
        entities = []

        # Numbers and IDs (order numbers, dates, etc.)
        entities.extend(re.findall(r'\b\d+(?:\.\d+)?\b', text))
        entities.extend(re.findall(r'#\w+', text))
        entities.extend(re.findall(r'\b[A-Z]{2,}\d+\b', text))  # Like "ABC123"

        # Dates
        entities.extend(re.findall(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', text))
        entities.extend(re.findall(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?\b', text, re.I))

        # Email-like patterns
        entities.extend(re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text))

        # Capitalized phrases (potential names)
        entities.extend(re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text))

        return list(set(entities))

    def _find_direct_quote(self, memory_content: str, response: str, min_length: int = 15) -> str:
        """Find if response contains a direct quote from memory."""
        memory_lower = memory_content.lower()
        response_lower = response.lower()

        # Try to find longest matching substring
        words = memory_lower.split()
        for length in range(min(10, len(words)), 2, -1):
            for i in range(len(words) - length + 1):
                phrase = ' '.join(words[i:i + length])
                if len(phrase) >= min_length and phrase in response_lower:
                    return phrase

        return ""

    def _topic_appears_in_response(self, topic: str, response_lower: str) -> bool:
        """Check if topic or related concepts appear in response."""
        # Simple check - could be enhanced with synonyms/embeddings
        topic_lower = topic.lower().replace('_', ' ').replace('-', ' ')
        return topic_lower in response_lower

    def _update_topic_stats(self, topics: list[str], was_used: bool) -> None:
        """Update topic usage statistics."""
        for topic in topics:
            if topic not in self._topic_usage_stats:
                self._topic_usage_stats[topic] = {"used": 0, "total": 0}
            self._topic_usage_stats[topic]["total"] += 1
            if was_used:
                self._topic_usage_stats[topic]["used"] += 1

    def auto_reinforce(
        self,
        usage_result: UsageDetectionResult,
        flr: FLR,
        apply_unused_penalty: bool = True,
    ) -> dict[str, float]:
        """Automatically apply reinforcement based on usage detection.

        Args:
            usage_result: Result from detect_usage()
            flr: FLR instance to apply reinforcement to
            apply_unused_penalty: Whether to apply negative signal to unused

        Returns:
            Dict mapping memory_id to applied signal
        """
        applied = {}

        # Reinforce used memories
        for usage in usage_result.used_memories:
            score, _ = flr.reinforce_with_metadata_feedback(
                memory_id=usage.memory_id,
                signal=usage.suggested_signal,
                is_user_feedback=False,  # This is implicit feedback
                session_id=usage_result.session_id,
            )
            applied[usage.memory_id] = usage.suggested_signal

        # Apply penalty to unused (optional)
        if apply_unused_penalty:
            for usage in usage_result.unused_memories:
                if usage.suggested_signal < 0:
                    score, _ = flr.reinforce_with_metadata_feedback(
                        memory_id=usage.memory_id,
                        signal=usage.suggested_signal,
                        is_user_feedback=False,
                        session_id=usage_result.session_id,
                    )
                    applied[usage.memory_id] = usage.suggested_signal

        return applied

    def get_topic_effectiveness(self) -> dict[str, float]:
        """Get topic effectiveness based on usage history.

        Returns:
            Dict mapping topic to usage rate (0-1)
        """
        effectiveness = {}
        for topic, stats in self._topic_usage_stats.items():
            if stats["total"] >= 3:  # Minimum samples
                effectiveness[topic] = stats["used"] / stats["total"]
        return effectiveness

    def get_query_optimization_hints(self, min_samples: int = 5) -> dict[str, Any]:
        """Get hints for optimizing future queries.

        Returns:
            Optimization hints based on usage patterns
        """
        topic_eff = self.get_topic_effectiveness()

        # Separate into effective and ineffective
        effective_topics = [
            (t, e) for t, e in topic_eff.items()
            if e >= 0.5 and self._topic_usage_stats[t]["total"] >= min_samples
        ]
        ineffective_topics = [
            (t, e) for t, e in topic_eff.items()
            if e < 0.3 and self._topic_usage_stats[t]["total"] >= min_samples
        ]

        # Sort by effectiveness
        effective_topics.sort(key=lambda x: x[1], reverse=True)
        ineffective_topics.sort(key=lambda x: x[1])

        # Calculate overall usage rate
        total_used = sum(len(r.used_memories) for r in self._usage_history)
        total_retrieved = sum(r.total_memories for r in self._usage_history)
        overall_usage_rate = total_used / total_retrieved if total_retrieved > 0 else 0

        return {
            "effective_topics": effective_topics[:10],
            "ineffective_topics": ineffective_topics[:10],
            "overall_usage_rate": overall_usage_rate,
            "recommendation": self._generate_recommendation(
                effective_topics, ineffective_topics, overall_usage_rate
            ),
            "topic_stats": {
                t: {"used": s["used"], "total": s["total"], "rate": s["used"]/s["total"] if s["total"] > 0 else 0}
                for t, s in self._topic_usage_stats.items()
                if s["total"] >= min_samples
            },
        }

    def _generate_recommendation(
        self,
        effective: list[tuple[str, float]],
        ineffective: list[tuple[str, float]],
        usage_rate: float,
    ) -> str:
        """Generate natural language recommendation."""
        lines = []

        if usage_rate < 0.3:
            lines.append("Low usage rate detected. Consider:")
            lines.append("- Reducing number of retrieved memories (fewer, more relevant)")
            lines.append("- Improving topic matching precision")
        elif usage_rate > 0.7:
            lines.append("Good usage rate. System is retrieving relevant memories.")

        if effective:
            topics = ", ".join([f"'{t[0]}'" for t in effective[:5]])
            lines.append(f"Boost these topics in queries: {topics}")

        if ineffective:
            topics = ", ".join([f"'{t[0]}'" for t in ineffective[:3]])
            lines.append(f"Consider removing these topics from attention hints: {topics}")

        return "\n".join(lines) if lines else "Insufficient data for recommendations."

    def get_stats(self) -> dict[str, Any]:
        """Get detector statistics."""
        if not self._usage_history:
            return {"history_size": 0}

        return {
            "history_size": len(self._usage_history),
            "topics_tracked": len(self._topic_usage_stats),
            "avg_usage_rate": sum(r.usage_rate for r in self._usage_history) / len(self._usage_history),
            "recent_usage_rates": [r.usage_rate for r in self._usage_history[-10:]],
        }

    def clear_history(self) -> None:
        """Clear usage history and stats."""
        self._usage_history.clear()
        self._topic_usage_stats.clear()
