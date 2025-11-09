"""
Pluggable importance scoring algorithms for Mindcore.

This module provides different algorithms for calculating message importance.
Users can choose an algorithm or implement their own custom one.

Usage:
    from mindcore.importance import TFIDFImportance, KeywordImportance

    # Use TF-IDF based importance
    importance_scorer = TFIDFImportance()
    score = importance_scorer.calculate(message_text, conversation_history)

    # Or use keyword-based importance
    importance_scorer = KeywordImportance(keywords=["urgent", "critical", "important"])
    score = importance_scorer.calculate(message_text)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from collections import Counter
import re


class ImportanceAlgorithm(ABC):
    """
    Abstract base class for importance scoring algorithms.

    Custom algorithms should inherit from this class and implement
    the calculate() method.
    """

    @abstractmethod
    def calculate(self, text: str, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> float:
        """
        Calculate importance score for a message.

        Args:
            text: Message text
            metadata: Optional metadata dict
            **kwargs: Additional context (e.g., conversation_history)

        Returns:
            Importance score between 0.0 and 1.0
        """
        pass


class LLMBasedImportance(ImportanceAlgorithm):
    """
    Default LLM-based importance (from metadata enrichment).

    This uses the importance score provided by the LLM during
    metadata enrichment.
    """

    def calculate(self, text: str, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> float:
        """
        Extract importance from LLM-generated metadata.

        Args:
            text: Message text (not used, importance comes from metadata)
            metadata: Metadata dict containing 'importance' key

        Returns:
            Importance score from metadata or 0.5 as default
        """
        if metadata and 'importance' in metadata:
            return float(metadata['importance'])
        return 0.5  # Default neutral importance


class KeywordImportance(ImportanceAlgorithm):
    """
    Keyword-based importance scoring.

    Assigns higher importance to messages containing specific keywords.
    """

    def __init__(
        self,
        high_importance_keywords: List[str] = None,
        low_importance_keywords: List[str] = None
    ):
        """
        Initialize keyword-based importance scorer.

        Args:
            high_importance_keywords: Keywords that increase importance
            low_importance_keywords: Keywords that decrease importance
        """
        self.high_keywords = high_importance_keywords or [
            'urgent', 'critical', 'important', 'asap', 'deadline',
            'required', 'must', 'essential', 'priority', 'emergency'
        ]
        self.low_keywords = low_importance_keywords or [
            'maybe', 'perhaps', 'casual', 'fyi', 'just wondering',
            'quick question', 'no rush', 'whenever'
        ]

    def calculate(self, text: str, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> float:
        """
        Calculate importance based on keyword presence.

        Args:
            text: Message text
            metadata: Not used
            **kwargs: Not used

        Returns:
            Importance score between 0.0 and 1.0
        """
        text_lower = text.lower()

        # Count high and low importance keywords
        high_count = sum(1 for keyword in self.high_keywords if keyword in text_lower)
        low_count = sum(1 for keyword in self.low_keywords if keyword in text_lower)

        # Base score
        base_score = 0.5

        # Adjust based on keywords
        high_boost = min(high_count * 0.15, 0.4)
        low_penalty = min(low_count * 0.1, 0.3)

        score = base_score + high_boost - low_penalty

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))


class LengthBasedImportance(ImportanceAlgorithm):
    """
    Length-based importance scoring.

    Assumes longer messages are more important (up to a point).
    """

    def __init__(self, min_length: int = 10, optimal_length: int = 200):
        """
        Initialize length-based scorer.

        Args:
            min_length: Minimum length for non-trivial messages
            optimal_length: Length at which importance peaks
        """
        self.min_length = min_length
        self.optimal_length = optimal_length

    def calculate(self, text: str, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> float:
        """
        Calculate importance based on message length.

        Args:
            text: Message text
            metadata: Not used
            **kwargs: Not used

        Returns:
            Importance score between 0.0 and 1.0
        """
        length = len(text)

        if length < self.min_length:
            # Very short messages (greetings, etc.)
            return 0.2

        if length < self.optimal_length:
            # Scale up to optimal length
            return 0.2 + (length / self.optimal_length) * 0.6

        if length < self.optimal_length * 2:
            # Peak importance at optimal length
            return 0.8

        # Very long messages might be less important (spam, etc.)
        return max(0.5, 0.8 - (length - self.optimal_length * 2) / 1000)


class SentimentBasedImportance(ImportanceAlgorithm):
    """
    Sentiment-based importance scoring.

    Assigns higher importance to emotionally charged messages.
    """

    def calculate(self, text: str, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> float:
        """
        Calculate importance based on sentiment intensity.

        Args:
            text: Message text
            metadata: Metadata dict containing sentiment
            **kwargs: Not used

        Returns:
            Importance score between 0.0 and 1.0
        """
        if not metadata or 'sentiment' not in metadata:
            return 0.5

        sentiment = metadata['sentiment']

        # Extract sentiment score
        if isinstance(sentiment, dict):
            score = sentiment.get('score', 0.5)
            overall = sentiment.get('overall', 'neutral')
        else:
            return 0.5

        # Neutral sentiment = lower importance
        if overall == 'neutral':
            return 0.4

        # Strong sentiment (very positive or negative) = higher importance
        # Score ranges from 0 to 1, where 0.5 is neutral
        intensity = abs(score - 0.5) * 2  # Convert to 0-1 range

        return 0.5 + intensity * 0.4


class CompositeImportance(ImportanceAlgorithm):
    """
    Composite importance scorer combining multiple algorithms.

    Calculates weighted average of multiple importance algorithms.
    """

    def __init__(self, algorithms: List[tuple] = None):
        """
        Initialize composite scorer.

        Args:
            algorithms: List of (algorithm, weight) tuples
                       Example: [(KeywordImportance(), 0.4), (LengthBasedImportance(), 0.3)]
        """
        self.algorithms = algorithms or [
            (LLMBasedImportance(), 0.5),
            (KeywordImportance(), 0.3),
            (LengthBasedImportance(), 0.2)
        ]

    def calculate(self, text: str, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> float:
        """
        Calculate weighted average importance.

        Args:
            text: Message text
            metadata: Optional metadata
            **kwargs: Additional context

        Returns:
            Weighted average importance score
        """
        total_weight = sum(weight for _, weight in self.algorithms)
        weighted_sum = 0.0

        for algorithm, weight in self.algorithms:
            try:
                score = algorithm.calculate(text, metadata, **kwargs)
                weighted_sum += score * weight
            except Exception:
                # If one algorithm fails, skip it
                total_weight -= weight
                continue

        if total_weight == 0:
            return 0.5

        return weighted_sum / total_weight


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_importance_algorithm(algorithm_name: str = "llm") -> ImportanceAlgorithm:
    """
    Get importance algorithm by name.

    Args:
        algorithm_name: Name of algorithm
            - "llm": LLM-based (default)
            - "keyword": Keyword-based
            - "length": Length-based
            - "sentiment": Sentiment-based
            - "composite": Composite (combines all)

    Returns:
        ImportanceAlgorithm instance

    Example:
        >>> scorer = get_importance_algorithm("keyword")
        >>> importance = scorer.calculate("Urgent: Please review ASAP")
    """
    algorithms = {
        "llm": LLMBasedImportance,
        "keyword": KeywordImportance,
        "length": LengthBasedImportance,
        "sentiment": SentimentBasedImportance,
        "composite": CompositeImportance,
    }

    if algorithm_name not in algorithms:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. "
            f"Choose from: {list(algorithms.keys())}"
        )

    return algorithms[algorithm_name]()


__all__ = [
    "ImportanceAlgorithm",
    "LLMBasedImportance",
    "KeywordImportance",
    "LengthBasedImportance",
    "SentimentBasedImportance",
    "CompositeImportance",
    "get_importance_algorithm",
]
