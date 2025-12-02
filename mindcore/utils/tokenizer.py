"""
Tokenizer utilities for text processing.

Supports accurate token counting via tiktoken (optional) with fallback
to character-based estimation.
"""

from typing import List, Optional

# Try to import tiktoken for accurate token counting
_tiktoken = None
_tiktoken_encoding = None


def _get_tiktoken_encoding():
    """Lazily load tiktoken encoder."""
    global _tiktoken, _tiktoken_encoding
    if _tiktoken is None:
        try:
            import tiktoken

            _tiktoken = tiktoken
            # Use cl100k_base (GPT-4, GPT-3.5-turbo)
            _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            _tiktoken = False  # Mark as unavailable
    return _tiktoken_encoding if _tiktoken else None


def simple_tokenize(text: str) -> List[str]:
    """
    Simple word tokenization.

    Args:
        text: Input text.

    Returns:
        List of tokens.
    """
    # Basic tokenization by splitting on whitespace and punctuation
    import re

    tokens = re.findall(r"\w+", text.lower())
    return tokens


def estimate_tokens(text: str, accurate: bool = True) -> int:
    """
    Estimate the number of tokens in text.

    Uses tiktoken for accurate counting if available (GPT-4/3.5 compatible),
    otherwise falls back to character-based approximation.

    Args:
        text: Input text.
        accurate: If True, use tiktoken when available. If False, always use approximation.

    Returns:
        Token count (exact with tiktoken, estimated otherwise).
    """
    if not text:
        return 0

    if accurate:
        encoding = _get_tiktoken_encoding()
        if encoding:
            return len(encoding.encode(text))

    # Fallback: rough estimation (4 characters per token)
    return max(1, len(text) // 4)


def count_tokens(text: str) -> int:
    """
    Count tokens accurately using tiktoken.

    Raises ImportError if tiktoken not installed.

    Args:
        text: Input text.

    Returns:
        Exact token count.
    """
    encoding = _get_tiktoken_encoding()
    if not encoding:
        raise ImportError(
            "tiktoken not installed for accurate token counting. "
            "Install with: pip install tiktoken"
        )
    return len(encoding.encode(text))


def has_accurate_tokenizer() -> bool:
    """Check if accurate token counting is available."""
    return _get_tiktoken_encoding() is not None


def truncate_text(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximately max_tokens.

    Uses tiktoken for accurate truncation if available.

    Args:
        text: Input text.
        max_tokens: Maximum number of tokens.

    Returns:
        Truncated text.
    """
    encoding = _get_tiktoken_encoding()

    if encoding:
        # Accurate truncation with tiktoken
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens) + "..."
    else:
        # Fallback to character-based
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    Extract top keywords from text (simple frequency-based).

    Args:
        text: Input text.
        top_n: Number of top keywords to extract.

    Returns:
        List of keywords.
    """
    from collections import Counter

    # Tokenize
    tokens = simple_tokenize(text)

    # Remove common stopwords
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
    }

    filtered_tokens = [t for t in tokens if t not in stopwords and len(t) > 2]

    # Count frequencies
    counter = Counter(filtered_tokens)

    # Return top N
    return [word for word, count in counter.most_common(top_n)]
