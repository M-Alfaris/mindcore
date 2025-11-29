"""
Tokenizer utilities for text processing.
"""
from typing import List


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
    tokens = re.findall(r'\w+', text.lower())
    return tokens


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in text.

    Uses a rough approximation: 1 token ≈ 4 characters for English text.

    Args:
        text: Input text.

    Returns:
        Estimated token count.
    """
    # Rough estimation: average of 4 characters per token
    return max(1, len(text) // 4)


def truncate_text(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximately max_tokens.

    Args:
        text: Input text.
        max_tokens: Maximum number of tokens.

    Returns:
        Truncated text.
    """
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
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his',
        'her', 'its', 'our', 'their'
    }

    filtered_tokens = [t for t in tokens if t not in stopwords and len(t) > 2]

    # Count frequencies
    counter = Counter(filtered_tokens)

    # Return top N
    return [word for word, count in counter.most_common(top_n)]


