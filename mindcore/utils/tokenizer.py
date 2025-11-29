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


from typing import Dict, Any, Optional


def extract_query_hints(text: str) -> Dict[str, Any]:
    """
    Extract keyword hints from a query for relevance search matching.

    This does NOT replace LLM enrichment - it only extracts simple keywords
    and patterns from the query to match against already-enriched message metadata.

    Purpose: Fast query-to-metadata matching for relevance search.
    - Messages are enriched with full metadata by the LLM EnrichmentAgent
    - This function extracts hints from the query to search those enriched messages
    - Uses regex/keywords only - NO LLM calls

    Args:
        text: Query text to extract hints from.

    Returns:
        Dict with:
        - keywords: List of keyword hints to match against message topics
        - intent_hint: Detected intent pattern or None
        - category_hints: List of category hints
    """
    import re

    text_lower = text.lower()

    # Extract keywords to match against enriched topics
    keywords = extract_keywords(text, top_n=10)

    # Detect intent based on patterns
    intent = None
    intent_patterns = {
        'ask_question': [
            r'\b(what|how|why|when|where|who|which|can you|could you|is there|are there)\b',
            r'\?$',
        ],
        'request_action': [
            r'\b(please|help|need|want|show|give|get|find|create|make|do|set|change|update|delete|remove)\b',
        ],
        'provide_info': [
            r'\b(here is|this is|i have|i found|the answer|solution|result)\b',
        ],
        'greeting': [
            r'^(hi|hello|hey|good morning|good afternoon|good evening)\b',
        ],
        'express_opinion': [
            r'\b(i think|i believe|in my opinion|i feel|seems like)\b',
        ],
        'complaint': [
            r'\b(not working|broken|error|issue|problem|bug|failed|wrong|bad|terrible)\b',
        ],
        'request_refund': [
            r'\b(refund|money back|cancel|cancellation|return|reimburse)\b',
        ],
    }

    for intent_type, patterns in intent_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                intent = intent_type
                break
        if intent:
            break

    # Detect categories based on keywords
    categories = []
    category_keywords = {
        'technical': ['code', 'error', 'bug', 'api', 'database', 'server', 'programming', 'function', 'debug'],
        'support': ['help', 'issue', 'problem', 'not working', 'broken', 'error', 'failed'],
        'billing': ['payment', 'invoice', 'charge', 'subscription', 'price', 'cost', 'billing', 'refund'],
        'account': ['account', 'password', 'login', 'register', 'profile', 'settings', 'user'],
        'feature': ['feature', 'add', 'new', 'request', 'suggestion', 'improve', 'enhancement'],
        'question': ['what', 'how', 'why', 'when', 'where', 'who', 'which', '?'],
        'casual': ['hi', 'hello', 'thanks', 'thank', 'bye', 'ok', 'okay', 'yes', 'no'],
    }

    for category, kws in category_keywords.items():
        for kw in kws:
            if kw in text_lower:
                if category not in categories:
                    categories.append(category)
                break

    return {
        'keywords': keywords,  # To match against enriched message topics
        'intent_hint': intent,  # To match against enriched message intent
        'category_hints': categories,  # To match against enriched message categories
    }
