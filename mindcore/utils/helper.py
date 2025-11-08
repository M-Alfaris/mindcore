"""
Helper utilities for Mindcore framework.
"""
import uuid
from datetime import datetime
from typing import Dict, Any, Optional


def generate_message_id() -> str:
    """
    Generate a unique message ID.

    Returns:
        UUID-based message ID.
    """
    return f"msg_{uuid.uuid4().hex}"


def generate_session_id() -> str:
    """
    Generate a unique session ID.

    Returns:
        UUID-based session ID.
    """
    return f"session_{uuid.uuid4().hex}"


def generate_thread_id() -> str:
    """
    Generate a unique thread ID.

    Returns:
        UUID-based thread ID.
    """
    return f"thread_{uuid.uuid4().hex}"


def current_timestamp() -> datetime:
    """
    Get current UTC timestamp.

    Returns:
        Current datetime in UTC.
    """
    return datetime.utcnow()


def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize text input.

    Args:
        text: Input text.
        max_length: Maximum length (optional).

    Returns:
        Sanitized text.
    """
    # Strip whitespace
    sanitized = text.strip()

    # Truncate if needed
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized


def merge_metadata(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge metadata dictionaries.

    Args:
        base: Base metadata dictionary.
        updates: Updates to merge.

    Returns:
        Merged metadata dictionary.
    """
    merged = base.copy()

    for key, value in updates.items():
        if key in merged and isinstance(merged[key], list) and isinstance(value, list):
            # Merge lists (unique values)
            merged[key] = list(set(merged[key] + value))
        elif key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Merge dicts recursively
            merged[key] = merge_metadata(merged[key], value)
        else:
            # Override
            merged[key] = value

    return merged


def validate_message_dict(message_dict: Dict[str, Any]) -> bool:
    """
    Validate message dictionary has required fields.

    Args:
        message_dict: Message dictionary to validate.

    Returns:
        True if valid, False otherwise.
    """
    required_fields = ['user_id', 'thread_id', 'session_id', 'role', 'text']

    for field in required_fields:
        if field not in message_dict:
            return False

    return True


def format_context_for_prompt(context: Dict[str, Any]) -> str:
    """
    Format assembled context for inclusion in LLM prompt.

    Args:
        context: Assembled context dictionary.

    Returns:
        Formatted context string.
    """
    parts = []

    if context.get('assembled_context'):
        parts.append("## Historical Context")
        parts.append(context['assembled_context'])
        parts.append("")

    if context.get('key_points'):
        parts.append("## Key Points")
        for point in context['key_points']:
            parts.append(f"- {point}")
        parts.append("")

    return "\n".join(parts)
