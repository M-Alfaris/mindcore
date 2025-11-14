"""
Centralized prompt templates for Mindcore AI agents.

This module contains all system prompts and templates used by the AI agents.
Users can customize these prompts by:
1. Editing this file directly
2. Providing custom prompts via configuration
3. Passing prompts to agent constructors

Usage:
    from mindcore.prompts import ENRICHMENT_SYSTEM_PROMPT, CONTEXT_ASSEMBLY_PROMPT

    # Use custom prompt
    agent = MetadataAgent(api_key="key", system_prompt=ENRICHMENT_SYSTEM_PROMPT)
"""

# =============================================================================
# METADATA ENRICHMENT PROMPTS
# =============================================================================

ENRICHMENT_SYSTEM_PROMPT = """You are a metadata enrichment AI agent. Your task is to analyze messages and extract structured metadata.

For each message, you must return a JSON object with the following fields:

{
  "topics": ["list of main topics discussed"],
  "categories": ["list of categories like 'question', 'statement', 'command', 'code', 'technical', 'casual', etc."],
  "importance": 0.0-1.0 (float, where 1.0 is most important),
  "sentiment": {
    "overall": "positive/negative/neutral",
    "score": 0.0-1.0 (float)
  },
  "intent": "primary intent of the message (e.g., 'ask_question', 'provide_info', 'request_action', 'express_opinion', 'greeting', etc.)",
  "tags": ["relevant tags or keywords"],
  "entities": ["named entities like people, places, technologies, products"],
  "key_phrases": ["important phrases from the message"]
}

Be concise and accurate. Focus on extracting the most relevant information."""


ENRICHMENT_USER_PROMPT_TEMPLATE = """Message to analyze:
Role: {role}
Text: {text}"""


# =============================================================================
# CONTEXT ASSEMBLY PROMPTS
# =============================================================================

CONTEXT_ASSEMBLY_SYSTEM_PROMPT = """You are a context assembly AI agent. Your task is to analyze a conversation history and current query to extract and summarize the most relevant context.

You will receive:
1. A list of historical messages with metadata
2. A current query or topic

Your task is to return a JSON object with:

{
  "assembled_context": "A clear, concise summary of relevant historical context that would help understand the current query. Focus on key facts, decisions, and relevant background.",
  "key_points": ["List of 3-5 most important points from the history"],
  "relevant_message_ids": ["List of message IDs that were most relevant"],
  "metadata": {
    "topics": ["main topics covered"],
    "sentiment": {
      "overall": "positive/negative/neutral",
      "trend": "improving/declining/stable"
    },
    "importance": 0.0-1.0 (overall importance of this context)
  }
}

Be selective and concise. Only include truly relevant information."""


CONTEXT_ASSEMBLY_USER_PROMPT_TEMPLATE = """Historical Messages:
{formatted_messages}

Current Query/Topic:
{query}

Analyze the messages and provide relevant context for the current query."""


# =============================================================================
# IMPORTANCE SCORING PROMPTS
# =============================================================================

IMPORTANCE_CRITERIA = """
Importance scoring criteria (0.0 to 1.0):
- 0.0-0.2: Greetings, small talk, casual conversation
- 0.3-0.4: General discussion, opinions, non-critical information
- 0.5-0.6: Relevant information, questions, normal conversation flow
- 0.7-0.8: Important decisions, key information, actionable items
- 0.9-1.0: Critical decisions, urgent matters, core business logic
"""


# =============================================================================
# CUSTOM PROMPT FUNCTIONS
# =============================================================================

def get_enrichment_prompt(role: str, text: str) -> str:
    """
    Get formatted enrichment prompt.

    Args:
        role: Message role
        text: Message text

    Returns:
        Formatted user prompt
    """
    return ENRICHMENT_USER_PROMPT_TEMPLATE.format(role=role, text=text)


def get_context_assembly_prompt(formatted_messages: str, query: str) -> str:
    """
    Get formatted context assembly prompt.

    Args:
        formatted_messages: Formatted message history
        query: User query

    Returns:
        Formatted user prompt
    """
    return CONTEXT_ASSEMBLY_USER_PROMPT_TEMPLATE.format(
        formatted_messages=formatted_messages,
        query=query
    )


# =============================================================================
# PROMPT CUSTOMIZATION
# =============================================================================

def load_custom_prompts(config_path: str = None) -> dict:
    """
    Load custom prompts from configuration file.

    Args:
        config_path: Path to custom prompts YAML/JSON file

    Returns:
        Dictionary of custom prompts

    Example:
        # custom_prompts.yaml
        enrichment_system_prompt: |
            Your custom enrichment prompt here...
        context_assembly_system_prompt: |
            Your custom context assembly prompt here...
    """
    if not config_path:
        return {}

    import yaml
    from pathlib import Path

    path = Path(config_path)
    if not path.exists():
        return {}

    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


__all__ = [
    "ENRICHMENT_SYSTEM_PROMPT",
    "ENRICHMENT_USER_PROMPT_TEMPLATE",
    "CONTEXT_ASSEMBLY_SYSTEM_PROMPT",
    "CONTEXT_ASSEMBLY_USER_PROMPT_TEMPLATE",
    "IMPORTANCE_CRITERIA",
    "get_enrichment_prompt",
    "get_context_assembly_prompt",
    "load_custom_prompts",
]
