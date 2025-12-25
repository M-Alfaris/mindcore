"""LLM Provider Configurations for SVL Metadata Extraction.

This module provides provider-specific configurations for extracting
SVL-compliant metadata using the latest LLM API features:

- OpenAI GPT-5: Responses API with reasoning_effort, structured outputs
- Claude: Extended thinking with budget_tokens, structured outputs
- Gemini: Thinking mode with thinkingBudget, structured outputs

Key Features:
- Structured outputs with JSON Schema validation
- Reasoning/thinking modes for better metadata extraction
- Temperature control (0 for deterministic outputs)
- Provider-specific beta headers and parameters

Example:
    from mindcore.v2.svl.llm_providers import (
        LLMProviderConfig,
        OpenAIConfig,
        ClaudeConfig,
        GeminiConfig,
    )

    # Get OpenAI config with reasoning
    config = OpenAIConfig(
        reasoning_effort="high",
        use_responses_api=True,
    )
    request_params = config.get_request_params(schema)

    # Get Claude config with extended thinking
    config = ClaudeConfig(
        thinking_budget=16000,
        use_interleaved_thinking=True,
    )
    request_params = config.get_request_params(schema)

References:
- OpenAI Responses API: https://platform.openai.com/docs/guides/responses-vs-chat-completions
- Claude Extended Thinking: https://docs.claude.com/en/docs/build-with-claude/extended-thinking
- Gemini Thinking: https://ai.google.dev/gemini-api/docs/thinking
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReasoningEffort(str, Enum):
    """OpenAI GPT-5 reasoning effort levels."""

    LOW = "low"  # Faster, fewer tokens
    MEDIUM = "medium"  # Balanced
    HIGH = "high"  # More thorough (default for gpt-5-pro)
    XHIGH = "xhigh"  # Maximum reasoning (gpt-5.1+ only)


class ThinkingMode(str, Enum):
    """Gemini 2.5 thinking mode settings (uses thinkingBudget)."""

    DISABLED = "disabled"  # thinkingBudget: 0
    DYNAMIC = "dynamic"  # thinkingBudget: -1 (auto-adjust)
    FIXED = "fixed"  # Custom thinkingBudget value


class ThinkingLevel(str, Enum):
    """Gemini 3 thinking level settings (uses thinkingLevel).

    Reference: https://ai.google.dev/gemini-api/docs/thinking

    Note: Gemini 3 Pro supports LOW and HIGH only.
    Gemini 3 Flash supports MINIMAL, LOW, MEDIUM, and HIGH.
    """

    MINIMAL = "minimal"  # Flash only - minimal thinking (may still think)
    LOW = "low"  # Minimizes latency and cost
    MEDIUM = "medium"  # Flash only - balanced
    HIGH = "high"  # Default - maximizes reasoning depth


@dataclass
class LLMProviderConfig(ABC):
    """Base class for LLM provider configurations."""

    # Common settings
    temperature: float = 0.0  # 0 for deterministic metadata extraction
    max_tokens: int = 4096
    seed: int | None = 42  # For reproducibility (OpenAI, Gemini)

    @abstractmethod
    def get_request_params(
        self,
        json_schema: dict[str, Any],
        include_reasoning: bool = True,
    ) -> dict[str, Any]:
        """Get provider-specific request parameters.

        Args:
            json_schema: JSON Schema for structured output
            include_reasoning: Enable reasoning/thinking mode

        Returns:
            Dict of request parameters for the provider's API
        """
        pass

    @abstractmethod
    def get_headers(self) -> dict[str, str]:
        """Get provider-specific headers (beta features, etc.)."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name."""
        pass


@dataclass
class OpenAIConfig(LLMProviderConfig):
    """OpenAI GPT-5 configuration with Responses API.

    Features:
    - Responses API for preserved reasoning across turns
    - reasoning_effort parameter (low/medium/high/xhigh)
    - Structured outputs with JSON Schema validation
    - text.format for structured output (not response_format)
    - instructions parameter for high-priority guidance
    - developer role for higher authority than user messages

    Reference: https://platform.openai.com/docs/guides/responses-vs-chat-completions

    API Notes (2025):
    - instructions parameter takes PRIORITY over prompts in input
    - instructions only applies to current request (not preserved with
      previous_response_id for conversation state)
    - o1-preview and o1-mini do NOT support system or developer messages
    - New reusable prompts available via prompt parameter (prompt id + version)
    - Remote MCP servers supported as built-in tools
    """

    model: str = "gpt-5"  # gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-pro
    reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH
    use_responses_api: bool = True  # Recommended for reasoning models
    store: bool = True  # Store conversation for reasoning preservation

    def get_provider_name(self) -> str:
        return "openai"

    def get_headers(self) -> dict[str, str]:
        """OpenAI doesn't require special headers for current features."""
        return {}

    def get_request_params(
        self,
        json_schema: dict[str, Any],
        include_reasoning: bool = True,
    ) -> dict[str, Any]:
        """Get OpenAI request parameters.

        For Responses API:
        - Uses text.format instead of response_format
        - Includes reasoning.effort for reasoning models
        - Functions are strict by default
        """
        params: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }

        if self.seed is not None:
            params["seed"] = self.seed

        if self.store:
            params["store"] = True

        # Structured output format
        if self.use_responses_api:
            # Responses API uses text.format
            params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "svl_metadata",
                    "schema": json_schema,
                    "strict": True,
                }
            }
        else:
            # Legacy Chat Completions uses response_format
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "svl_metadata",
                    "schema": json_schema,
                    "strict": True,
                },
            }

        # Reasoning settings for reasoning models
        if include_reasoning and self.model.startswith("gpt-5"):
            params["reasoning"] = {
                "effort": self.reasoning_effort.value,
            }

        return params

    def get_chat_completions_params(
        self,
        json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Get legacy Chat Completions API parameters.

        Note: Responses API is recommended for GPT-5.
        Chat Completions drops reasoning between turns.
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "svl_metadata",
                    "schema": json_schema,
                    "strict": True,
                },
            },
        }


@dataclass
class ClaudeConfig(LLMProviderConfig):
    """Claude configuration with Extended Thinking.

    Features:
    - Extended thinking with budget_tokens (min 1,024 tokens)
    - Interleaved thinking for tool use (Claude 4+ only)
    - Structured outputs (strict: true)
    - Summarized thinking in Claude 4 (full thinking is encrypted)

    Reference: https://docs.claude.com/en/docs/build-with-claude/extended-thinking

    Model Availability:
    - Extended thinking: Sonnet 3.7+, Sonnet 4, Sonnet 4.5, Haiku 4.5,
                        Opus 4, Opus 4.1, Opus 4.5

    API Changes (2025):
    - Default top_p changed from 0.999 to 0.99 for all models
    - With extended thinking, top_p can be set to 0.95-1.0
    - Claude 4 returns summarized thinking by default (full thinking encrypted
      in signature field for security)
    - Interleaved thinking requires beta header: interleaved-thinking-2025-05-14

    Note: When extended thinking is enabled:
    - Temperature, top_p, top_k are NOT supported in same request
    - Forced tool use is NOT supported in Claude 3.7
    - Use Claude 4+ for best structured output + thinking
    """

    model: str = "claude-sonnet-4-5-20250514"  # Claude 4.5 Sonnet
    thinking_budget: int = 16000  # Max tokens for internal reasoning (min 1,024)
    use_interleaved_thinking: bool = True  # Think between tool calls (Claude 4+)
    use_extended_thinking: bool = True

    # Note: Temperature not supported with extended thinking
    temperature: float = 0.0  # Ignored when thinking enabled
    top_p: float = 0.99  # New default as of 2025 (was 0.999)

    def get_provider_name(self) -> str:
        return "anthropic"

    def get_headers(self) -> dict[str, str]:
        """Get Claude beta headers."""
        headers = {}

        # Structured outputs beta
        headers["anthropic-beta"] = "structured-outputs-2025-11-13"

        # Interleaved thinking beta (for tool use)
        if self.use_interleaved_thinking:
            # Combine beta headers
            headers["anthropic-beta"] = (
                "structured-outputs-2025-11-13,interleaved-thinking-2025-05-14"
            )

        return headers

    def get_request_params(
        self,
        json_schema: dict[str, Any],
        include_reasoning: bool = True,
    ) -> dict[str, Any]:
        """Get Claude request parameters.

        Note: With extended thinking enabled, structured output
        applies only to the final response, not thinking blocks.
        """
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
        }

        # Extended thinking configuration
        if include_reasoning and self.use_extended_thinking:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
            # Note: temperature not supported with thinking
        else:
            params["temperature"] = self.temperature

        # Structured output format
        params["output_format"] = {
            "type": "json_schema",
            "json_schema": json_schema,
        }

        return params

    def get_tool_use_params(
        self,
        json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Get params for strict tool use (alternative to output_format).

        Use this when you need forced tool calling behavior.
        """
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "tools": [
                {
                    "name": "extract_svl_metadata",
                    "description": "Extract SVL-compliant metadata from the message",
                    "input_schema": json_schema,
                    "strict": True,  # Guarantee schema validation
                }
            ],
            "tool_choice": {"type": "tool", "name": "extract_svl_metadata"},
        }


@dataclass
class GeminiConfig(LLMProviderConfig):
    """Gemini configuration with Thinking Mode.

    Features:
    - Gemini 2.5: thinkingBudget for token-based thinking control
    - Gemini 3: thinkingLevel for level-based thinking control
    - Structured outputs with JSON Schema
    - propertyOrdering for Gemini 2.0+
    - Thought signatures for Gemini 3 multi-turn conversations

    Reference: https://ai.google.dev/gemini-api/docs/thinking

    IMPORTANT API Version Differences:
    - Gemini 2.5 models use `thinkingBudget` (int: 0=off, -1=dynamic, N=fixed)
    - Gemini 3 models use `thinkingLevel` (str: minimal/low/medium/high)
    - You CANNOT mix these parameters - API will return an error
    - Gemini 3 requires thought_signatures to maintain reasoning across turns
    """

    model: str = "gemini-2.5-flash"  # gemini-2.5-pro, gemini-2.5-flash, gemini-3-pro, gemini-3-flash

    # Gemini 2.5 settings (thinkingBudget-based)
    thinking_mode: ThinkingMode = ThinkingMode.DYNAMIC
    thinking_budget: int | None = None  # Only for FIXED mode

    # Gemini 3 settings (thinkingLevel-based)
    thinking_level: ThinkingLevel | None = None  # Set this for Gemini 3 models

    property_ordering: list[str] | None = None  # Required for Gemini 2.0+

    def get_provider_name(self) -> str:
        return "google"

    def get_headers(self) -> dict[str, str]:
        """Gemini doesn't require special headers for current features."""
        return {}

    def is_gemini_3(self) -> bool:
        """Check if model is Gemini 3 (uses thinkingLevel instead of thinkingBudget)."""
        return "gemini-3" in self.model.lower()

    def get_request_params(
        self,
        json_schema: dict[str, Any],
        include_reasoning: bool = True,
    ) -> dict[str, Any]:
        """Get Gemini request parameters.

        Note:
        - Gemini 2.0+ requires propertyOrdering in the schema.
        - Gemini 3 uses thinkingLevel, Gemini 2.5 uses thinkingBudget.
        """
        params: dict[str, Any] = {
            "model": self.model,
            "generation_config": {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "application/json",
                "response_schema": self._prepare_schema(json_schema),
            },
        }

        # Thinking configuration - different for Gemini 3 vs 2.5
        if include_reasoning:
            if self.is_gemini_3():
                # Gemini 3 uses thinkingLevel (string-based levels)
                level = self.thinking_level or ThinkingLevel.HIGH
                params["generation_config"]["thinking_config"] = {
                    "thinking_level": level.value.upper()  # API expects uppercase
                }
            else:
                # Gemini 2.5 uses thinkingBudget (token-based)
                if self.thinking_mode == ThinkingMode.DISABLED:
                    params["generation_config"]["thinking_config"] = {
                        "thinking_budget": 0
                    }
                elif self.thinking_mode == ThinkingMode.DYNAMIC:
                    params["generation_config"]["thinking_config"] = {
                        "thinking_budget": -1  # Auto-adjust based on complexity
                    }
                elif self.thinking_mode == ThinkingMode.FIXED and self.thinking_budget:
                    params["generation_config"]["thinking_config"] = {
                        "thinking_budget": self.thinking_budget
                    }

        return params

    def get_gemini3_params(
        self,
        json_schema: dict[str, Any],
        thinking_level: ThinkingLevel = ThinkingLevel.HIGH,
        thought_signatures: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get Gemini 3 specific parameters with thought signature support.

        Gemini 3 requires thought signatures to maintain reasoning context
        across API calls. Missing signatures will result in a 400 error.

        Args:
            json_schema: JSON Schema for structured output
            thinking_level: Thinking level (LOW/HIGH for Pro, all for Flash)
            thought_signatures: Previous thought signatures to include

        Returns:
            Request parameters with thinking_level and signature handling

        Example:
            # First call
            params = config.get_gemini3_params(schema, ThinkingLevel.HIGH)
            response = model.generate_content(**params)
            signatures = extract_thought_signatures(response)  # Save these

            # Subsequent call - must include signatures
            params = config.get_gemini3_params(
                schema, ThinkingLevel.HIGH,
                thought_signatures=signatures
            )
        """
        params: dict[str, Any] = {
            "model": self.model,
            "generation_config": {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "application/json",
                "response_schema": self._prepare_schema(json_schema),
                "thinking_config": {
                    "thinking_level": thinking_level.value.upper()
                }
            },
        }

        # Include thought signatures if provided (required for multi-turn)
        if thought_signatures:
            params["thought_signatures"] = thought_signatures

        return params

    def _prepare_schema(self, json_schema: dict[str, Any]) -> dict[str, Any]:
        """Prepare schema for Gemini, adding propertyOrdering if needed."""
        schema = dict(json_schema)

        # Gemini 2.0+ requires propertyOrdering
        if self.property_ordering:
            schema["propertyOrdering"] = self.property_ordering
        elif "properties" in schema and "propertyOrdering" not in schema:
            # Auto-generate ordering from properties
            schema["propertyOrdering"] = list(schema["properties"].keys())

        return schema


@dataclass
class GenericConfig(LLMProviderConfig):
    """Generic configuration for other LLM providers.

    Use this for providers that support:
    - response_format with JSON Schema
    - Standard temperature/max_tokens parameters
    """

    model: str = ""
    provider: str = "generic"

    def get_provider_name(self) -> str:
        return self.provider

    def get_headers(self) -> dict[str, str]:
        return {}

    def get_request_params(
        self,
        json_schema: dict[str, Any],
        include_reasoning: bool = True,
    ) -> dict[str, Any]:
        """Get generic request parameters."""
        params: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self.seed is not None:
            params["seed"] = self.seed

        # Standard response_format
        params["response_format"] = {
            "type": "json_object",
        }

        return params


# Provider factory
_PROVIDER_CONFIGS: dict[str, type[LLMProviderConfig]] = {
    "openai": OpenAIConfig,
    "anthropic": ClaudeConfig,
    "claude": ClaudeConfig,
    "google": GeminiConfig,
    "gemini": GeminiConfig,
    "generic": GenericConfig,
}


def get_provider_config(
    provider: str,
    **kwargs: Any,
) -> LLMProviderConfig:
    """Get a provider configuration by name.

    Args:
        provider: Provider name (openai, anthropic, claude, google, gemini)
        **kwargs: Provider-specific configuration options

    Returns:
        Configured LLMProviderConfig instance

    Example:
        config = get_provider_config(
            "openai",
            model="gpt-5-pro",
            reasoning_effort=ReasoningEffort.XHIGH,
        )
    """
    provider_lower = provider.lower()

    if provider_lower not in _PROVIDER_CONFIGS:
        # Fallback to generic
        return GenericConfig(model=kwargs.get("model", ""), provider=provider)

    config_class = _PROVIDER_CONFIGS[provider_lower]
    return config_class(**kwargs)


# Recommended configurations for metadata extraction
RECOMMENDED_CONFIGS = {
    "openai": OpenAIConfig(
        model="gpt-5",
        reasoning_effort=ReasoningEffort.HIGH,
        use_responses_api=True,
        temperature=0.0,
    ),
    "anthropic": ClaudeConfig(
        model="claude-sonnet-4-5-20250514",
        thinking_budget=16000,
        use_extended_thinking=True,
        use_interleaved_thinking=True,
    ),
    # Gemini 2.5 (thinkingBudget-based)
    "google": GeminiConfig(
        model="gemini-2.5-flash",
        thinking_mode=ThinkingMode.DYNAMIC,
        temperature=0.0,
    ),
    # Gemini 3 (thinkingLevel-based)
    "gemini3": GeminiConfig(
        model="gemini-3-flash",
        thinking_level=ThinkingLevel.HIGH,
        temperature=0.0,
    ),
}


def get_recommended_config(provider: str) -> LLMProviderConfig:
    """Get recommended configuration for a provider.

    These configurations are optimized for metadata extraction:
    - Temperature 0 for deterministic output
    - Reasoning/thinking enabled for accurate classification
    - Structured outputs for schema validation

    Args:
        provider: Provider name. Special values:
            - "gemini3" or "gemini-3": Gemini 3 with thinkingLevel
            - "google" or "gemini": Gemini 2.5 with thinkingBudget

    Returns:
        Recommended configuration
    """
    provider_lower = provider.lower()
    if provider_lower in RECOMMENDED_CONFIGS:
        return RECOMMENDED_CONFIGS[provider_lower]
    if provider_lower == "claude":
        return RECOMMENDED_CONFIGS["anthropic"]
    if provider_lower in ("gemini-3", "gemini3-flash", "gemini3-pro"):
        return RECOMMENDED_CONFIGS["gemini3"]
    if provider_lower == "gemini":
        return RECOMMENDED_CONFIGS["google"]
    return GenericConfig(provider=provider)


# =============================================================================
# API-Level Context Injection (No Prompt Modification)
# =============================================================================


@dataclass
class FeedbackInjection:
    """Configuration for injecting feedback via API without modifying user prompt.

    Supports multiple injection methods:
    1. System/Instructions: High-authority guidance separate from user input
    2. Schema Descriptions: Embed hints in structured output schema
    3. Developer Role: Higher priority than user messages (OpenAI)
    4. Meta Messages: Hidden context in Claude (isMeta: true)
    """

    # What to inject
    effective_topics: list[tuple[str, float]] = field(default_factory=list)
    ineffective_topics: list[tuple[str, float]] = field(default_factory=list)
    effective_categories: list[tuple[str, float]] = field(default_factory=list)
    ineffective_categories: list[tuple[str, float]] = field(default_factory=list)

    # Overall guidance
    overall_usage_rate: float = 0.5
    recommendations: list[str] = field(default_factory=list)

    @classmethod
    def from_feedback(cls, feedback: dict[str, Any]) -> FeedbackInjection:
        """Create from FLR.get_metadata_feedback_for_extractor() output."""
        return cls(
            effective_topics=feedback.get("high_quality_topics", []),
            ineffective_topics=feedback.get("low_quality_topics", []),
            effective_categories=feedback.get("high_quality_categories", []),
            ineffective_categories=feedback.get("low_quality_categories", []),
        )

    @classmethod
    def from_optimizer(cls, optimization: dict[str, Any]) -> FeedbackInjection:
        """Create from QueryOptimizer.get_recommendations() output."""
        return cls(
            effective_topics=[
                (t["topic"], t.get("usage_rate", 0.5))
                for t in optimization.get("top_performing_topics", [])
            ],
            ineffective_topics=[
                (t["topic"], t.get("usage_rate", 0.5))
                for t in optimization.get("underperforming_topics", [])
            ],
            overall_usage_rate=optimization.get("overall_usage_rate", 0.5),
            recommendations=optimization.get("recommendations", []),
        )

    def to_system_instruction(self) -> str:
        """Generate system-level instruction text.

        Use this in:
        - OpenAI: instructions parameter or developer role message
        - Claude: system prompt
        - Gemini: systemInstruction
        """
        lines = []

        if self.effective_topics:
            topics = ", ".join([f"'{t[0]}' ({t[1]:.0%})" for t in self.effective_topics[:5]])
            lines.append(f"Prioritize these topics when assigning metadata: {topics}")

        if self.ineffective_topics:
            topics = ", ".join([f"'{t[0]}'" for t in self.ineffective_topics[:3]])
            lines.append(f"Avoid these topics unless strongly relevant: {topics}")

        if self.effective_categories:
            cats = ", ".join([f"'{c[0]}'" for c in self.effective_categories[:5]])
            lines.append(f"Prefer these categories: {cats}")

        if self.recommendations:
            for rec in self.recommendations[:2]:
                lines.append(rec)

        return "\n".join(lines) if lines else ""


class ContextInjector:
    """Injects feedback context via API-level mechanisms.

    This class handles provider-specific context injection WITHOUT
    modifying the user's prompt text.

    Methods:
    - get_openai_injection(): Returns instructions param and developer messages
    - get_claude_injection(): Returns system prompt and meta messages
    - get_gemini_injection(): Returns systemInstruction
    - annotate_schema(): Adds hints to JSON Schema descriptions
    """

    def __init__(self, feedback: FeedbackInjection):
        """Initialize with feedback to inject.

        Args:
            feedback: FeedbackInjection with topics/categories to boost/avoid
        """
        self.feedback = feedback

    def get_openai_injection(self) -> dict[str, Any]:
        """Get OpenAI-specific injection parameters.

        Returns params for:
        - instructions: High-priority guidance (Responses API)
        - developer message: Higher authority than user (Chat API)

        Example:
            injector = ContextInjector(feedback)
            injection = injector.get_openai_injection()

            # For Responses API
            response = client.responses.create(
                model="gpt-5",
                input=user_prompt,
                instructions=injection["instructions"],
            )

            # For Chat Completions API
            response = client.chat.completions.create(
                model="gpt-5",
                messages=injection["messages"] + [{"role": "user", "content": user_prompt}],
            )
        """
        instruction_text = self.feedback.to_system_instruction()

        return {
            # For Responses API (high priority)
            "instructions": instruction_text if instruction_text else None,

            # For Chat Completions API (developer role = higher than user)
            "messages": [
                {
                    "role": "developer",
                    "content": f"[Metadata Quality Guidance]\n{instruction_text}",
                }
            ] if instruction_text else [],
        }

    def get_claude_injection(self) -> dict[str, Any]:
        """Get Claude/Anthropic-specific injection parameters.

        Returns:
        - system: System prompt (separate from user turn)
        - meta_messages: Hidden context messages (isMeta: true)

        Example:
            injector = ContextInjector(feedback)
            injection = injector.get_claude_injection()

            response = client.messages.create(
                model="claude-sonnet-4-5",
                system=base_system + injection["system_suffix"],
                messages=injection["meta_messages"] + [{"role": "user", "content": user_prompt}],
            )
        """
        instruction_text = self.feedback.to_system_instruction()

        return {
            # Append to system prompt
            "system_suffix": f"\n\n[Quality Guidance]\n{instruction_text}" if instruction_text else "",

            # Hidden meta messages (sent to API, not shown in UI)
            "meta_messages": [
                {
                    "role": "user",
                    "content": f"[INTERNAL GUIDANCE - NOT USER INPUT]\n{instruction_text}",
                    # Note: isMeta handling is application-level, not API-level
                    # Your wrapper should filter these from UI display
                },
            ] if instruction_text else [],
        }

    def get_gemini_injection(self) -> dict[str, Any]:
        """Get Gemini-specific injection parameters.

        Returns:
        - systemInstruction: Separate system context

        Example:
            injector = ContextInjector(feedback)
            injection = injector.get_gemini_injection()

            response = model.generate_content(
                contents=user_prompt,
                generation_config=config,
                system_instruction=base_system + injection["system_suffix"],
            )
        """
        instruction_text = self.feedback.to_system_instruction()

        return {
            "system_suffix": f"\n\n[Quality Guidance]\n{instruction_text}" if instruction_text else "",
        }

    def annotate_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Annotate JSON Schema with effectiveness hints.

        This embeds feedback directly in schema descriptions,
        which LLMs use for structured output generation.

        Args:
            schema: Original JSON Schema

        Returns:
            Schema with annotated descriptions

        Example:
            injector = ContextInjector(feedback)
            annotated_schema = injector.annotate_schema(original_schema)

            # The schema now has descriptions like:
            # "topics": {
            #     "description": "Topics from SVL. Prefer: 'refund' (85%), 'billing' (72%). Avoid: 'general'."
            # }
        """
        schema = dict(schema)  # Copy

        if "properties" not in schema:
            return schema

        # Annotate topics
        if "topics" in schema["properties"]:
            desc = schema["properties"]["topics"].get("description", "Topics from SVL vocabulary")
            if self.feedback.effective_topics:
                preferred = ", ".join([f"'{t[0]}' ({t[1]:.0%})" for t in self.feedback.effective_topics[:5]])
                desc += f" Prefer: {preferred}."
            if self.feedback.ineffective_topics:
                avoid = ", ".join([f"'{t[0]}'" for t in self.feedback.ineffective_topics[:3]])
                desc += f" Avoid: {avoid}."
            schema["properties"]["topics"]["description"] = desc

        # Annotate categories
        if "categories" in schema["properties"]:
            desc = schema["properties"]["categories"].get("description", "Categories from SVL vocabulary")
            if self.feedback.effective_categories:
                preferred = ", ".join([f"'{c[0]}'" for c in self.feedback.effective_categories[:5]])
                desc += f" Prefer: {preferred}."
            if self.feedback.ineffective_categories:
                avoid = ", ".join([f"'{c[0]}'" for c in self.feedback.ineffective_categories[:3]])
                desc += f" Avoid: {avoid}."
            schema["properties"]["categories"]["description"] = desc

        return schema

    def get_injection_for_provider(self, provider: str) -> dict[str, Any]:
        """Get injection for any supported provider.

        Args:
            provider: Provider name (openai, anthropic, google, etc.)

        Returns:
            Provider-specific injection parameters
        """
        provider = provider.lower()

        if provider in ("openai", "gpt"):
            return self.get_openai_injection()
        elif provider in ("anthropic", "claude"):
            return self.get_claude_injection()
        elif provider in ("google", "gemini"):
            return self.get_gemini_injection()
        else:
            # Fallback: return system text that can be prepended
            return {
                "system_suffix": self.feedback.to_system_instruction(),
            }


def create_injector_from_flr(flr_feedback: dict[str, Any]) -> ContextInjector:
    """Create a ContextInjector from FLR feedback.

    Args:
        flr_feedback: Output from FLR.get_metadata_feedback_for_extractor()

    Returns:
        ContextInjector ready to use

    Example:
        feedback = flr.get_metadata_feedback_for_extractor()
        injector = create_injector_from_flr(feedback)

        # For OpenAI
        injection = injector.get_openai_injection()
        response = client.responses.create(
            model="gpt-5",
            input=user_prompt,
            instructions=base_instructions + injection["instructions"],
        )
    """
    return ContextInjector(FeedbackInjection.from_feedback(flr_feedback))


def create_injector_from_optimizer(optimizer_recommendations: dict[str, Any]) -> ContextInjector:
    """Create a ContextInjector from QueryOptimizer recommendations.

    Args:
        optimizer_recommendations: Output from QueryOptimizer.get_recommendations()

    Returns:
        ContextInjector ready to use
    """
    return ContextInjector(FeedbackInjection.from_optimizer(optimizer_recommendations))
