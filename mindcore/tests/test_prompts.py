"""
Tests for prompts module.

Tests prompt templates, formatting, and custom prompt loading.
"""

import pytest
import tempfile
import os
from pathlib import Path
from mindcore.prompts import (
    ENRICHMENT_SYSTEM_PROMPT,
    ENRICHMENT_USER_PROMPT_TEMPLATE,
    CONTEXT_ASSEMBLY_SYSTEM_PROMPT,
    CONTEXT_ASSEMBLY_USER_PROMPT_TEMPLATE,
    IMPORTANCE_CRITERIA,
    get_enrichment_prompt,
    get_context_assembly_prompt,
    load_custom_prompts
)


class TestPromptConstants:
    """Test prompt constant definitions."""

    def test_enrichment_system_prompt_exists(self):
        """Test enrichment system prompt is defined."""
        assert isinstance(ENRICHMENT_SYSTEM_PROMPT, str)
        assert len(ENRICHMENT_SYSTEM_PROMPT) > 0
        assert "metadata enrichment" in ENRICHMENT_SYSTEM_PROMPT.lower()

    def test_enrichment_system_prompt_contains_requirements(self):
        """Test enrichment prompt contains required fields."""
        required_fields = [
            "topics",
            "categories",
            "importance",
            "sentiment",
            "intent",
            "tags",
            "entities",
            "key_phrases"
        ]

        for field in required_fields:
            assert field in ENRICHMENT_SYSTEM_PROMPT, (
                f"Enrichment prompt missing field: {field}"
            )

    def test_enrichment_user_prompt_template_exists(self):
        """Test enrichment user prompt template is defined."""
        assert isinstance(ENRICHMENT_USER_PROMPT_TEMPLATE, str)
        assert "{role}" in ENRICHMENT_USER_PROMPT_TEMPLATE
        assert "{text}" in ENRICHMENT_USER_PROMPT_TEMPLATE

    def test_context_assembly_system_prompt_exists(self):
        """Test context assembly system prompt is defined."""
        assert isinstance(CONTEXT_ASSEMBLY_SYSTEM_PROMPT, str)
        assert len(CONTEXT_ASSEMBLY_SYSTEM_PROMPT) > 0
        assert "context assembly" in CONTEXT_ASSEMBLY_SYSTEM_PROMPT.lower()

    def test_context_assembly_system_prompt_contains_requirements(self):
        """Test context assembly prompt contains required fields."""
        required_fields = [
            "assembled_context",
            "key_points",
            "relevant_message_ids",
            "metadata"
        ]

        for field in required_fields:
            assert field in CONTEXT_ASSEMBLY_SYSTEM_PROMPT, (
                f"Context assembly prompt missing field: {field}"
            )

    def test_context_assembly_user_prompt_template_exists(self):
        """Test context assembly user prompt template is defined."""
        assert isinstance(CONTEXT_ASSEMBLY_USER_PROMPT_TEMPLATE, str)
        assert "{formatted_messages}" in CONTEXT_ASSEMBLY_USER_PROMPT_TEMPLATE
        assert "{query}" in CONTEXT_ASSEMBLY_USER_PROMPT_TEMPLATE

    def test_importance_criteria_exists(self):
        """Test importance criteria is defined."""
        assert isinstance(IMPORTANCE_CRITERIA, str)
        assert len(IMPORTANCE_CRITERIA) > 0
        assert "0.0" in IMPORTANCE_CRITERIA
        assert "1.0" in IMPORTANCE_CRITERIA


class TestGetEnrichmentPrompt:
    """Test get_enrichment_prompt function."""

    def test_format_with_role_and_text(self):
        """Test formatting with role and text."""
        prompt = get_enrichment_prompt(role="user", text="Hello, how are you?")

        assert "user" in prompt
        assert "Hello, how are you?" in prompt
        assert isinstance(prompt, str)

    def test_different_roles(self):
        """Test formatting with different roles."""
        roles = ["user", "assistant", "system"]

        for role in roles:
            prompt = get_enrichment_prompt(role=role, text="Test message")
            assert role in prompt

    def test_special_characters_in_text(self):
        """Test handling of special characters."""
        special_text = "Test with 'quotes', \"double quotes\", and\nnewlines"
        prompt = get_enrichment_prompt(role="user", text=special_text)

        assert special_text in prompt

    def test_empty_text(self):
        """Test with empty text."""
        prompt = get_enrichment_prompt(role="user", text="")
        assert isinstance(prompt, str)

    def test_long_text(self):
        """Test with very long text."""
        long_text = "x" * 10000
        prompt = get_enrichment_prompt(role="user", text=long_text)

        assert long_text in prompt


class TestGetContextAssemblyPrompt:
    """Test get_context_assembly_prompt function."""

    def test_format_with_messages_and_query(self):
        """Test formatting with messages and query."""
        formatted_messages = """
        1. [user] Hello!
        2. [assistant] Hi there!
        3. [user] How are you?
        """
        query = "conversation greeting"

        prompt = get_context_assembly_prompt(
            formatted_messages=formatted_messages,
            query=query
        )

        assert formatted_messages in prompt
        assert query in prompt
        assert isinstance(prompt, str)

    def test_empty_messages(self):
        """Test with empty message history."""
        prompt = get_context_assembly_prompt(
            formatted_messages="",
            query="test query"
        )

        assert "test query" in prompt

    def test_empty_query(self):
        """Test with empty query."""
        prompt = get_context_assembly_prompt(
            formatted_messages="Message 1\nMessage 2",
            query=""
        )

        assert "Message 1" in prompt

    def test_special_characters(self):
        """Test with special characters in messages and query."""
        messages = "User: What's the difference between 'X' and \"Y\"?"
        query = "comparison & analysis"

        prompt = get_context_assembly_prompt(
            formatted_messages=messages,
            query=query
        )

        assert messages in prompt
        assert query in prompt


class TestLoadCustomPrompts:
    """Test load_custom_prompts function."""

    def test_load_from_nonexistent_file(self):
        """Test loading from nonexistent file returns empty dict."""
        result = load_custom_prompts("/nonexistent/path/prompts.yaml")
        assert result == {}

    def test_load_from_none_path(self):
        """Test loading with None path returns empty dict."""
        result = load_custom_prompts(None)
        assert result == {}

    def test_load_valid_yaml_file(self):
        """Test loading valid YAML file."""
        # Create temporary YAML file
        yaml_content = """
enrichment_system_prompt: |
  Custom enrichment prompt here.
  With multiple lines.

context_assembly_system_prompt: |
  Custom context assembly prompt.

custom_field: "Custom value"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            result = load_custom_prompts(temp_path)

            assert isinstance(result, dict)
            assert "enrichment_system_prompt" in result
            assert "context_assembly_system_prompt" in result
            assert "custom_field" in result

            assert "Custom enrichment prompt" in result["enrichment_system_prompt"]
            assert "Custom context assembly" in result["context_assembly_system_prompt"]
            assert result["custom_field"] == "Custom value"

        finally:
            os.unlink(temp_path)

    def test_load_empty_yaml_file(self):
        """Test loading empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            result = load_custom_prompts(temp_path)
            assert result == {}
        finally:
            os.unlink(temp_path)

    def test_load_yaml_with_only_prompts(self):
        """Test loading YAML with only prompt fields."""
        yaml_content = """
enrichment_system_prompt: "Custom enrichment"
context_assembly_system_prompt: "Custom context"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            result = load_custom_prompts(temp_path)

            assert len(result) == 2
            assert result["enrichment_system_prompt"] == "Custom enrichment"
            assert result["context_assembly_system_prompt"] == "Custom context"

        finally:
            os.unlink(temp_path)

    def test_load_yaml_with_nested_structure(self):
        """Test loading YAML with nested structure."""
        yaml_content = """
enrichment_system_prompt: "Custom prompt"

metadata:
  author: "Test"
  version: "1.0"

settings:
  temperature: 0.5
  max_tokens: 1000
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            result = load_custom_prompts(temp_path)

            assert "enrichment_system_prompt" in result
            assert "metadata" in result
            assert "settings" in result

            assert isinstance(result["metadata"], dict)
            assert result["metadata"]["author"] == "Test"

        finally:
            os.unlink(temp_path)

    def test_load_multiline_prompts(self):
        """Test loading multiline prompts with proper formatting."""
        yaml_content = """
enrichment_system_prompt: |
  You are a metadata enrichment AI agent.

  Your task is to analyze messages and extract:
  - Topics
  - Categories
  - Sentiment

  Return JSON format.

context_assembly_system_prompt: |
  You are a context assembly agent.

  Focus on:
  1. Relevant context
  2. Key points
  3. Message IDs
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            result = load_custom_prompts(temp_path)

            enrichment = result["enrichment_system_prompt"]
            context = result["context_assembly_system_prompt"]

            # Check multiline content preserved
            assert "metadata enrichment AI agent" in enrichment
            assert "Topics" in enrichment
            assert "Categories" in enrichment
            assert "\n" in enrichment

            assert "context assembly agent" in context
            assert "Relevant context" in context
            assert "\n" in context

        finally:
            os.unlink(temp_path)


class TestPromptsIntegration:
    """Integration tests for prompts module."""

    def test_all_prompt_constants_are_strings(self):
        """Test all prompt constants are strings."""
        constants = [
            ENRICHMENT_SYSTEM_PROMPT,
            ENRICHMENT_USER_PROMPT_TEMPLATE,
            CONTEXT_ASSEMBLY_SYSTEM_PROMPT,
            CONTEXT_ASSEMBLY_USER_PROMPT_TEMPLATE,
            IMPORTANCE_CRITERIA
        ]

        for constant in constants:
            assert isinstance(constant, str)
            assert len(constant) > 0

    def test_prompt_functions_return_strings(self):
        """Test prompt functions return strings."""
        enrichment = get_enrichment_prompt("user", "test")
        context = get_context_assembly_prompt("messages", "query")

        assert isinstance(enrichment, str)
        assert isinstance(context, str)

    def test_prompts_contain_json_instructions(self):
        """Test prompts mention JSON format."""
        assert "JSON" in ENRICHMENT_SYSTEM_PROMPT or "json" in ENRICHMENT_SYSTEM_PROMPT
        assert "JSON" in CONTEXT_ASSEMBLY_SYSTEM_PROMPT or "json" in CONTEXT_ASSEMBLY_SYSTEM_PROMPT

    def test_realistic_prompt_usage(self):
        """Test realistic usage scenario."""
        # Enrichment
        enrichment_prompt = get_enrichment_prompt(
            role="user",
            text="This is urgent! We need to fix the critical bug in production ASAP."
        )

        assert "user" in enrichment_prompt
        assert "urgent" in enrichment_prompt
        assert "critical" in enrichment_prompt

        # Context assembly
        formatted_msgs = """
        1. [user] What's the status of the deployment?
        2. [assistant] It's currently in progress.
        3. [user] Any issues so far?
        """

        context_prompt = get_context_assembly_prompt(
            formatted_messages=formatted_msgs,
            query="deployment status"
        )

        assert "deployment" in context_prompt
        assert "status" in context_prompt
        assert "[user]" in context_prompt

    def test_custom_prompts_override(self):
        """Test custom prompts can override defaults."""
        # Create custom prompts file
        custom_yaml = """
enrichment_system_prompt: "CUSTOM ENRICHMENT PROMPT"
context_assembly_system_prompt: "CUSTOM CONTEXT PROMPT"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(custom_yaml)
            temp_path = f.name

        try:
            custom = load_custom_prompts(temp_path)

            # Custom prompts should be different from defaults
            assert custom["enrichment_system_prompt"] != ENRICHMENT_SYSTEM_PROMPT
            assert custom["context_assembly_system_prompt"] != CONTEXT_ASSEMBLY_SYSTEM_PROMPT

            # Custom prompts should be usable
            assert custom["enrichment_system_prompt"] == "CUSTOM ENRICHMENT PROMPT"
            assert custom["context_assembly_system_prompt"] == "CUSTOM CONTEXT PROMPT"

        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
