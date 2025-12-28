"""Tests for SVL Gate - The semantic validation layer kernel.

Tests verify the three core guarantees:
1. Canonicalization - Transforms heterogeneous inputs to unified schema
2. Policy Enforcement - Validates all data against SVL vocabulary
3. Governance Choke Point - No bypass paths exist
"""

import pytest

from mindcore.svl import (
    DEFAULT_SVL,
    GateDecision,
    GatePolicy,
    GateResult,
    PolicyViolation,
    RetryConfig,
    SharedVocabularyLayer,
    SVLGate,
    SVLSchema,
)


@pytest.fixture
def svl():
    """Create test SVL instance."""
    return SharedVocabularyLayer(
        schema=SVLSchema(
            version="1.0.0",
            topics=["billing", "support", "orders", "shipping", "general"],
            categories=["support", "account", "product", "urgent"],
            memory_types=["episodic", "semantic", "preference", "procedural"],
            sentiments=["positive", "negative", "neutral", "mixed"],
            access_levels=["private", "team", "shared", "global"],
        )
    )


@pytest.fixture
def gate(svl):
    """Create test gate instance."""
    return SVLGate(svl=svl)


@pytest.fixture
def strict_gate(svl):
    """Create gate with strict policy - no auto-correction allowed."""
    policy = GatePolicy(
        strict_mode=True,
        enforce_vocabulary=True,
        allow_canonicalization=False,  # Disable auto-correction for strict mode
        allow_fallback=False,
    )
    return SVLGate(svl=svl, policy=policy)


class TestCanonicalization:
    """Test canonicalization - transforming heterogeneous inputs to unified schema."""

    def test_canonicalize_memory_type_variations(self, gate):
        """Test that memory type variations are canonicalized."""
        # Test various memory type inputs
        variations = [
            ("episode", "episodic"),
            ("event", "episodic"),
            ("fact", "semantic"),
            ("knowledge", "semantic"),
            ("procedure", "procedural"),
            ("howto", "procedural"),
        ]

        for input_type, expected in variations:
            result = gate.process_inbound(
                llm_output={
                    "content": "Test content",
                    "memory_type": input_type,
                },
                user_id="user123",
            )

            assert result.success, f"Failed for {input_type}: {result.errors}"
            assert result.memory["memory_type"] == expected

    def test_canonicalize_sentiment_variations(self, gate):
        """Test that sentiment variations are canonicalized."""
        variations = [
            ("happy", "positive"),
            ("satisfied", "positive"),
            ("sad", "negative"),
            ("frustrated", "negative"),
            ("angry", "negative"),
            ("good", "positive"),
            ("bad", "negative"),
        ]

        for input_sentiment, expected in variations:
            result = gate.process_inbound(
                llm_output={
                    "content": "Test content",
                    "memory_type": "episodic",
                    "sentiment": input_sentiment,
                },
                user_id="user123",
            )

            assert result.success
            assert result.memory["sentiment"] == expected

    def test_canonicalize_topic_normalization(self, gate):
        """Test that topics are normalized."""
        result = gate.process_inbound(
            llm_output={
                "content": "Test content",
                "memory_type": "episodic",
                "topics": ["Billing", "SUPPORT", "orders"],  # Mixed case
            },
            user_id="user123",
        )

        assert result.success
        # Topics should be normalized to lowercase canonical form
        assert set(result.memory["topics"]) == {"billing", "support", "orders"}

    def test_canonicalize_importance_clamping(self, gate):
        """Test that importance is clamped to 0-1 range."""
        # Test value > 1
        result = gate.process_inbound(
            llm_output={
                "content": "Test content",
                "memory_type": "episodic",
                "importance": 1.5,
            },
            user_id="user123",
        )

        assert result.success
        assert result.memory["importance"] == 1.0
        assert result.canonicalized

        # Test value < 0
        result = gate.process_inbound(
            llm_output={
                "content": "Test content",
                "memory_type": "episodic",
                "importance": -0.5,
            },
            user_id="user123",
        )

        assert result.success
        assert result.memory["importance"] == 0.0

    def test_canonicalize_adds_defaults(self, gate):
        """Test that missing optional fields get defaults."""
        result = gate.process_inbound(
            llm_output={
                "content": "Test content",
                "memory_type": "episodic",
            },
            user_id="user123",
        )

        assert result.success
        # Check defaults are added
        assert result.memory["sentiment"] == "neutral"
        assert result.memory["access_level"] == "private"
        assert result.memory["importance"] == 0.5
        assert result.memory["topics"] == []
        assert result.memory["categories"] == []
        assert result.memory["entities"] == []


class TestPolicyEnforcement:
    """Test policy enforcement - validation against SVL vocabulary."""

    def test_reject_missing_required_fields(self, strict_gate):
        """Test that missing required fields are rejected."""
        # Missing content
        result = strict_gate.process_inbound(
            llm_output={"memory_type": "episodic"},
            user_id="user123",
        )

        assert not result.success
        assert result.decision == GateDecision.REJECT
        assert any(e.violation == PolicyViolation.MISSING_REQUIRED_FIELD for e in result.errors)

    def test_reject_invalid_memory_type(self, strict_gate):
        """Test that invalid memory types are rejected in strict mode."""
        result = strict_gate.process_inbound(
            llm_output={
                "content": "Test content",
                "memory_type": "invalid_type",
            },
            user_id="user123",
        )

        assert not result.success
        assert any(e.violation == PolicyViolation.INVALID_MEMORY_TYPE for e in result.errors)

    def test_reject_invalid_topics(self, strict_gate):
        """Test that invalid topics are rejected in strict mode."""
        result = strict_gate.process_inbound(
            llm_output={
                "content": "Test content",
                "memory_type": "episodic",
                "topics": ["invalid_topic", "not_in_vocab"],
            },
            user_id="user123",
        )

        assert not result.success
        assert any(e.violation == PolicyViolation.INVALID_TOPIC for e in result.errors)

    def test_reject_content_too_short(self, gate):
        """Test that content below minimum length is rejected."""
        gate._policy.min_content_length = 5

        result = gate.process_inbound(
            llm_output={
                "content": "Hi",  # Too short
                "memory_type": "episodic",
            },
            user_id="user123",
        )

        assert not result.success
        assert any(e.violation == PolicyViolation.CONTENT_TOO_SHORT for e in result.errors)

    def test_accept_valid_memory(self, gate):
        """Test that valid memories are accepted."""
        result = gate.process_inbound(
            llm_output={
                "content": "This is a valid test memory",
                "memory_type": "episodic",
                "topics": ["billing"],
                "categories": ["support"],
                "importance": 0.7,
                "sentiment": "neutral",
            },
            user_id="user123",
        )

        assert result.success
        assert result.decision in (GateDecision.ACCEPT, GateDecision.CANONICALIZE)


class TestRetryStrategies:
    """Test retry strategies for handling validation failures."""

    def test_retry_with_error_feedback(self, gate):
        """Test retry with error feedback."""
        # Mock LLM call that fixes errors on retry
        call_count = 0

        def mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call might fail
                return (
                    '{"content": "Fixed content", "memory_type": "episodic", "topics": ["billing"]}'
                )
            return '{"content": "Fixed content", "memory_type": "episodic"}'

        result = gate.process_inbound(
            llm_output={
                "content": "Test",
                "memory_type": "invalid_type",
            },
            user_id="user123",
            llm_call=mock_llm,
        )

        # With fallback enabled, should eventually succeed
        assert result.success or result.decision == GateDecision.FALLBACK

    def test_fallback_fixes_invalid_topics(self, gate):
        """Test that fallback strategy removes invalid topics."""
        result = gate.process_inbound(
            llm_output={
                "content": "Test content with invalid topics",
                "memory_type": "episodic",
                "topics": ["invalid_topic", "billing"],  # One valid, one invalid
            },
            user_id="user123",
        )

        # Fallback should remove invalid and keep valid
        assert result.success
        if result.decision == GateDecision.FALLBACK:
            assert "billing" in result.memory["topics"]
            assert "invalid_topic" not in result.memory["topics"]


class TestOutboundValidation:
    """Test outbound validation - data going to LLM."""

    def test_outbound_redacts_sensitive_fields(self, gate):
        """Test that sensitive fields are redacted on outbound."""
        memory = {
            "memory_id": "mem_123",
            "content": "Test content",
            "memory_type": "episodic",
            "user_id": "user123",
            "embedding": [0.1, 0.2, 0.3],  # Should be redacted
            "robust_reinforcement": {"signals": []},  # Should be redacted
        }

        result = gate.process_outbound(memory)

        assert result.success
        assert "embedding" not in result.memory
        assert "robust_reinforcement" not in result.memory

    def test_outbound_keeps_core_fields(self, gate):
        """Test that core fields are preserved on outbound."""
        memory = {
            "memory_id": "mem_123",
            "content": "Test content",
            "memory_type": "episodic",
            "user_id": "user123",
            "topics": ["billing"],
            "importance": 0.8,
        }

        result = gate.process_outbound(memory)

        assert result.success
        assert result.memory["memory_id"] == "mem_123"
        assert result.memory["content"] == "Test content"
        assert result.memory["topics"] == ["billing"]


class TestGovernanceChokePoint:
    """Test that SVL Gate is a mandatory choke point with no bypass."""

    def test_context_injection(self, gate):
        """Test that user_id and other context is always injected."""
        result = gate.process_inbound(
            llm_output={
                "content": "Test content",
                "memory_type": "episodic",
            },
            user_id="user123",
            agent_id="agent456",
            session_id="session789",
        )

        assert result.success
        assert result.memory["user_id"] == "user123"
        assert result.memory["agent_id"] == "agent456"
        assert result.memory["session_id"] == "session789"

    def test_memory_id_generated(self, gate):
        """Test that memory_id is always generated if missing."""
        result = gate.process_inbound(
            llm_output={
                "content": "Test content",
                "memory_type": "episodic",
            },
            user_id="user123",
        )

        assert result.success
        assert result.memory["memory_id"]
        assert result.memory["memory_id"].startswith("mem_")

    def test_vocabulary_version_stamped(self, gate):
        """Test that vocabulary version is stamped on memory."""
        result = gate.process_inbound(
            llm_output={
                "content": "Test content",
                "memory_type": "episodic",
            },
            user_id="user123",
        )

        assert result.success
        assert result.memory["vocabulary_version"] == "1.0.0"


class TestJsonParsing:
    """Test JSON parsing from various LLM output formats."""

    def test_parse_json_string(self, gate):
        """Test parsing JSON from string input."""
        result = gate.process_inbound(
            llm_output='{"content": "Test content", "memory_type": "episodic"}',
            user_id="user123",
        )

        assert result.success

    def test_parse_json_from_markdown(self, gate):
        """Test parsing JSON from markdown code block."""
        result = gate.process_inbound(
            llm_output="""Here's the memory:
```json
{"content": "Test content", "memory_type": "episodic"}
```
""",
            user_id="user123",
        )

        assert result.success

    def test_reject_invalid_json(self, gate):
        """Test rejection of invalid JSON."""
        result = gate.process_inbound(
            llm_output="This is not valid JSON at all",
            user_id="user123",
        )

        assert not result.success
        assert any(e.violation == PolicyViolation.INVALID_JSON for e in result.errors)


class TestStatistics:
    """Test gate statistics tracking."""

    def test_stats_tracking(self, gate):
        """Test that statistics are tracked correctly."""
        # Process some valid and invalid requests
        gate.process_inbound(
            {"content": "Valid", "memory_type": "episodic"},
            user_id="u1",
        )
        gate.process_inbound(
            {"content": "Also valid", "memory_type": "semantic"},
            user_id="u1",
        )

        stats = gate.get_stats()

        assert stats["total_inbound"] == 2
        assert stats["accepted"] == 2
        assert stats["rejected"] == 0

    def test_violation_stats(self, strict_gate):
        """Test that violation types are tracked."""
        # Create some violations
        strict_gate.process_inbound(
            {"memory_type": "episodic"},  # Missing content
            user_id="u1",
        )
        strict_gate.process_inbound(
            {"content": "Test", "memory_type": "invalid"},  # Invalid type
            user_id="u1",
        )

        stats = strict_gate.get_stats()

        assert stats["rejected"] == 2
        assert "missing_required_field" in stats["violations_by_type"]
        assert "invalid_memory_type" in stats["violations_by_type"]


class TestGatePolicy:
    """Test different policy configurations."""

    def test_non_strict_mode_with_fallback(self, svl):
        """Test that non-strict mode uses fallback strategies when canonicalization disabled."""
        policy = GatePolicy(
            strict_mode=False,
            allow_fallback=True,
            enforce_vocabulary=True,
            allow_canonicalization=False,  # Disable to force fallback path
        )
        gate = SVLGate(svl=svl, policy=policy)

        result = gate.process_inbound(
            llm_output={
                "content": "Test content",
                "memory_type": "invalid_type",  # Invalid - will need fallback
            },
            user_id="user123",
        )

        # Should succeed with fallback since canonicalization is disabled
        assert result.success
        assert result.decision == GateDecision.FALLBACK

    def test_non_strict_mode_with_canonicalization(self, svl):
        """Test that non-strict mode uses canonicalization when enabled."""
        policy = GatePolicy(
            strict_mode=False,
            allow_fallback=True,
            enforce_vocabulary=True,
            allow_canonicalization=True,  # Enable canonicalization
        )
        gate = SVLGate(svl=svl, policy=policy)

        result = gate.process_inbound(
            llm_output={
                "content": "Test content",
                "memory_type": "invalid_type",  # Will be canonicalized to episodic
            },
            user_id="user123",
        )

        # Should succeed with canonicalization (preferred over fallback)
        assert result.success
        assert result.decision == GateDecision.CANONICALIZE
        assert result.memory["memory_type"] == "episodic"

    def test_vocabulary_enforcement_disabled(self, svl):
        """Test that vocabulary enforcement can be disabled."""
        policy = GatePolicy(
            enforce_vocabulary=False,
            strict_mode=True,
        )
        gate = SVLGate(svl=svl, policy=policy)

        result = gate.process_inbound(
            llm_output={
                "content": "Test content",
                "memory_type": "episodic",
                "topics": ["any_topic_allowed"],  # Not in vocab
            },
            user_id="user123",
        )

        # Should succeed because vocabulary enforcement is disabled
        assert result.success


class TestGatedStorage:
    """Test the gated storage wrappers."""

    def test_gated_mindcore_store(self, svl):
        """Test GatedMindcore store operation."""
        from mindcore.svl.gated_storage import GatedMindcore

        memory = GatedMindcore(
            storage="sqlite:///:memory:",
            vocabulary=svl,
        )

        result = memory.store(
            llm_output={
                "content": "Test memory content",
                "memory_type": "preference",
                "topics": ["billing"],
            },
            user_id="user123",
        )

        assert result.success
        assert result.memory_id is not None

        memory.close()

    def test_gated_mindcore_rejects_invalid(self, svl):
        """Test that GatedMindcore rejects invalid data when canonicalization is disabled."""
        from mindcore.svl.gated_storage import GatedMindcore

        policy = GatePolicy(
            strict_mode=True,
            allow_fallback=False,
            allow_canonicalization=False,  # Disable auto-correction to test rejection
        )
        memory = GatedMindcore(
            storage="sqlite:///:memory:",
            vocabulary=svl,
            gate_policy=policy,
        )

        result = memory.store(
            llm_output={
                "content": "Test",
                "memory_type": "invalid_type",
            },
            user_id="user123",
        )

        assert not result.success

        memory.close()
