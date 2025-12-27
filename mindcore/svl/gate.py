"""SVL Gate - The Semantic Validation Layer Kernel.

SVL Gate is the mandatory choke point for ALL data flows in MindCore.
Nothing enters or exits the memory system without passing through this gate.

This implements the three core guarantees:
1. CANONICALIZATION - Transforms heterogeneous inputs into unified representation
2. POLICY ENFORCEMENT - Guarantees alignment with system rules
3. GOVERNANCE CHOKE POINT - No bypass paths exist

Architecture:
    LLM Output -> SVL Gate (canonicalize + validate) -> FLR -> CLST
    CLST -> FLR -> SVL Gate (validate outbound) -> LLM Input

The SVL Gate is analogous to:
- OS kernels (system call interface)
- Database query planners (query validation/optimization)
- Language compilers (type checking, semantic analysis)

Example:
    from mindcore.svl import SVLGate, SharedVocabularyLayer

    svl = SharedVocabularyLayer()
    gate = SVLGate(svl=svl)

    # All LLM outputs must pass through the gate
    result = gate.process_inbound(
        llm_output={"content": "...", "memory_type": "preference"},
        llm_call=my_llm_function,  # For retries
    )

    if result.success:
        # Now safe to store in FLR/CLST
        flr.store(result.memory)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable


logger = logging.getLogger(__name__)


# =============================================================================
# Gate Result Types
# =============================================================================


class GateDecision(str, Enum):
    """Decision made by the SVL Gate."""

    ACCEPT = "accept"  # Data passed all checks
    REJECT = "reject"  # Data failed validation, cannot proceed
    RETRY = "retry"  # Data needs retry with LLM
    CANONICALIZE = "canonicalize"  # Data was auto-corrected
    FALLBACK = "fallback"  # Used fallback strategy


class PolicyViolation(str, Enum):
    """Types of policy violations."""

    INVALID_MEMORY_TYPE = "invalid_memory_type"
    INVALID_TOPIC = "invalid_topic"
    INVALID_CATEGORY = "invalid_category"
    INVALID_SENTIMENT = "invalid_sentiment"
    INVALID_ACCESS_LEVEL = "invalid_access_level"
    INVALID_IMPORTANCE = "invalid_importance"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    INVALID_JSON = "invalid_json"
    SEMANTIC_METADATA_ERROR = "semantic_metadata_error"
    CONTENT_TOO_SHORT = "content_too_short"
    CONTENT_TOO_LONG = "content_too_long"


@dataclass
class ValidationError:
    """A single validation error."""

    violation: PolicyViolation
    field: str
    message: str
    value: Any = None
    allowed_values: list[str] | None = None


@dataclass
class GateResult:
    """Result of processing data through the SVL Gate.

    This is the canonical output format for all gate operations.
    """

    success: bool
    decision: GateDecision
    memory: dict[str, Any] | None = None

    # Validation info
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Processing info
    canonicalized: bool = False  # True if data was auto-corrected
    canonicalization_changes: list[str] = field(default_factory=list)
    retry_count: int = 0
    strategy_used: str = "primary"

    # Quality metrics
    quality_score: float = 1.0  # 1.0 = perfect, degrades with fallbacks
    needs_review: bool = False  # True if human review recommended

    # Timing
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "decision": self.decision.value,
            "memory": self.memory,
            "errors": [
                {
                    "violation": e.violation.value,
                    "field": e.field,
                    "message": e.message,
                    "value": e.value,
                    "allowed_values": e.allowed_values,
                }
                for e in self.errors
            ],
            "warnings": self.warnings,
            "canonicalized": self.canonicalized,
            "canonicalization_changes": self.canonicalization_changes,
            "retry_count": self.retry_count,
            "strategy_used": self.strategy_used,
            "quality_score": self.quality_score,
            "needs_review": self.needs_review,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class RetryConfig:
    """Configuration for retry strategies."""

    max_retries: int = 3
    retry_with_error_feedback: bool = True
    retry_with_simplified_schema: bool = True
    use_rule_based_fallback: bool = True
    use_default_fallback: bool = True

    # Timeouts
    llm_timeout_seconds: float = 30.0

    # Quality thresholds
    min_quality_score: float = 0.3  # Below this, reject even with fallback


@dataclass
class GatePolicy:
    """Policy configuration for the SVL Gate.

    Defines strict rules that CANNOT be bypassed.
    """

    # Content policies
    min_content_length: int = 1
    max_content_length: int = 100000

    # Required fields
    required_fields: list[str] = field(
        default_factory=lambda: ["content", "memory_type", "user_id"]
    )

    # Strict mode - reject on ANY validation error
    strict_mode: bool = True

    # Allow canonicalization (auto-correction)
    allow_canonicalization: bool = True

    # Allow fallback strategies
    allow_fallback: bool = True

    # Require topics/categories to be from vocabulary
    enforce_vocabulary: bool = True

    # Outbound policies
    redact_sensitive_fields: list[str] = field(
        default_factory=lambda: ["embedding", "robust_reinforcement"]
    )


# =============================================================================
# SVL Gate - The Kernel
# =============================================================================


class SVLGate:
    """The Semantic Validation Layer Gate - Mandatory kernel for all data flows.

    This class implements the three core SVL guarantees:

    1. CANONICALIZATION
       - Normalizes heterogeneous inputs to unified schema
       - Maps close-match values to vocabulary terms
       - Fills in missing optional fields with defaults

    2. POLICY ENFORCEMENT
       - Validates ALL data against SVL vocabulary
       - No bypass path exists (validate=False is not an option)
       - Rejects data that violates policies

    3. GOVERNANCE CHOKE POINT
       - Single entry/exit point for all data
       - Full audit trail of all decisions
       - Retry and fallback strategies for resilience

    Example:
        gate = SVLGate(svl=svl)

        # Process LLM output (inbound)
        result = gate.process_inbound(
            llm_output={"content": "...", "memory_type": "preference"},
            user_id="user123",
            llm_call=my_llm_function,
        )

        if result.success:
            # Safe to store
            memory = Memory.from_dict(result.memory)

        # Process for LLM input (outbound)
        result = gate.process_outbound(memory)
        context = result.memory  # Safe to send to LLM
    """

    def __init__(
        self,
        svl: Any,  # SharedVocabularyLayer
        policy: GatePolicy | None = None,
        retry_config: RetryConfig | None = None,
    ):
        """Initialize the SVL Gate.

        Args:
            svl: SharedVocabularyLayer for vocabulary access
            policy: Gate policy configuration
            retry_config: Retry strategy configuration
        """
        self._svl = svl
        self._policy = policy or GatePolicy()
        self._retry_config = retry_config or RetryConfig()

        # Statistics
        self._stats = {
            "total_inbound": 0,
            "total_outbound": 0,
            "accepted": 0,
            "rejected": 0,
            "canonicalized": 0,
            "retried": 0,
            "fallback_used": 0,
            "violations_by_type": {},
        }

        # Build canonicalization maps for fuzzy matching
        self._build_canonicalization_maps()

    def _build_canonicalization_maps(self) -> None:
        """Build maps for fuzzy matching and canonicalization."""
        # Topic variations -> canonical topic
        self._topic_map: dict[str, str] = {}
        for topic in self._svl.schema.get_all_topics():
            # Add exact match
            self._topic_map[topic.lower()] = topic
            # Add variations
            self._topic_map[topic.lower().replace("_", " ")] = topic
            self._topic_map[topic.lower().replace("-", " ")] = topic
            self._topic_map[topic.lower().replace("_", "")] = topic

        # Category variations -> canonical category
        self._category_map: dict[str, str] = {}
        for category in self._svl.schema.get_all_categories():
            self._category_map[category.lower()] = category
            self._category_map[category.lower().replace("_", " ")] = category
            self._category_map[category.lower().replace("-", " ")] = category

        # Memory type variations -> canonical
        self._memory_type_map: dict[str, str] = {}
        for mtype in self._svl.schema.memory_types:
            self._memory_type_map[mtype.lower()] = mtype
            # Common variations
            if mtype == "episodic":
                self._memory_type_map["episode"] = mtype
                self._memory_type_map["event"] = mtype
            elif mtype == "semantic":
                self._memory_type_map["fact"] = mtype
                self._memory_type_map["knowledge"] = mtype
            elif mtype == "procedural":
                self._memory_type_map["procedure"] = mtype
                self._memory_type_map["howto"] = mtype
                self._memory_type_map["how-to"] = mtype

        # Sentiment variations
        self._sentiment_map: dict[str, str] = {
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral",
            "mixed": "mixed",
            "happy": "positive",
            "sad": "negative",
            "angry": "negative",
            "frustrated": "negative",
            "satisfied": "positive",
            "good": "positive",
            "bad": "negative",
        }

        # Access level variations
        self._access_level_map: dict[str, str] = {
            "private": "private",
            "team": "team",
            "shared": "shared",
            "global": "global",
            "public": "global",
            "personal": "private",
        }

    # =========================================================================
    # INBOUND PROCESSING (LLM -> Memory System)
    # =========================================================================

    def process_inbound(
        self,
        llm_output: dict[str, Any] | str,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        llm_call: Callable[[str], str] | None = None,
    ) -> GateResult:
        """Process inbound data from LLM to memory system.

        This is the ONLY way to get data into FLR/CLST.
        All data must pass through this gate.

        Args:
            llm_output: Raw output from LLM (JSON dict or string)
            user_id: User identifier (required)
            agent_id: Agent identifier (optional)
            session_id: Session identifier (optional)
            llm_call: Optional LLM function for retry strategies

        Returns:
            GateResult with decision and processed memory
        """
        start_time = time.time()
        self._stats["total_inbound"] += 1

        # Step 1: Parse input
        if isinstance(llm_output, str):
            try:
                data = self._parse_json(llm_output)
            except ValueError as e:
                return self._create_reject_result(
                    errors=[
                        ValidationError(
                            violation=PolicyViolation.INVALID_JSON,
                            field="input",
                            message=str(e),
                            value=llm_output[:100] if len(llm_output) > 100 else llm_output,
                        )
                    ],
                    start_time=start_time,
                )
        else:
            data = dict(llm_output)

        # Step 2: Inject required context
        data["user_id"] = user_id
        if agent_id:
            data["agent_id"] = agent_id
        if session_id:
            data["session_id"] = session_id
        if "memory_id" not in data or not data["memory_id"]:
            data["memory_id"] = f"mem_{uuid.uuid4().hex[:12]}"
        if "created_at" not in data:
            data["created_at"] = datetime.now(timezone.utc).isoformat()
        if "vocabulary_version" not in data:
            data["vocabulary_version"] = self._svl.schema.version

        # Step 3: Canonicalize (transform to unified representation)
        if self._policy.allow_canonicalization:
            data, changes = self._canonicalize(data)
        else:
            changes = []

        # Step 4: Validate against policies
        errors = self._validate(data)

        # Step 5: Handle validation errors
        if errors:
            # Try retry strategies if LLM call is available
            if llm_call and self._policy.allow_fallback:
                retry_result = self._retry_with_strategies(
                    original_data=data,
                    errors=errors,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    llm_call=llm_call,
                    start_time=start_time,
                )
                if retry_result.success:
                    return retry_result

            # Check if we can use fallback strategies without LLM
            if self._policy.allow_fallback:
                fallback_result = self._apply_fallback_strategies(
                    data=data,
                    errors=errors,
                    start_time=start_time,
                )
                if fallback_result.success:
                    return fallback_result

            # Strict mode: reject
            if self._policy.strict_mode:
                self._stats["rejected"] += 1
                return self._create_reject_result(
                    errors=errors,
                    start_time=start_time,
                )

        # Step 6: Success!
        self._stats["accepted"] += 1
        if changes:
            self._stats["canonicalized"] += 1

        return GateResult(
            success=True,
            decision=GateDecision.CANONICALIZE if changes else GateDecision.ACCEPT,
            memory=data,
            canonicalized=bool(changes),
            canonicalization_changes=changes,
            quality_score=1.0,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    # =========================================================================
    # OUTBOUND PROCESSING (Memory System -> LLM)
    # =========================================================================

    def process_outbound(
        self,
        memory: Any,  # Memory or dict
        include_metadata: bool = True,
    ) -> GateResult:
        """Process outbound data from memory system to LLM.

        This validates and sanitizes data before sending to LLM.

        Args:
            memory: Memory object or dict to process
            include_metadata: Include full metadata or just core fields

        Returns:
            GateResult with sanitized memory data
        """
        start_time = time.time()
        self._stats["total_outbound"] += 1

        # Convert to dict if needed
        if hasattr(memory, "to_dict"):
            data = memory.to_dict()
        else:
            data = dict(memory)

        # Validate the stored memory
        errors = self._validate(data, is_outbound=True)
        if errors:
            logger.warning(
                "Outbound memory has validation errors: %s",
                [e.message for e in errors],
            )

        # Redact sensitive fields
        for field_name in self._policy.redact_sensitive_fields:
            data.pop(field_name, None)

        # Optionally trim metadata for context window efficiency
        if not include_metadata:
            # Keep only core fields for context
            core_fields = [
                "memory_id",
                "content",
                "memory_type",
                "topics",
                "categories",
                "importance",
                "created_at",
            ]
            data = {k: v for k, v in data.items() if k in core_fields}

        return GateResult(
            success=True,
            decision=GateDecision.ACCEPT,
            memory=data,
            warnings=[e.message for e in errors] if errors else [],
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    # =========================================================================
    # CANONICALIZATION
    # =========================================================================

    def _canonicalize(self, data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Canonicalize data to unified representation.

        This transforms heterogeneous inputs into the canonical SVL schema.

        Args:
            data: Input data to canonicalize

        Returns:
            (canonicalized_data, list_of_changes)
        """
        changes = []
        result = dict(data)

        # Canonicalize memory_type
        if "memory_type" in result:
            original = result["memory_type"]
            canonical = self._canonicalize_value(original, self._memory_type_map, "episodic")
            if canonical != original:
                result["memory_type"] = canonical
                changes.append(f"memory_type: '{original}' -> '{canonical}'")

        # Canonicalize topics
        if "topics" in result and isinstance(result["topics"], list):
            canonical_topics = []
            for topic in result["topics"]:
                canonical = self._canonicalize_value(topic, self._topic_map, None)
                if canonical:
                    if canonical != topic:
                        changes.append(f"topic: '{topic}' -> '{canonical}'")
                    canonical_topics.append(canonical)
                else:
                    # Keep original if no match (will be caught in validation)
                    canonical_topics.append(topic)
            result["topics"] = canonical_topics

        # Canonicalize categories
        if "categories" in result and isinstance(result["categories"], list):
            canonical_categories = []
            for category in result["categories"]:
                canonical = self._canonicalize_value(category, self._category_map, None)
                if canonical:
                    if canonical != category:
                        changes.append(f"category: '{category}' -> '{canonical}'")
                    canonical_categories.append(canonical)
                else:
                    canonical_categories.append(category)
            result["categories"] = canonical_categories

        # Canonicalize sentiment
        if "sentiment" in result:
            original = result["sentiment"]
            canonical = self._canonicalize_value(original, self._sentiment_map, "neutral")
            if canonical != original:
                result["sentiment"] = canonical
                changes.append(f"sentiment: '{original}' -> '{canonical}'")

        # Canonicalize access_level
        if "access_level" in result:
            original = result["access_level"]
            canonical = self._canonicalize_value(original, self._access_level_map, "private")
            if canonical != original:
                result["access_level"] = canonical
                changes.append(f"access_level: '{original}' -> '{canonical}'")

        # Normalize importance to 0-1 range
        if "importance" in result:
            original = result["importance"]
            try:
                value = float(original)
                # Clamp to valid range
                if value < 0:
                    result["importance"] = 0.0
                    changes.append(f"importance: {original} -> 0.0 (clamped)")
                elif value > 1:
                    result["importance"] = 1.0
                    changes.append(f"importance: {original} -> 1.0 (clamped)")
                else:
                    result["importance"] = value
            except (ValueError, TypeError):
                result["importance"] = 0.5
                changes.append(f"importance: '{original}' -> 0.5 (default)")

        # Ensure required list fields exist
        for list_field in ["topics", "categories", "entities"]:
            if list_field not in result:
                result[list_field] = []
            elif not isinstance(result[list_field], list):
                original = result[list_field]
                result[list_field] = [result[list_field]] if result[list_field] else []
                changes.append(f"{list_field}: converted to list")

        # Set defaults for optional fields
        if "sentiment" not in result:
            result["sentiment"] = "neutral"
        if "access_level" not in result:
            result["access_level"] = "private"
        if "importance" not in result:
            result["importance"] = 0.5
        if "reinforcement_score" not in result:
            result["reinforcement_score"] = 0.0
        if "access_count" not in result:
            result["access_count"] = 0

        return result, changes

    def _canonicalize_value(
        self,
        value: str,
        mapping: dict[str, str],
        default: str | None,
    ) -> str | None:
        """Canonicalize a single value using a mapping.

        Args:
            value: Value to canonicalize
            mapping: Mapping of variations to canonical values
            default: Default if no match found

        Returns:
            Canonical value or default
        """
        if not value:
            return default

        # Try exact match first
        if value in mapping.values():
            return value

        # Try lowercase match
        lower_value = value.lower().strip()
        if lower_value in mapping:
            return mapping[lower_value]

        # Try without common separators
        normalized = lower_value.replace("_", "").replace("-", "").replace(" ", "")
        for key, canonical in mapping.items():
            if key.replace("_", "").replace("-", "").replace(" ", "") == normalized:
                return canonical

        return default

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _validate(
        self,
        data: dict[str, Any],
        is_outbound: bool = False,
    ) -> list[ValidationError]:
        """Validate data against SVL policies.

        Args:
            data: Data to validate
            is_outbound: True if validating outbound data (less strict)

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required fields
        for required_field in self._policy.required_fields:
            if required_field not in data or data[required_field] is None:
                errors.append(
                    ValidationError(
                        violation=PolicyViolation.MISSING_REQUIRED_FIELD,
                        field=required_field,
                        message=f"Missing required field: {required_field}",
                    )
                )

        # Skip further validation if missing required fields
        if errors:
            return errors

        # Validate content length
        content = data.get("content", "")
        if len(content) < self._policy.min_content_length:
            errors.append(
                ValidationError(
                    violation=PolicyViolation.CONTENT_TOO_SHORT,
                    field="content",
                    message=f"Content too short (min {self._policy.min_content_length})",
                    value=len(content),
                )
            )
        if len(content) > self._policy.max_content_length:
            errors.append(
                ValidationError(
                    violation=PolicyViolation.CONTENT_TOO_LONG,
                    field="content",
                    message=f"Content too long (max {self._policy.max_content_length})",
                    value=len(content),
                )
            )

        # Validate memory_type
        memory_type = data.get("memory_type", "")
        if memory_type not in self._svl.schema.memory_types:
            errors.append(
                ValidationError(
                    violation=PolicyViolation.INVALID_MEMORY_TYPE,
                    field="memory_type",
                    message=f"Invalid memory_type: {memory_type}",
                    value=memory_type,
                    allowed_values=self._svl.schema.memory_types,
                )
            )

        # Validate topics (only if vocabulary enforcement is enabled)
        if self._policy.enforce_vocabulary:
            valid_topics = set(self._svl.schema.get_all_topics())
            if valid_topics:  # Only check if vocabulary has topics defined
                topics = data.get("topics", [])
                invalid_topics = [t for t in topics if t not in valid_topics]
                if invalid_topics:
                    errors.append(
                        ValidationError(
                            violation=PolicyViolation.INVALID_TOPIC,
                            field="topics",
                            message=f"Invalid topics: {invalid_topics}",
                            value=invalid_topics,
                            allowed_values=list(valid_topics)[:20],  # Limit for readability
                        )
                    )

            # Validate categories
            valid_categories = set(self._svl.schema.get_all_categories())
            if valid_categories:
                categories = data.get("categories", [])
                invalid_categories = [c for c in categories if c not in valid_categories]
                if invalid_categories:
                    errors.append(
                        ValidationError(
                            violation=PolicyViolation.INVALID_CATEGORY,
                            field="categories",
                            message=f"Invalid categories: {invalid_categories}",
                            value=invalid_categories,
                            allowed_values=list(valid_categories)[:20],
                        )
                    )

        # Validate sentiment
        sentiment = data.get("sentiment")
        if sentiment and sentiment not in self._svl.schema.sentiments:
            errors.append(
                ValidationError(
                    violation=PolicyViolation.INVALID_SENTIMENT,
                    field="sentiment",
                    message=f"Invalid sentiment: {sentiment}",
                    value=sentiment,
                    allowed_values=self._svl.schema.sentiments,
                )
            )

        # Validate access_level
        access_level = data.get("access_level")
        if access_level and access_level not in self._svl.schema.access_levels:
            errors.append(
                ValidationError(
                    violation=PolicyViolation.INVALID_ACCESS_LEVEL,
                    field="access_level",
                    message=f"Invalid access_level: {access_level}",
                    value=access_level,
                    allowed_values=self._svl.schema.access_levels,
                )
            )

        # Validate importance range
        importance = data.get("importance")
        if importance is not None:
            try:
                imp_value = float(importance)
                if imp_value < 0 or imp_value > 1:
                    errors.append(
                        ValidationError(
                            violation=PolicyViolation.INVALID_IMPORTANCE,
                            field="importance",
                            message=f"Importance must be between 0 and 1: {importance}",
                            value=importance,
                        )
                    )
            except (ValueError, TypeError):
                errors.append(
                    ValidationError(
                        violation=PolicyViolation.INVALID_IMPORTANCE,
                        field="importance",
                        message=f"Importance must be a number: {importance}",
                        value=importance,
                    )
                )

        # Validate semantic_metadata if present
        if data.get("semantic_metadata"):
            is_valid, meta_errors = self._svl.validate_metadata(data["semantic_metadata"])
            if not is_valid:
                for error_msg in meta_errors:
                    errors.append(
                        ValidationError(
                            violation=PolicyViolation.SEMANTIC_METADATA_ERROR,
                            field="semantic_metadata",
                            message=error_msg,
                        )
                    )

        # Track violation statistics
        for error in errors:
            violation_key = error.violation.value
            self._stats["violations_by_type"][violation_key] = (
                self._stats["violations_by_type"].get(violation_key, 0) + 1
            )

        return errors

    # =========================================================================
    # RETRY STRATEGIES
    # =========================================================================

    def _retry_with_strategies(
        self,
        original_data: dict[str, Any],
        errors: list[ValidationError],
        user_id: str,
        agent_id: str | None,
        session_id: str | None,
        llm_call: Callable[[str], str],
        start_time: float,
    ) -> GateResult:
        """Apply retry strategies when validation fails.

        Strategy order:
        1. Retry with error feedback - Tell LLM what failed
        2. Retry with simplified schema - Fewer required fields
        3. Rule-based fallback - Pattern matching

        Args:
            original_data: Original data that failed validation
            errors: Validation errors
            user_id: User ID
            agent_id: Agent ID
            session_id: Session ID
            llm_call: Function to call LLM
            start_time: Start time for latency tracking

        Returns:
            GateResult (success or failure)
        """
        retry_count = 0
        last_errors = errors

        # Strategy 1: Retry with error feedback
        if self._retry_config.retry_with_error_feedback:
            for attempt in range(self._retry_config.max_retries):
                retry_count += 1
                self._stats["retried"] += 1

                prompt = self._create_retry_prompt(
                    original_content=original_data.get("content", ""),
                    errors=last_errors,
                    attempt=attempt + 1,
                )

                try:
                    response = llm_call(prompt)
                    new_data = self._parse_json(response)

                    # Inject context
                    new_data["user_id"] = user_id
                    if agent_id:
                        new_data["agent_id"] = agent_id
                    if session_id:
                        new_data["session_id"] = session_id
                    new_data["memory_id"] = original_data.get("memory_id")
                    new_data["created_at"] = original_data.get("created_at")

                    # Canonicalize and validate
                    new_data, changes = self._canonicalize(new_data)
                    new_errors = self._validate(new_data)

                    if not new_errors:
                        return GateResult(
                            success=True,
                            decision=GateDecision.RETRY,
                            memory=new_data,
                            canonicalized=bool(changes),
                            canonicalization_changes=changes,
                            retry_count=retry_count,
                            strategy_used="retry_with_errors",
                            quality_score=0.9 - (attempt * 0.1),
                            processing_time_ms=(time.time() - start_time) * 1000,
                        )

                    last_errors = new_errors

                except Exception as e:
                    logger.warning("Retry attempt %d failed: %s", attempt + 1, e)

        # Strategy 2: Retry with simplified schema
        if self._retry_config.retry_with_simplified_schema:
            retry_count += 1
            self._stats["retried"] += 1

            try:
                prompt = self._create_simplified_prompt(
                    content=original_data.get("content", ""),
                )
                response = llm_call(prompt)
                new_data = self._parse_json(response)

                # Apply minimal required fields
                new_data["user_id"] = user_id
                new_data["memory_id"] = original_data.get("memory_id")
                new_data["created_at"] = original_data.get("created_at")

                # Canonicalize
                new_data, changes = self._canonicalize(new_data)

                # Lenient validation - only check required fields
                if all(new_data.get(f) for f in ["content", "memory_type", "user_id"]):
                    return GateResult(
                        success=True,
                        decision=GateDecision.RETRY,
                        memory=new_data,
                        canonicalized=bool(changes),
                        canonicalization_changes=changes,
                        retry_count=retry_count,
                        strategy_used="retry_simplified",
                        quality_score=0.7,
                        warnings=["Used simplified schema - some fields may use defaults"],
                        processing_time_ms=(time.time() - start_time) * 1000,
                    )

            except Exception as e:
                logger.warning("Simplified retry failed: %s", e)

        # No LLM retry worked
        return GateResult(
            success=False,
            decision=GateDecision.REJECT,
            errors=last_errors,
            retry_count=retry_count,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    def _apply_fallback_strategies(
        self,
        data: dict[str, Any],
        errors: list[ValidationError],
        start_time: float,
    ) -> GateResult:
        """Apply non-LLM fallback strategies.

        Args:
            data: Data with validation errors
            errors: Validation errors
            start_time: Start time for latency tracking

        Returns:
            GateResult (success or failure)
        """
        self._stats["fallback_used"] += 1

        # Strategy: Fix specific errors with defaults/rules
        fixed_data = dict(data)
        fixed_errors = []

        for error in errors:
            if error.violation == PolicyViolation.INVALID_TOPIC:
                # Remove invalid topics, keep valid ones or use default
                valid_topics = set(self._svl.schema.get_all_topics())
                fixed_data["topics"] = [
                    t for t in fixed_data.get("topics", []) if t in valid_topics
                ]
                if not fixed_data["topics"]:
                    fixed_data["topics"] = ["general"] if "general" in valid_topics else []

            elif error.violation == PolicyViolation.INVALID_CATEGORY:
                valid_categories = set(self._svl.schema.get_all_categories())
                fixed_data["categories"] = [
                    c for c in fixed_data.get("categories", []) if c in valid_categories
                ]

            elif error.violation == PolicyViolation.INVALID_MEMORY_TYPE:
                fixed_data["memory_type"] = "episodic"

            elif error.violation == PolicyViolation.INVALID_SENTIMENT:
                fixed_data["sentiment"] = "neutral"

            elif error.violation == PolicyViolation.INVALID_ACCESS_LEVEL:
                fixed_data["access_level"] = "private"

            elif error.violation == PolicyViolation.INVALID_IMPORTANCE:
                fixed_data["importance"] = 0.5

            else:
                # Can't fix this error
                fixed_errors.append(error)

        # Check if all errors were fixed
        if not fixed_errors:
            return GateResult(
                success=True,
                decision=GateDecision.FALLBACK,
                memory=fixed_data,
                canonicalized=True,
                canonicalization_changes=["Applied fallback defaults"],
                strategy_used="rule_based_fallback",
                quality_score=0.5,
                needs_review=True,
                warnings=["Used fallback strategy - some values were auto-corrected"],
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        return GateResult(
            success=False,
            decision=GateDecision.REJECT,
            errors=fixed_errors,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _create_retry_prompt(
        self,
        original_content: str,
        errors: list[ValidationError],
        attempt: int,
    ) -> str:
        """Create a retry prompt with error feedback."""
        error_list = "\n".join([f"  - {e.message}" for e in errors])

        # Include allowed values where available
        value_hints = []
        for error in errors:
            if error.allowed_values:
                value_hints.append(f"  Valid {error.field}: {', '.join(error.allowed_values[:10])}")

        hints_str = "\n".join(value_hints) if value_hints else ""

        return f"""The previous metadata extraction had validation errors. Please fix them.

ERRORS:
{error_list}

{f"ALLOWED VALUES:{chr(10)}{hints_str}" if hints_str else ""}

Original content: {original_content}

Return ONLY valid JSON with these fields:
- content: The memory content (string)
- memory_type: One of {self._svl.schema.memory_types}
- topics: List of valid topics
- categories: List of valid categories
- importance: Number between 0 and 1
- sentiment: One of {self._svl.schema.sentiments}

Attempt {attempt}/{self._retry_config.max_retries}. Be precise with vocabulary values.

JSON only, no explanation:"""

    def _create_simplified_prompt(self, content: str) -> str:
        """Create a simplified extraction prompt."""
        return f"""Extract basic metadata from this content. Use simple, safe values.

Content: {content}

Return JSON:
{{
    "content": "...",
    "memory_type": "episodic",
    "topics": [],
    "categories": [],
    "importance": 0.5,
    "sentiment": "neutral"
}}

JSON only:"""

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Parse JSON from text, handling common LLM output issues."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        import re

        json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from: {text[:200]}...")

    def _create_reject_result(
        self,
        errors: list[ValidationError],
        start_time: float,
    ) -> GateResult:
        """Create a rejection result."""
        return GateResult(
            success=False,
            decision=GateDecision.REJECT,
            errors=errors,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get gate statistics."""
        total = self._stats["total_inbound"] + self._stats["total_outbound"]
        return {
            "total_processed": total,
            "total_inbound": self._stats["total_inbound"],
            "total_outbound": self._stats["total_outbound"],
            "accepted": self._stats["accepted"],
            "rejected": self._stats["rejected"],
            "canonicalized": self._stats["canonicalized"],
            "retried": self._stats["retried"],
            "fallback_used": self._stats["fallback_used"],
            "acceptance_rate": (
                self._stats["accepted"] / self._stats["total_inbound"]
                if self._stats["total_inbound"] > 0
                else 0
            ),
            "violations_by_type": self._stats["violations_by_type"],
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            "total_inbound": 0,
            "total_outbound": 0,
            "accepted": 0,
            "rejected": 0,
            "canonicalized": 0,
            "retried": 0,
            "fallback_used": 0,
            "violations_by_type": {},
        }
