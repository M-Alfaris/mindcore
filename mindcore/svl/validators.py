"""SVL Content Validators - Security validators for content sanitization.

This module provides validators for detecting and handling:
1. PII (Personally Identifiable Information)
2. Prompt injection attempts
3. Sensitive data patterns

These validators can be integrated with the SVL Gate for enhanced security.

Example:
    from mindcore.svl.validators import (
        ContentValidator,
        PIIDetector,
        PromptInjectionDetector,
        SensitivePatternDetector,
    )

    # Create composite validator
    validator = ContentValidator(
        validators=[
            PIIDetector(),
            PromptInjectionDetector(),
            SensitivePatternDetector(),
        ]
    )

    # Validate content
    result = validator.validate("User email is test@example.com")
    if not result.is_valid:
        print(f"Issues found: {result.issues}")

Design Notes:
- Validators are designed to be composable
- Each validator returns detailed issue information
- Validators can redact, warn, or reject based on configuration
- Performance-optimized with compiled regex patterns
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class ValidationAction(str, Enum):
    """Action to take when validation issue is found."""

    REJECT = "reject"  # Reject the content entirely
    REDACT = "redact"  # Redact the sensitive content
    WARN = "warn"  # Log warning but allow


class IssueType(str, Enum):
    """Types of validation issues."""

    PII_EMAIL = "pii_email"
    PII_PHONE = "pii_phone"
    PII_SSN = "pii_ssn"
    PII_CREDIT_CARD = "pii_credit_card"
    PII_IP_ADDRESS = "pii_ip_address"
    PII_API_KEY = "pii_api_key"
    PII_PASSWORD = "pii_password"  # noqa: S105 # nosec B105 - This is a category name, not a password
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK_ATTEMPT = "jailbreak_attempt"
    SENSITIVE_KEYWORD = "sensitive_keyword"
    SQL_INJECTION = "sql_injection"
    SCRIPT_INJECTION = "script_injection"


@dataclass
class ValidationIssue:
    """A single validation issue found in content."""

    issue_type: IssueType
    description: str
    matched_text: str = ""
    position: int = -1
    severity: str = "medium"  # low, medium, high, critical
    suggested_action: ValidationAction = ValidationAction.WARN

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_type": self.issue_type.value,
            "description": self.description,
            "matched_text": self.matched_text[:50] + "..."
            if len(self.matched_text) > 50
            else self.matched_text,
            "position": self.position,
            "severity": self.severity,
            "suggested_action": self.suggested_action.value,
        }


@dataclass
class ValidationResult:
    """Result of content validation."""

    is_valid: bool
    content: str  # Original or redacted content
    issues: list[ValidationIssue] = field(default_factory=list)
    was_redacted: bool = False
    redaction_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "issues": [i.to_dict() for i in self.issues],
            "was_redacted": self.was_redacted,
            "redaction_count": self.redaction_count,
        }


class BaseValidator(ABC):
    """Base class for content validators."""

    @abstractmethod
    def validate(self, content: str) -> list[ValidationIssue]:
        """Validate content and return list of issues found.

        Args:
            content: Text content to validate

        Returns:
            List of ValidationIssue objects found
        """

    @abstractmethod
    def redact(self, content: str) -> tuple[str, int]:
        """Redact sensitive content.

        Args:
            content: Text content to redact

        Returns:
            Tuple of (redacted_content, redaction_count)
        """


class PIIDetector(BaseValidator):
    """Detector for Personally Identifiable Information (PII)."""

    def __init__(
        self,
        detect_email: bool = True,
        detect_phone: bool = True,
        detect_ssn: bool = True,
        detect_credit_card: bool = True,
        detect_ip_address: bool = True,
        detect_api_keys: bool = True,
        detect_passwords: bool = True,
        action: ValidationAction = ValidationAction.WARN,
    ):
        self.detect_email = detect_email
        self.detect_phone = detect_phone
        self.detect_ssn = detect_ssn
        self.detect_credit_card = detect_credit_card
        self.detect_ip_address = detect_ip_address
        self.detect_api_keys = detect_api_keys
        self.detect_passwords = detect_passwords
        self.action = action

        # Compile regex patterns for performance
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> dict[IssueType, re.Pattern]:
        """Compile regex patterns for PII detection."""
        patterns = {}

        if self.detect_email:
            # Email pattern
            patterns[IssueType.PII_EMAIL] = re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", re.IGNORECASE
            )

        if self.detect_phone:
            # Phone patterns (US format, international, etc.)
            patterns[IssueType.PII_PHONE] = re.compile(
                r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
            )

        if self.detect_ssn:
            # Social Security Number pattern
            patterns[IssueType.PII_SSN] = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")

        if self.detect_credit_card:
            # Credit card patterns (Visa, MasterCard, Amex, etc.)
            patterns[IssueType.PII_CREDIT_CARD] = re.compile(
                r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"
            )

        if self.detect_ip_address:
            # IP address pattern (IPv4)
            patterns[IssueType.PII_IP_ADDRESS] = re.compile(
                r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
            )

        if self.detect_api_keys:
            # Common API key patterns
            patterns[IssueType.PII_API_KEY] = re.compile(
                r"\b(?:sk-[A-Za-z0-9]{32,}|"  # OpenAI
                r"AIza[A-Za-z0-9_-]{35}|"  # Google
                r"AKIA[A-Z0-9]{16}|"  # AWS Access Key
                r"ghp_[A-Za-z0-9]{36}|"  # GitHub
                r"xox[baprs]-[A-Za-z0-9-]+)"  # Slack
                r"\b",
                re.IGNORECASE,
            )

        if self.detect_passwords:
            # Password in plaintext patterns
            patterns[IssueType.PII_PASSWORD] = re.compile(
                r'(?:password|passwd|pwd|secret|token|api_key|apikey|auth)[\s:=]+["\']?([^\s"\']{8,})["\']?',
                re.IGNORECASE,
            )

        return patterns

    def validate(self, content: str) -> list[ValidationIssue]:
        """Detect PII in content."""
        issues = []

        for issue_type, pattern in self._patterns.items():
            for match in pattern.finditer(content):
                issues.append(
                    ValidationIssue(
                        issue_type=issue_type,
                        description=f"Detected {issue_type.value.replace('pii_', '').replace('_', ' ')}",
                        matched_text=match.group(),
                        position=match.start(),
                        severity="high"
                        if issue_type
                        in (IssueType.PII_SSN, IssueType.PII_CREDIT_CARD, IssueType.PII_API_KEY)
                        else "medium",
                        suggested_action=self.action,
                    )
                )

        return issues

    def redact(self, content: str) -> tuple[str, int]:
        """Redact PII from content."""
        redacted = content
        count = 0

        redaction_map = {
            IssueType.PII_EMAIL: "[EMAIL_REDACTED]",
            IssueType.PII_PHONE: "[PHONE_REDACTED]",
            IssueType.PII_SSN: "[SSN_REDACTED]",
            IssueType.PII_CREDIT_CARD: "[CARD_REDACTED]",
            IssueType.PII_IP_ADDRESS: "[IP_REDACTED]",
            IssueType.PII_API_KEY: "[API_KEY_REDACTED]",
            IssueType.PII_PASSWORD: "[PASSWORD_REDACTED]",
        }

        for issue_type, pattern in self._patterns.items():
            replacement = redaction_map.get(issue_type, "[REDACTED]")
            new_content, num = pattern.subn(replacement, redacted)
            count += num
            redacted = new_content

        return redacted, count


class PromptInjectionDetector(BaseValidator):
    """Detector for prompt injection attempts."""

    def __init__(
        self,
        sensitivity: str = "medium",  # low, medium, high
        action: ValidationAction = ValidationAction.REJECT,
    ):
        self.sensitivity = sensitivity
        self.action = action
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> list[tuple[re.Pattern, str, str]]:
        """Compile prompt injection detection patterns.

        Returns:
            List of (pattern, description, severity) tuples
        """
        patterns = []

        # Always detect these (high severity)
        high_severity = [
            (
                r"ignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions?",
                "Instruction override attempt",
            ),
            (
                r"disregard\s+(?:all\s+)?(?:previous|your)\s+(?:instructions?|rules?|guidelines?)",
                "Rule disregard attempt",
            ),
            (
                r"forget\s+(?:everything|all)\s+(?:you|I)\s+(?:said|told)",
                "Memory manipulation attempt",
            ),
            (r"you\s+are\s+now\s+(?:a\s+)?(?:different|new|evil)", "Identity manipulation attempt"),
            (
                r"pretend\s+(?:you\s+are|to\s+be)\s+(?:a\s+)?(?!helpful)",
                "Role manipulation attempt",
            ),
            (
                r"(?:system|developer|admin)\s*(?:mode|prompt|override)",
                "System mode access attempt",
            ),
            (r"bypass\s+(?:your\s+)?(?:safety|content|ethical)", "Safety bypass attempt"),
            (r"jailbreak", "Jailbreak keyword detected"),
            (r"DAN\s*(?:mode)?", "DAN jailbreak attempt"),
        ]

        for pattern_str, description in high_severity:
            patterns.append(
                (
                    re.compile(pattern_str, re.IGNORECASE),
                    description,
                    "critical",
                )
            )

        # Medium sensitivity patterns
        if self.sensitivity in ("medium", "high"):
            medium_severity = [
                (r"(?:act|behave)\s+as\s+(?:if|though)", "Behavior modification attempt"),
                (r"new\s+(?:instructions?|rules?|guidelines?)", "New instruction injection"),
                (
                    r"(?:override|replace|change)\s+(?:your\s+)?(?:rules?|instructions?)",
                    "Rule override attempt",
                ),
                (r"you\s+(?:must|should|will)\s+(?:always|never)", "Behavior forcing attempt"),
                (r"\[(?:system|SYSTEM)\]", "System tag injection"),
                (r"<\|(?:im_start|im_end|system)\|>", "Token injection attempt"),
            ]

            for pattern_str, description in medium_severity:
                patterns.append(
                    (
                        re.compile(pattern_str, re.IGNORECASE),
                        description,
                        "high",
                    )
                )

        # High sensitivity patterns (may have false positives)
        if self.sensitivity == "high":
            low_severity = [
                (r"(?:don\'?t|do\s+not)\s+(?:follow|obey|listen)", "Obedience override attempt"),
                (
                    r"(?:reveal|show|tell)\s+(?:me\s+)?(?:your\s+)?(?:system|initial|original)\s+prompt",
                    "Prompt extraction attempt",
                ),
                (
                    r"what\s+(?:are|were)\s+your\s+(?:original|initial)\s+instructions?",
                    "Instruction extraction attempt",
                ),
            ]

            for pattern_str, description in low_severity:
                patterns.append(
                    (
                        re.compile(pattern_str, re.IGNORECASE),
                        description,
                        "medium",
                    )
                )

        return patterns

    def validate(self, content: str) -> list[ValidationIssue]:
        """Detect prompt injection attempts."""
        issues = []

        for pattern, description, severity in self._patterns:
            for match in pattern.finditer(content):
                issues.append(
                    ValidationIssue(
                        issue_type=IssueType.PROMPT_INJECTION
                        if severity == "critical"
                        else IssueType.JAILBREAK_ATTEMPT,
                        description=description,
                        matched_text=match.group(),
                        position=match.start(),
                        severity=severity,
                        suggested_action=self.action,
                    )
                )

        return issues

    def redact(self, content: str) -> tuple[str, int]:
        """Redact prompt injection attempts (replaces with warning)."""
        redacted = content
        count = 0

        for pattern, _, _ in self._patterns:
            new_content, num = pattern.subn("[INJECTION_BLOCKED]", redacted)
            count += num
            redacted = new_content

        return redacted, count


class SensitivePatternDetector(BaseValidator):
    """Detector for sensitive data patterns like SQL injection, XSS, etc."""

    def __init__(
        self,
        detect_sql_injection: bool = True,
        detect_script_injection: bool = True,
        detect_sensitive_keywords: bool = True,
        action: ValidationAction = ValidationAction.WARN,
        custom_patterns: list[tuple[str, str, str]] | None = None,
    ):
        self.detect_sql_injection = detect_sql_injection
        self.detect_script_injection = detect_script_injection
        self.detect_sensitive_keywords = detect_sensitive_keywords
        self.action = action
        self.custom_patterns = custom_patterns or []
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> list[tuple[re.Pattern, IssueType, str]]:
        """Compile sensitive pattern detection patterns."""
        patterns = []

        if self.detect_sql_injection:
            sql_patterns = [
                r"(?:UNION\s+(?:ALL\s+)?SELECT|INSERT\s+INTO|DELETE\s+FROM|DROP\s+TABLE|UPDATE\s+\w+\s+SET)",
                r"(?:OR|AND)\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+['\"]?",
                r"--\s*(?:$|\n)|/\*.*?\*/",
                r";\s*(?:DROP|DELETE|TRUNCATE|ALTER)\s+",
            ]
            for pattern_str in sql_patterns:
                patterns.append(
                    (
                        re.compile(pattern_str, re.IGNORECASE),
                        IssueType.SQL_INJECTION,
                        "SQL injection pattern detected",
                    )
                )

        if self.detect_script_injection:
            script_patterns = [
                r"<script\b[^>]*>.*?</script>",
                r"javascript\s*:",
                r"on(?:load|error|click|mouseover)\s*=",
                r"<iframe\b[^>]*>",
                r"eval\s*\(",
            ]
            for pattern_str in script_patterns:
                patterns.append(
                    (
                        re.compile(pattern_str, re.IGNORECASE | re.DOTALL),
                        IssueType.SCRIPT_INJECTION,
                        "Script injection pattern detected",
                    )
                )

        if self.detect_sensitive_keywords:
            # Sensitive keywords that might indicate data leakage
            keyword_patterns = [
                r"\b(?:BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY)",
                r"\b(?:Bearer\s+[A-Za-z0-9._~+/-]+=*)",
            ]
            for pattern_str in keyword_patterns:
                patterns.append(
                    (
                        re.compile(pattern_str, re.IGNORECASE),
                        IssueType.SENSITIVE_KEYWORD,
                        "Sensitive keyword detected",
                    )
                )

        # Add custom patterns
        for pattern_str, issue_type_str, description in self.custom_patterns:
            try:
                issue_type = IssueType(issue_type_str)
            except ValueError:
                issue_type = IssueType.SENSITIVE_KEYWORD

            patterns.append(
                (
                    re.compile(pattern_str, re.IGNORECASE),
                    issue_type,
                    description,
                )
            )

        return patterns

    def validate(self, content: str) -> list[ValidationIssue]:
        """Detect sensitive patterns."""
        issues = []

        for pattern, issue_type, description in self._patterns:
            for match in pattern.finditer(content):
                issues.append(
                    ValidationIssue(
                        issue_type=issue_type,
                        description=description,
                        matched_text=match.group(),
                        position=match.start(),
                        severity="high" if issue_type == IssueType.SQL_INJECTION else "medium",
                        suggested_action=self.action,
                    )
                )

        return issues

    def redact(self, content: str) -> tuple[str, int]:
        """Redact sensitive patterns."""
        redacted = content
        count = 0

        for pattern, issue_type, _ in self._patterns:
            replacement = f"[{issue_type.value.upper()}_REDACTED]"
            new_content, num = pattern.subn(replacement, redacted)
            count += num
            redacted = new_content

        return redacted, count


class ContentValidator:
    """Composite validator that runs multiple validators."""

    def __init__(
        self,
        validators: list[BaseValidator] | None = None,
        default_action: ValidationAction = ValidationAction.WARN,
        auto_redact: bool = False,
    ):
        """Initialize the content validator.

        Args:
            validators: List of validators to run. If None, uses defaults.
            default_action: Default action when issues are found
            auto_redact: Automatically redact content before returning
        """
        self.validators = validators or [
            PIIDetector(),
            PromptInjectionDetector(),
            SensitivePatternDetector(),
        ]
        self.default_action = default_action
        self.auto_redact = auto_redact

    def validate(self, content: str) -> ValidationResult:
        """Validate content using all validators.

        Args:
            content: Text content to validate

        Returns:
            ValidationResult with all issues found
        """
        all_issues = []

        for validator in self.validators:
            issues = validator.validate(content)
            all_issues.extend(issues)

        # Determine if content should be rejected
        should_reject = any(
            issue.suggested_action == ValidationAction.REJECT for issue in all_issues
        )

        # Optionally redact content
        final_content = content
        was_redacted = False
        redaction_count = 0

        if self.auto_redact and all_issues:
            for validator in self.validators:
                final_content, count = validator.redact(final_content)
                redaction_count += count
            was_redacted = redaction_count > 0

        return ValidationResult(
            is_valid=not should_reject,
            content=final_content,
            issues=all_issues,
            was_redacted=was_redacted,
            redaction_count=redaction_count,
        )

    def validate_and_redact(self, content: str) -> ValidationResult:
        """Validate content and automatically redact sensitive data.

        This is a convenience method that always redacts, regardless of
        the auto_redact setting.

        Args:
            content: Text content to validate and redact

        Returns:
            ValidationResult with redacted content
        """
        all_issues = []
        redacted = content
        total_redactions = 0

        for validator in self.validators:
            issues = validator.validate(redacted)
            all_issues.extend(issues)

            redacted, count = validator.redact(redacted)
            total_redactions += count

        should_reject = any(
            issue.suggested_action == ValidationAction.REJECT for issue in all_issues
        )

        return ValidationResult(
            is_valid=not should_reject,
            content=redacted,
            issues=all_issues,
            was_redacted=total_redactions > 0,
            redaction_count=total_redactions,
        )


def create_default_validator(
    detect_pii: bool = True,
    detect_prompt_injection: bool = True,
    detect_sensitive_patterns: bool = True,
    pii_action: ValidationAction = ValidationAction.WARN,
    injection_action: ValidationAction = ValidationAction.REJECT,
    auto_redact: bool = False,
) -> ContentValidator:
    """Create a content validator with sensible defaults.

    Args:
        detect_pii: Enable PII detection
        detect_prompt_injection: Enable prompt injection detection
        detect_sensitive_patterns: Enable sensitive pattern detection
        pii_action: Action for PII issues
        injection_action: Action for injection issues
        auto_redact: Automatically redact sensitive content

    Returns:
        Configured ContentValidator instance
    """
    validators = []

    if detect_pii:
        validators.append(PIIDetector(action=pii_action))

    if detect_prompt_injection:
        validators.append(PromptInjectionDetector(action=injection_action))

    if detect_sensitive_patterns:
        validators.append(SensitivePatternDetector(action=pii_action))

    return ContentValidator(validators=validators, auto_redact=auto_redact)
