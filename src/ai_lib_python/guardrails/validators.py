"""
High-level validators for content safety and compliance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_lib_python.guardrails.base import (
    CompositeGuardrail,
    Guardrail,
    GuardrailResult,
    GuardrailSeverity,
)
from ai_lib_python.guardrails.filters import (
    EmailFilter,
    KeywordFilter,
    LengthFilter,
    ProfanityFilter,
    RegexFilter,
    UrlFilter,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class ContentValidator:
    """High-level content validator with pre-configured guardrails.

    Provides ready-to-use validators for common use cases.

    Example:
        >>> validator = ContentValidator.create_input_validator()
        >>> result = validator.check("User input here")
        >>> if not result.is_safe:
        ...     print(f"Violations: {[v.message for v in result.violations]}")
        >>>
        >>> # Filter content
        >>> filtered = validator.filter("Input with sensitive data")
    """

    def __init__(self, guardrails: Sequence[Guardrail]) -> None:
        """Initialize content validator.

        Args:
            guardrails: List of guardrails to apply
        """
        self._composite = CompositeGuardrail(
            rule_id="content-validator",
            guardrails=list(guardrails),
            severity=GuardrailSeverity.WARNING,
            stop_on_first=False,
        )

    def check(self, content: str) -> GuardrailResult:
        """Check content against all guardrails.

        Args:
            content: Text content to validate

        Returns:
            GuardrailResult with all violations
        """
        return self._composite.check(content)

    def filter(self, content: str) -> str:
        """Filter content by applying all guardrails.

        Args:
            content: Text content to filter

        Returns:
            Filtered content
        """
        return self._composite.filter(content)

    def add_guardrail(self, guardrail: Guardrail) -> None:
        """Add a guardrail to the validator."""
        self._composite.add_guardrail(guardrail)

    def remove_guardrail(self, rule_id: str) -> bool:
        """Remove a guardrail by rule ID."""
        return self._composite.remove_guardrail(rule_id)

    @classmethod
    def create_input_validator(
        cls,
        severity: GuardrailSeverity = GuardrailSeverity.WARNING,
    ) -> ContentValidator:
        """Create a validator for user input.

        Filters for common input issues:
        - Length limits
        - Profanity
        - URLs
        - Emails

        Args:
            severity: Severity level for violations

        Returns:
            Configured ContentValidator
        """
        guardrails: list[Guardrail] = [
            LengthFilter(
                rule_id="input-length",
                min_length=1,
                max_length=10000,
                severity=severity,
                count_mode="chars",
            ),
            ProfanityFilter(
                rule_id="input-profanity",
                severity=severity,
            ),
            UrlFilter(
                rule_id="input-urls",
                action="allow",
                severity=severity,
                allowed_domains=["example.com"],  # Add your allowed domains
            ),
            EmailFilter(
                rule_id="input-emails",
                action="allow",
                severity=severity,
                allowed_domains=["example.com"],  # Add your allowed domains
            ),
        ]

        return cls(guardrails)

    @classmethod
    def create_output_validator(
        cls,
        severity: GuardrailSeverity = GuardrailSeverity.WARNING,
    ) -> ContentValidator:
        """Create a validator for AI model output.

        Filters for common output issues:
        - Excessive length
        - Profanity
        - Malicious URLs
        - PII patterns

        Args:
            severity: Severity level for violations

        Returns:
            Configured ContentValidator
        """
        guardrails: list[Guardrail] = [
            LengthFilter(
                rule_id="output-length",
                min_length=1,
                max_length=50000,
                severity=severity,
                count_mode="chars",
            ),
            ProfanityFilter(
                rule_id="output-profanity",
                severity=severity,
            ),
            UrlFilter(
                rule_id="output-urls",
                action="deny",
                severity=severity,
                blocked_domains=["malicious.example.com", "spam.example.com"],
            ),
            # Block common PII patterns
            RegexFilter(
                rule_id="output-credit-card",
                pattern=r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
                severity=GuardrailSeverity.CRITICAL,
                replacement="[CARD REDECTED]",
            ),
            RegexFilter(
                rule_id="output-ssn",
                pattern=r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
                severity=GuardrailSeverity.CRITICAL,
                replacement="[SSN REDECTED]",
            ),
        ]

        return cls(guardrails)

    @classmethod
    def create_pii_validator(
        cls,
        severity: GuardrailSeverity = GuardrailSeverity.CRITICAL,
    ) -> ContentValidator:
        """Create a validator focusing on PII detection.

        Detects and blocks Personally Identifiable Information.

        Args:
            severity: Severity level for violations

        Returns:
            Configured ContentValidator
        """
        guardrails: list[Guardrail] = [
            # Email addresses
            EmailFilter(
                rule_id="pii-email",
                action="block",
                severity=severity,
            ),
            # Phone numbers (various formats)
            RegexFilter(
                rule_id="pii-phone",
                pattern=r"\b\d{3}[-\s.]?\d{3}[-\s.]?\d{4}\b",
                severity=severity,
                replacement="[PHONE REDACTED]",
            ),
            # Credit card numbers
            RegexFilter(
                rule_id="pii-credit-card",
                pattern=r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
                severity=severity,
                replacement="[CARD REDACTED]",
            ),
            # Social Security Number
            RegexFilter(
                rule_id="pii-ssn",
                pattern=r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
                severity=severity,
                replacement="[SSN REDACTED]",
            ),
            # IP addresses
            RegexFilter(
                rule_id="pii-ip",
                pattern=r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
                severity=severity,
                replacement="[IP REDACTED]",
            ),
        ]

        return cls(guardrails)

    @classmethod
    def create_custom_validator(
        cls,
        forbidden_keywords: list[str] | None = None,
        allowed_domains: list[str] | None = None,
        max_length: int | None = None,
        severity: GuardrailSeverity = GuardrailSeverity.WARNING,
    ) -> ContentValidator:
        """Create a custom validator with specific rules.

        Args:
            forbidden_keywords: List of keywords to block
            allowed_domains: List of allowed URL/email domains
            max_length: Maximum content length
            severity: Severity level for violations

        Returns:
            Configured ContentValidator
        """
        guardrails: list[Guardrail] = []

        if forbidden_keywords:
            guardrails.append(
                KeywordFilter(
                    rule_id="custom-keywords",
                    keywords=forbidden_keywords,
                    severity=severity,
                )
            )

        if allowed_domains:
            guardrails.extend([
                UrlFilter(
                    rule_id="custom-urls",
                    action="allow",
                    severity=severity,
                    allowed_domains=allowed_domains,
                ),
                EmailFilter(
                    rule_id="custom-emails",
                    action="allow",
                    severity=severity,
                    allowed_domains=allowed_domains,
                ),
            ])

        if max_length:
            guardrails.append(
                LengthFilter(
                    rule_id="custom-length",
                    max_length=max_length,
                    severity=severity,
                )
            )

        return cls(guardrails)

    @classmethod
    def create_code_validator(
        cls,
        severity: GuardrailSeverity = GuardrailSeverity.INFO,
    ) -> ContentValidator:
        """Create a validator for code snippets.

        Less strict validation suitable for code-related content.

        Args:
            severity: Severity level for violations

        Returns:
            Configured ContentValidator
        """
        guardrails: list[Guardrail] = [
            # Allow URLs (useful for code examples)
            UrlFilter(
                rule_id="code-urls",
                action="allow",
                severity=severity,
                allowed_domains=[],  # Allow all domains
            ),
            # Allow emails (common in code)
            EmailFilter(
                rule_id="code-emails",
                action="allow",
                severity=severity,
                allowed_domains=[],  # Allow all domains
            ),
            # Length limit for code snippets
            LengthFilter(
                rule_id="code-length",
                min_length=1,
                max_length=100000,
                severity=severity,
                count_mode="chars",
            ),
        ]

        return cls(guardrails)


class SafetyValidator(ContentValidator):
    """Validator focused specifically on content safety.

    Detects potentially harmful or inappropriate content.

    Example:
        >>> validator = SafetyValidator()
        >>> result = validator.check("Input to check")
        >>> if not result.is_safe:
        ...     print("Content flagged as unsafe")
    """

    # Potentially harmful keywords
    _HARMFUL_KEYWORDS = [
        "bomb", "drug", "weapon", "kill",
        "illegal", "hack", "steal",
    ]

    # Self-harm indicators
    _SELF_HARM_KEYWORDS = [
        "suicide", "end it all", "kill myself",
        "want to die", "don't want to live",
    ]

    def __init__(
        self,
        severity: GuardrailSeverity = GuardrailSeverity.CRITICAL,
        block_self_harm: bool = True,
        block_harmful: bool = True,
        block_profane: bool = True,
    ) -> None:
        """Initialize safety validator.

        Args:
            severity: Severity level for violations
            block_self_harm: Block self-harm related content
            block_harmful: Block harmful content
            block_profane: Block profane content
        """
        guardrails: list[Guardrail] = []

        if block_self_harm:
            guardrails.append(
                KeywordFilter(
                    rule_id="safety-self-harm",
                    keywords=self._SELF_HARM_KEYWORDS,
                    severity=severity,
                    case_sensitive=False,
                )
            )

        if block_harmful:
            guardrails.append(
                KeywordFilter(
                    rule_id="safety-harmful",
                    keywords=self._HARMFUL_KEYWORDS,
                    severity=severity,
                    case_sensitive=False,
                )
            )

        if block_profane:
            guardrails.append(
                ProfanityFilter(
                    rule_id="safety-profanity",
                    severity=severity,
                )
            )

        super().__init__(guardrails)


class ComplianceValidator(ContentValidator):
    """Validator for regulatory compliance.

    Helps enforce compliance with data protection regulations.

    Example:
        >>> validator = ComplianceValidator(
        ...     gdpr_mode=True,
        ...     hipaa_mode=False,
        ... )
        >>> result = validator.check("Content to check")
    """

    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-\s.]?\d{3}[-\s.]?\d{4}\b",
        "ssn": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    }

    def __init__(
        self,
        gdpr_mode: bool = True,
        hipaa_mode: bool = False,
        severity: GuardrailSeverity = GuardrailSeverity.CRITICAL,
    ) -> None:
        """Initialize compliance validator.

        Args:
            gdpr_mode: Enable GDPR compliance (detect PII)
            hipaa_mode: Enable HIPAA compliance (detect PHI)
            severity: Severity level for violations
        """
        guardrails: list[Guardrail] = []

        if gdpr_mode:
            # GDPR: Detect and block PII
            for name, pattern in self.PII_PATTERNS.items():
                guardrails.append(
                    RegexFilter(
                        rule_id=f"gdpr-{name}",
                        pattern=pattern,
                        severity=severity,
                        replacement=f"[{name.upper()} REDACTED]",
                        message=f"GDPR violation: {name} detected",
                    )
                )

        if hipaa_mode:
            # HIPAA: Additional protected health information patterns
            # (Additional patterns can be added based on requirements)
            guardrails.extend([
                RegexFilter(
                    rule_id="hipaa-medical-record",
                    pattern=r"\bMR[A-Z]?\d{6,}\b",
                    severity=severity,
                    replacement="[MRN REDACTED]",
                    message="HIPAA violation: Medical record number detected",
                ),
            ])

        super().__init__(guardrails)
