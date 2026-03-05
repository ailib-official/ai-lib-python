"""
Base classes for guardrail filtering and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class GuardrailSeverity(Enum):
    """Severity levels for guardrail violations."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class GuardrailViolation:
    """Represents a single guardrail rule violation."""

    rule_id: str
    message: str
    severity: GuardrailSeverity
    matched_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate severity."""
        if not isinstance(self.severity, GuardrailSeverity):
            self.severity = GuardrailSeverity(self.severity)

    def to_dict(self) -> dict[str, Any]:
        """Convert violation to dictionary."""
        return {
            "rule_id": self.rule_id,
            "message": self.message,
            "severity": self.severity.value,
            "matched_text": self.matched_text,
            "metadata": self.metadata,
        }


@dataclass
class GuardrailResult:
    """Result of applying a guardrail to content."""

    is_safe: bool
    violations: list[GuardrailViolation] = field(default_factory=list)
    filtered_content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def safe(cls, content: str | None = None) -> GuardrailResult:
        """Create a safe result."""
        return cls(is_safe=True, violations=[], filtered_content=content)

    @classmethod
    def violated(
        cls,
        violations: list[GuardrailViolation],
        filtered_content: str | None = None,
    ) -> GuardrailResult:
        """Create a violated result."""
        return cls(
            is_safe=False,
            violations=violations,
            filtered_content=filtered_content,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "is_safe": self.is_safe,
            "violations": [v.to_dict() for v in self.violations],
            "filtered_content": self.filtered_content,
            "metadata": self.metadata,
        }


class Guardrail:
    """Base class for all guardrail filters and validators.

    Guardrails check content for safety, compliance, or policy violations.
    They can be applied to both user inputs and AI model outputs.

    Example:
        >>> from ai_lib_python.guardrails import KeywordFilter
        >>>
        >>> filter = KeywordFilter(
        ...     rule_id="no-api-keys",
        ...     keywords=["sk-", "Bearer"],
        ...     severity=GuardrailSeverity.CRITICAL,
        ... )
        >>>
        >>> result = filter.check("Here is my key: sk-12345")
        >>> if not result.is_safe:
        ...     print(f"Blocked: {result.violations[0].message}")
    """

    def __init__(
        self,
        rule_id: str,
        severity: GuardrailSeverity = GuardrailSeverity.WARNING,
        enabled: bool = True,
    ) -> None:
        """Initialize the guardrail.

        Args:
            rule_id: Unique identifier for this guardrail rule
            severity: Severity level for violations
            enabled: Whether this guardrail is active
        """
        if not rule_id:
            raise ValueError("rule_id must be non-empty")

        if not isinstance(severity, GuardrailSeverity):
            severity = GuardrailSeverity(severity)

        self._rule_id = rule_id
        self._severity = severity
        self._enabled = enabled

    @property
    def rule_id(self) -> str:
        """Get the rule ID."""
        return self._rule_id

    @property
    def severity(self) -> GuardrailSeverity:
        """Get the violation severity."""
        return self._severity

    @property
    def enabled(self) -> bool:
        """Check if this guardrail is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable this guardrail."""
        self._enabled = True

    def disable(self) -> None:
        """Disable this guardrail."""
        self._enabled = False

    def check(self, content: str) -> GuardrailResult:
        """Check if content violates this guardrail.

        Args:
            content: Text content to check

        Returns:
            GuardrailResult with violation details if unsafe
        """
        if not self._enabled:
            return GuardrailResult.safe(content=content)

        if not content:
            return GuardrailResult.safe(content=content)

        return self._check_impl(content)

    def filter(self, content: str) -> str:
        """Filter content by removing or replacing violations.

        Args:
            content: Text content to filter

        Returns:
            Filtered content with violations removed/replaced
        """
        result = self.check(content)
        return result.filtered_content if result.filtered_content is not None else content

    def _check_impl(self, content: str) -> GuardrailResult:
        """Implementation of the guardrail check logic.

        Subclasses must override this method.

        Args:
            content: Text content to check

        Returns:
            GuardrailResult with violations if content is unsafe
        """
        raise NotImplementedError("Subclasses must implement _check_impl")

    def _filter_impl(self, content: str) -> str:
        """Implementation of the content filter logic.

        Subclasses can override this method to provide filtering capabilities.

        Args:
            content: Text content to filter

        Returns:
            Filtered content
        """
        # Default: return original content (no filtering)
        return content


class CompositeGuardrail(Guardrail):
    """Combines multiple guardrails into a single check.

    All guardrails are evaluated and all violations are collected.

    Example:
        >>> from ai_lib_python.guardrails import KeywordFilter, LengthFilter
        >>>
        >>> composite = CompositeGuardrail(
        ...     rule_id="composite-check",
        ...     guardrails=[
        ...         KeywordFilter("no-api-keys", ["sk-"], GuardrailSeverity.CRITICAL),
        ...         LengthFilter("max-length", max_length=1000),
        ...     ],
        ... )
    """

    def __init__(
        self,
        rule_id: str,
        guardrails: list[Guardrail],
        severity: GuardrailSeverity = GuardrailSeverity.WARNING,
        stop_on_first: bool = False,
    ) -> None:
        """Initialize composite guardrail.

        Args:
            rule_id: Unique identifier
            guardrails: List of guardrails to compose
            severity: Default severity for composite violations
            stop_on_first: Stop after first violation or continue checking all
        """
        super().__init__(rule_id, severity)
        self._guardrails = guardrails
        self._stop_on_first = stop_on_first

    @property
    def guardrails(self) -> list[Guardrail]:
        """Get the list of guardrails."""
        return self._guardrails

    def add_guardrail(self, guardrail: Guardrail) -> None:
        """Add a guardrail to the composite."""
        self._guardrails.append(guardrail)

    def remove_guardrail(self, rule_id: str) -> bool:
        """Remove a guardrail by rule ID."""
        for i, g in enumerate(self._guardrails):
            if g._rule_id == rule_id:
                self._guardrails.pop(i)
                return True
        return False

    def _check_impl(self, content: str) -> GuardrailResult:
        """Check all guardrails and collect violations."""
        all_violations: list[GuardrailViolation] = []

        for guardrail in self._guardrails:
            result = guardrail.check(content)

            if not result.is_safe:
                all_violations.extend(result.violations)

                if self._stop_on_first:
                    return GuardrailResult.violated(all_violations, content)

        if all_violations:
            return GuardrailResult.violated(all_violations, content)

        return GuardrailResult.safe(content=content)

    def _filter_impl(self, content: str) -> str:
        """Apply all filters in sequence."""
        filtered = content
        for guardrail in self._guardrails:
            filtered = guardrail.filter(filtered)
        return filtered


class ConditionalGuardrail(Guardrail):
    """Applies a guardrail only when a condition is met.

    Useful for context-aware filtering.

    Example:
        >>> filter = ConditionalGuardrail(
        ...     rule_id="filter-in-chat",
        ...     guardrail=KeywordFilter("no-api-keys", ["sk-"]),
        ...     condition=lambda ctx: ctx.get("mode") == "chat",
        ... )
    """

    def __init__(
        self,
        rule_id: str,
        guardrail: Guardrail,
        condition: Callable[[dict[str, Any]], bool],
        severity: GuardrailSeverity = GuardrailSeverity.WARNING,
    ) -> None:
        """Initialize conditional guardrail.

        Args:
            rule_id: Unique identifier
            guardrail: Guardrail to apply conditionally
            condition: Function that takes context dict and returns bool
            severity: Severity for violations
        """
        super().__init__(rule_id, severity)
        self._guardrail = guardrail
        self._condition = condition
        self._context: dict[str, Any] = {}

    def set_context(self, context: dict[str, Any]) -> None:
        """Set the context for condition evaluation."""
        self._context = context

    def _check_impl(self, content: str) -> GuardrailResult:
        """Check only if condition is met."""
        if self._condition(self._context):
            return self._guardrail.check(content)
        return GuardrailResult.safe(content=content)

    def _filter_impl(self, content: str) -> str:
        """Filter only if condition is met."""
        if self._condition(self._context):
            return self._guardrail.filter(content)
        return content
