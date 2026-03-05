"""
Concrete filter implementations for common guardrail use cases.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ai_lib_python.guardrails.base import (
    Guardrail,
    GuardrailResult,
    GuardrailSeverity,
    GuardrailViolation,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class KeywordFilter(Guardrail):
    """Filters content based on keyword matches.

    Useful for blocking specific words, phrases, or patterns.

    Example:
        >>> filter = KeywordFilter(
        ...     rule_id="no-api-keys",
        ...     keywords=["sk-", "Bearer", "password"],
        ...     severity=GuardrailSeverity.CRITICAL,
        ...     case_sensitive=False,
        ... )
        >>>
        >>> result = filter.check("Here's my API key: sk-12345")
        >>> assert not result.is_safe
        >>> print(result.violations[0].matched_text)  # "sk-"
    """

    def __init__(
        self,
        rule_id: str,
        keywords: list[str],
        severity: GuardrailSeverity = GuardrailSeverity.WARNING,
        case_sensitive: bool = False,
        match_substring: bool = True,
        replacement: str | None = None,
    ) -> None:
        """Initialize keyword filter.

        Args:
            rule_id: Unique identifier
            keywords: List of keywords to filter
            severity: Severity for violations
            case_sensitive: Whether matching is case sensitive
            match_substring: Match keywords anywhere or whole words
            replacement: String to replace matches with (for filtering)
        """
        super().__init__(rule_id, severity)
        self._keywords = keywords if case_sensitive else [k.lower() for k in keywords]
        self._case_sensitive = case_sensitive
        self._match_substring = match_substring
        self._replacement = replacement or "[REDACTED]"

    @property
    def keywords(self) -> list[str]:
        """Get the list of keywords."""
        return self._keywords

    def _check_impl(self, content: str) -> GuardrailResult:
        """Check for keyword matches."""
        violations = []
        text_to_check = content if self._case_sensitive else content.lower()

        for keyword in self._keywords:
            if self._match_substring:
                if keyword in text_to_check:
                    violations.append(
                        self._create_violation(
                            f"Found forbidden keyword: {keyword}",
                            keyword,
                        )
                    )
            else:
                # Match whole words only
                pattern = r"\b" + re.escape(keyword) + r"\b"
                if re.search(pattern, text_to_check, re.IGNORECASE if not self._case_sensitive else 0):
                    violations.append(
                        self._create_violation(
                            f"Found forbidden keyword: {keyword}",
                            keyword,
                        )
                    )

        if violations:
            return GuardrailResult.violated(violations, content)

        return GuardrailResult.safe(content=content)

    def _filter_impl(self, content: str) -> str:
        """Replace matched keywords."""
        filtered = content
        for keyword in self._keywords:
            if self._match_substring:
                filtered = filtered.replace(
                    keyword,
                    self._replacement,
                )
                if not self._case_sensitive:
                    # Also replace case-insensitive matches
                    case_pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                    filtered = case_pattern.sub(self._replacement, filtered)
            else:
                pattern = r"\b" + re.escape(keyword) + r"\b"
                flags = re.IGNORECASE if not self._case_sensitive else 0
                filtered = re.sub(pattern, self._replacement, filtered, flags=flags)

        return filtered

    def _create_violation(
        self,
        message: str,
        matched_text: str,
    ) -> GuardrailViolation:
        """Create a violation."""
        return GuardrailViolation(
            rule_id=self._rule_id,
            message=message,
            severity=self._severity,
            matched_text=matched_text,
        )


class RegexFilter(Guardrail):
    """Filters content using regular expressions.

    More flexible than keyword filtering for complex patterns.

    Example:
        >>> # Match API key patterns
        >>> filter = RegexFilter(
        ...     rule_id="api-keys",
        ...     pattern=r"sk-[a-zA-Z0-9]{32}",
        ...     severity=GuardrailSeverity.CRITICAL,
        ... )
    """

    def __init__(
        self,
        rule_id: str,
        pattern: str,
        severity: GuardrailSeverity = GuardrailSeverity.WARNING,
        flags: int = 0,
        replacement: str | None = None,
        message: str | None = None,
    ) -> None:
        """Initialize regex filter.

        Args:
            rule_id: Unique identifier
            pattern: Regular expression pattern
            severity: Severity for violations
            flags: Regex flags (re.IGNORECASE, etc.)
            replacement: String to replace matches with
            message: Custom violation message (pattern used if None)
        """
        super().__init__(rule_id, severity)
        self._pattern = re.compile(pattern, flags)
        self._replacement = replacement or "[REDACTED]"
        self._message = message

    @property
    def pattern(self) -> re.Pattern[str]:
        """Get the compiled pattern."""
        return self._pattern

    def _check_impl(self, content: str) -> GuardrailResult:
        """Check content against regex pattern."""
        match = self._pattern.search(content)

        if match:
            from ai_lib_python.guardrails.base import GuardrailViolation

            message = self._message or f"Content matches forbidden pattern: {self._pattern.pattern}"

            return GuardrailResult.violated(
                [
                    GuardrailViolation(
                        rule_id=self._rule_id,
                        message=message,
                        severity=self._severity,
                        matched_text=match.group(0),
                    )
                ],
                content,
            )

        return GuardrailResult.safe(content=content)

    def _filter_impl(self, content: str) -> str:
        """Replace all pattern matches."""
        return self._pattern.sub(self._replacement, content)


class LengthFilter(Guardrail):
    """Filters content based on length constraints.

    Useful for preventing overly long or short inputs/outputs.

    Example:
        >>> filter = LengthFilter(
        ...     rule_id="length-limits",
        ...     min_length=10,
        ...     max_length=1000,
        ...     severity=GuardrailSeverity.ERROR,
        ... )
    """

    def __init__(
        self,
        rule_id: str,
        min_length: int | None = None,
        max_length: int | None = None,
        severity: GuardrailSeverity = GuardrailSeverity.WARNING,
        count_mode: str = "chars",
    ) -> None:
        """Initialize length filter.

        Args:
            rule_id: Unique identifier
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            severity: Severity for violations
            count_mode: How to count (chars, words, tokens_chars, tokens_words)
        """
        super().__init__(rule_id, severity)

        if min_length is not None and min_length < 0:
            raise ValueError("min_length must be non-negative")

        if max_length is not None and max_length < 0:
            raise ValueError("max_length must be non-negative")

        if (min_length is not None and max_length is not None) and min_length > max_length:
            raise ValueError("min_length cannot be greater than max_length")

        self._min_length = min_length
        self._max_length = max_length
        self._count_mode = count_mode
        self._counter: Callable[[str], int]

        if count_mode == "chars":
            self._counter = len
        elif count_mode == "words":
            self._counter = lambda x: len(x.split())
        elif count_mode == "tokens_chars":
            # Rough approximation of token count (4 chars per token)
            self._counter = lambda x: len(x) // 4
        elif count_mode == "tokens_words":
            # Average 0.75 words per token
            self._counter = lambda x: int(len(x.split()) * 0.75)
        else:
            raise ValueError(f"Invalid count_mode: {count_mode}")

    def _check_impl(self, content: str) -> GuardrailResult:
        """Check content length."""
        length = self._counter(content)
        violations = []

        if self._min_length is not None and length < self._min_length:
            violations.append(
                GuardrailViolation(
                    rule_id=self._rule_id,
                    message=f"Content too short: {length} {self._count_mode}, minimum {self._min_length}",
                    severity=self._severity,
                )
            )

        if self._max_length is not None and length > self._max_length:
            violations.append(
                GuardrailViolation(
                    rule_id=self._rule_id,
                    message=f"Content too long: {length} {self._count_mode}, maximum {self._max_length}",
                    severity=self._severity,
                )
            )

        if violations:
            return GuardrailResult.violated(violations, content)

        return GuardrailResult.safe(content=content)


class ProfanityFilter(Guardrail):
    """Filters profanity and inappropriate language.

    Uses a built-in list of common profanity words.

    Example:
        >>> filter = ProfanityFilter(
        ...     rule_id="no-profanity",
        ...     severity=GuardrailSeverity.WARNING,
        ...     replacement="[CENSORED]",
        ... )
    """

    _DEFAULT_PROFANITY_LIST = [
        # Common profanity words (can be extended)
        "damn", "hell", "crap",
        # Additional words can be added as needed
    ]

    def __init__(
        self,
        rule_id: str,
        severity: GuardrailSeverity = GuardrailSeverity.WARNING,
        profanity_list: list[str] | None = None,
        case_sensitive: bool = False,
        replacement: str | None = None,
    ) -> None:
        """Initialize profanity filter.

        Args:
            rule_id: Unique identifier
            severity: Severity for violations
            profanity_list: Custom list of profanity words
            case_sensitive: Whether matching is case sensitive
            replacement: String to replace profanity with
        """
        super().__init__(rule_id, severity)
        self._keywords = profanity_list or self._DEFAULT_PROFANITY_LIST[:]
        self._case_sensitive = case_sensitive
        self._replacement = replacement or "***"

    def _check_impl(self, content: str) -> GuardrailResult:
        """Check for profanity."""
        violations = []
        text_to_check = content if self._case_sensitive else content.lower()

        for keyword in self._keywords:
            kw = keyword if self._case_sensitive else keyword.lower()
            if kw in text_to_check:
                # Find the actual matched text (preserving case)
                pattern = re.escape(keyword)
                if not self._case_sensitive:
                    pattern_obj = re.compile(pattern, re.IGNORECASE)
                else:
                    pattern_obj = re.compile(pattern)

                match = pattern_obj.search(content)
                if match:
                    violations.append(
                        GuardrailViolation(
                            rule_id=self._rule_id,
                            message=f"Profanity detected: {keyword}",
                            severity=self._severity,
                            matched_text=match.group(0),
                        )
                    )

        if violations:
            return GuardrailResult.violated(violations, content)

        return GuardrailResult.safe(content=content)

    def _filter_impl(self, content: str) -> str:
        """Replace profanity."""
        filtered = content

        for keyword in self._keywords:
            pattern = re.escape(keyword)
            flags = re.IGNORECASE if not self._case_sensitive else 0
            filtered = re.sub(pattern, self._replacement, filtered, flags=flags)

        return filtered


class UrlFilter(Guardrail):
    """Filters or extracts URLs from content.

    Useful for preventing malicious links or extracting URLs.

    Example:
        >>> filter = UrlFilter(
        ...     rule_id="block-urls",
        ...     action="block",
        ...     severity=GuardrailSeverity.INFO,
        ... )
    """

    _URL_PATTERN = re.compile(
        r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w .-]*(?:\?[^\s]*)?",
    )

    def __init__(
        self,
        rule_id: str,
        action: str = "block",
        severity: GuardrailSeverity = GuardrailSeverity.INFO,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> None:
        """Initialize URL filter.

        Args:
            rule_id: Unique identifier
            action: "block" (deny all), "allow" (allow only allowed_domains),
                    "deny" (deny blocked_domains)
            severity: Severity for violations
            allowed_domains: List of allowed domains (for "allow" action)
            blocked_domains: List of blocked domains (for "deny" action)
        """
        super().__init__(rule_id, severity)

        if action not in ("block", "allow", "deny"):
            raise ValueError(f"Invalid action: {action}")

        self._action = action
        self._allowed_domains = [d.lower() for d in (allowed_domains or [])]
        self._blocked_domains = [d.lower() for d in (blocked_domains or [])]

    def _check_impl(self, content: str) -> GuardrailResult:
        """Check content for URLs."""
        matches = list(self._URL_PATTERN.finditer(content))

        if not matches:
            return GuardrailResult.safe(content=content)

        violations = []

        for match in matches:
            url = match.group(0)

            if self._action == "block":
                violations.append(self._create_url_violation(f"URL detected: {url}", url))
            elif self._action == "deny":
                # Check if URL is in blocked domains
                for domain in self._blocked_domains:
                    if domain.lower() in url.lower():
                        violations.append(
                            self._create_url_violation(f"Blocked domain URL: {url}", url)
                        )
                        break
            elif self._action == "allow":
                # Check if URL is in allowed domains
                allowed = any(d.lower() in url.lower() for d in self._allowed_domains)
                if not allowed:
                    violations.append(
                        self._create_url_violation(f"URL from non-allowed domain: {url}", url)
                    )

        if violations:
            return GuardrailResult.violated(violations, content)

        return GuardrailResult.safe(content=content)

    def _create_url_violation(self, message: str, url: str) -> GuardrailViolation:
        """Create a URL violation."""
        return GuardrailViolation(
            rule_id=self._rule_id,
            message=message,
            severity=self._severity,
            matched_text=url,
        )


class EmailFilter(Guardrail):
    """Filters or detects email addresses in content.

    Useful for preventing PII leakage or logging email detection.

    Example:
        >>> filter = EmailFilter(
        ...     rule_id="no-emails",
        ...     action="block",
        ...     severity=GuardrailSeverity.WARNING,
        ... )
    """

    _EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    )

    def __init__(
        self,
        rule_id: str,
        action: str = "block",
        severity: GuardrailSeverity = GuardrailSeverity.INFO,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        replacement: str | None = None,
    ) -> None:
        """Initialize email filter.

        Args:
            rule_id: Unique identifier
            action: "block" (deny all), "allow" (allow only allowed_domains),
                    "deny" (deny blocked_domains)
            severity: Severity for violations
            allowed_domains: List of allowed email domains
            blocked_domains: List of blocked email domains
            replacement: String to replace emails with
        """
        super().__init__(rule_id, severity)

        if action not in ("block", "allow", "deny"):
            raise ValueError(f"Invalid action: {action}")

        self._action = action
        self._allowed_domains = [d.lower() for d in (allowed_domains or [])]
        self._blocked_domains = [d.lower() for d in (blocked_domains or [])]
        self._replacement = replacement or "[REDACTED]"

    def _check_impl(self, content: str) -> GuardrailResult:
        """Check content for email addresses."""
        matches = list(self._EMAIL_PATTERN.finditer(content))

        if not matches:
            return GuardrailResult.safe(content=content)

        violations = []

        for match in matches:
            email = match.group(0)

            if self._action == "block":
                violations.append(self._create_email_violation(f"Email detected: {email}", email))
            elif self._action == "deny":
                # Check if email is in blocked domains
                for domain in self._blocked_domains:
                    if email.lower().endswith(domain.lower()):
                        violations.append(
                            self._create_email_violation(f"Blocked domain email: {email}", email)
                        )
                        break
            elif self._action == "allow":
                # Check if email is in allowed domains
                allowed = any(email.lower().endswith(d.lower()) for d in self._allowed_domains)
                if not allowed:
                    violations.append(
                        self._create_email_violation(
                            f"Email from non-allowed domain: {email}", email
                        )
                    )

        if violations:
            return GuardrailResult.violated(violations, content)

        return GuardrailResult.safe(content=content)

    def _filter_impl(self, content: str) -> str:
        """Replace email addresses."""
        return self._EMAIL_PATTERN.sub(self._replacement, content)

    def _create_email_violation(self, message: str, email: str) -> GuardrailViolation:
        """Create an email violation."""
        return GuardrailViolation(
            rule_id=self._rule_id,
            message=message,
            severity=self._severity,
            matched_text=email,
        )
