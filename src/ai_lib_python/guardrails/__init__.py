"""
Guardrails module for content filtering and safety checks.

This module provides a flexible framework for filtering and validating
both user inputs and AI model outputs to ensure safety and compliance.

Core principle: All logic is operators, all configuration is protocol.
"""

from ai_lib_python.guardrails.base import Guardrail, GuardrailResult, GuardrailViolation
from ai_lib_python.guardrails.filters import (
    EmailFilter,
    KeywordFilter,
    LengthFilter,
    ProfanityFilter,
    RegexFilter,
    UrlFilter,
)
from ai_lib_python.guardrails.validators import ContentValidator

__all__ = [
    # Base classes
    "Guardrail",
    "GuardrailViolation",
    "GuardrailResult",
    # Filters
    "KeywordFilter",
    "RegexFilter",
    "LengthFilter",
    "ProfanityFilter",
    "UrlFilter",
    "EmailFilter",
    # Validators
    "ContentValidator",
]
