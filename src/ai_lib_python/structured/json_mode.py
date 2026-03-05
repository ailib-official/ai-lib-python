"""
JSON mode support for structured output.

Provides configuration and utilities for JSON mode responses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel

from ai_lib_python.structured.schema import json_schema_from_pydantic
from ai_lib_python.structured.validator import OutputValidator, ValidationResult

T = TypeVar("T", bound=BaseModel)


class JsonMode(str, Enum):
    """JSON mode options."""

    # Standard JSON mode (guaranteed valid JSON)
    JSON = "json"

    # JSON with schema validation (OpenAI)
    JSON_SCHEMA = "json_schema"

    # No JSON mode (default)
    OFF = "off"


@dataclass
class JsonModeConfig:
    """Configuration for JSON mode.

    Attributes:
        mode: JSON mode to use
        schema: JSON schema for validation
        schema_name: Name for the schema (OpenAI)
        strict: Whether to enforce strict schema compliance
    """

    mode: JsonMode = JsonMode.JSON
    schema: dict[str, Any] | None = None
    schema_name: str = "response"
    strict: bool = True

    @classmethod
    def json_object(cls) -> JsonModeConfig:
        """Create config for simple JSON object mode.

        Returns:
            JsonModeConfig instance
        """
        return cls(mode=JsonMode.JSON)

    @classmethod
    def from_schema(
        cls,
        schema: dict[str, Any],
        name: str = "response",
        strict: bool = True,
    ) -> JsonModeConfig:
        """Create config from JSON schema.

        Args:
            schema: JSON schema dictionary
            name: Schema name
            strict: Whether to enforce strict compliance

        Returns:
            JsonModeConfig instance
        """
        return cls(
            mode=JsonMode.JSON_SCHEMA,
            schema=schema,
            schema_name=name,
            strict=strict,
        )

    @classmethod
    def from_pydantic(
        cls,
        model: type,
        name: str | None = None,
        strict: bool = True,
    ) -> JsonModeConfig:
        """Create config from Pydantic model.

        Args:
            model: Pydantic model class
            name: Schema name (defaults to model name)
            strict: Whether to enforce strict compliance

        Returns:
            JsonModeConfig instance
        """
        schema = json_schema_from_pydantic(model)
        return cls(
            mode=JsonMode.JSON_SCHEMA,
            schema=schema,
            schema_name=name or model.__name__,
            strict=strict,
        )

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI API format.

        Returns:
            Dictionary for OpenAI request
        """
        if self.mode == JsonMode.OFF:
            return {}

        if self.mode == JsonMode.JSON:
            return {"response_format": {"type": "json_object"}}

        if self.mode == JsonMode.JSON_SCHEMA and self.schema:
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.schema_name,
                        "strict": self.strict,
                        "schema": self.schema,
                    },
                }
            }

        return {}

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic API format.

        Returns:
            Dictionary for Anthropic request

        Note:
            Anthropic doesn't have native JSON mode, but you can
            instruct in the system prompt.
        """
        # Anthropic relies on system prompt instructions
        return {}


@dataclass
class StructuredOutput:
    """Structured output result with validation.

    Attributes:
        raw: Raw response content
        parsed: Parsed JSON data
        validated: Validated model instance (if Pydantic)
        validation_result: Validation result
    """

    raw: str
    parsed: dict[str, Any] | None = None
    validated: Any = None
    validation_result: ValidationResult = field(default_factory=ValidationResult)

    @property
    def is_valid(self) -> bool:
        """Check if output is valid."""
        return self.validation_result.valid

    @property
    def data(self) -> Any:
        """Get the best available data representation."""
        if self.validated is not None:
            return self.validated
        if self.parsed is not None:
            return self.parsed
        return self.raw

    def as_model(self, model: type[T]) -> T:
        """Get output as Pydantic model.

        Args:
            model: Pydantic model class

        Returns:
            Model instance

        Raises:
            ValueError: If data is not valid
        """
        if not self.is_valid:
            raise ValueError("Output is not valid")
        if isinstance(self.validated, model):
            return self.validated
        if self.parsed:
            return model.model_validate(self.parsed)
        raise ValueError("No parsed data available")

    @classmethod
    def from_response(
        cls,
        content: str,
        validator: OutputValidator | None = None,
    ) -> StructuredOutput:
        """Create structured output from response.

        Args:
            content: Response content
            validator: Optional validator

        Returns:
            StructuredOutput instance
        """
        # Try to parse JSON
        parsed = None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            pass

        # Validate if validator provided
        validation_result = ValidationResult(valid=True, data=parsed)
        validated = None

        if validator and parsed is not None:
            validation_result = validator.validate(parsed)
            if validation_result.valid:
                validated = validation_result.data

        return cls(
            raw=content,
            parsed=parsed,
            validated=validated,
            validation_result=validation_result,
        )


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract JSON from text that may contain markdown code blocks.

    Args:
        text: Text potentially containing JSON

    Returns:
        Parsed JSON or None

    Example:
        >>> text = '''Here is the data:
        ... ```json
        ... {"name": "Alice"}
        ... ```
        ... '''
        >>> extract_json(text)
        {'name': 'Alice'}
    """
    import re

    # Try direct parsing first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code blocks
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",  # ```json ... ```
        r"```\s*([\s\S]*?)\s*```",  # ``` ... ```
        r"\{[\s\S]*\}",  # Raw JSON object
        r"\[[\s\S]*\]",  # Raw JSON array
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                candidate = match.group(1) if match.lastindex else match.group(0)
                parsed = json.loads(candidate.strip())
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, IndexError):
                continue

    return None
