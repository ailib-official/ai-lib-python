"""
Output validation for structured responses.

Validates JSON output against schemas and Pydantic models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T")
ModelT = TypeVar("ModelT", bound=BaseModel)


class ValidationError(Exception):
    """Error raised when validation fails.

    Attributes:
        message: Error message
        path: JSON path to the error location
        value: The invalid value
    """

    def __init__(
        self,
        message: str,
        path: str | None = None,
        value: Any = None,
    ) -> None:
        """Initialize validation error.

        Args:
            message: Error message
            path: JSON path to error
            value: Invalid value
        """
        self.message = message
        self.path = path
        self.value = value
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message."""
        msg = self.message
        if self.path:
            msg = f"{self.path}: {msg}"
        return msg


@dataclass
class ValidationResult:
    """Result of validation.

    Attributes:
        valid: Whether validation passed
        errors: List of validation errors
        data: Validated/parsed data
    """

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    data: Any = None

    def __bool__(self) -> bool:
        """Return True if validation passed."""
        return self.valid

    def raise_if_invalid(self) -> None:
        """Raise ValidationError if validation failed."""
        if not self.valid:
            raise ValidationError("; ".join(self.errors))


class OutputValidator:
    """Validator for structured output.

    Validates JSON strings and dictionaries against schemas
    or Pydantic models.

    Example:
        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>>
        >>> validator = OutputValidator(User)
        >>> result = validator.validate('{"name": "Alice", "age": 30}')
        >>> print(result.valid)  # True
        >>> print(result.data)   # User(name='Alice', age=30)
    """

    def __init__(
        self,
        schema: dict[str, Any] | type[BaseModel] | None = None,
        strict: bool = True,
    ) -> None:
        """Initialize validator.

        Args:
            schema: JSON schema dict or Pydantic model class
            strict: Whether to use strict validation
        """
        self._pydantic_model: type[BaseModel] | None = None
        self._json_schema: dict[str, Any] | None = None
        self._strict = strict

        if schema is not None:
            if isinstance(schema, dict):
                self._json_schema = schema
            elif hasattr(schema, "model_validate"):
                # Pydantic model
                self._pydantic_model = schema
                self._json_schema = schema.model_json_schema()
            else:
                raise ValueError(
                    "Schema must be a JSON schema dict or Pydantic model class"
                )

    def validate(self, data: str | dict[str, Any]) -> ValidationResult:
        """Validate data against the schema.

        Args:
            data: JSON string or dictionary to validate

        Returns:
            ValidationResult with validation status and parsed data
        """
        # Parse JSON if string
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError as e:
                return ValidationResult(
                    valid=False,
                    errors=[f"Invalid JSON: {e}"],
                )
        else:
            parsed = data

        # Validate against Pydantic model if available
        if self._pydantic_model is not None:
            return self._validate_pydantic(parsed)

        # Validate against JSON schema if available
        if self._json_schema is not None:
            return self._validate_json_schema(parsed)

        # No schema, just return parsed data
        return ValidationResult(valid=True, data=parsed)

    def validate_or_raise(self, data: str | dict[str, Any]) -> Any:
        """Validate data and raise if invalid.

        Args:
            data: Data to validate

        Returns:
            Validated/parsed data

        Raises:
            ValidationError: If validation fails
        """
        result = self.validate(data)
        result.raise_if_invalid()
        return result.data

    def parse(self, data: str | dict[str, Any], model: type[ModelT]) -> ModelT:
        """Parse and validate data into a Pydantic model.

        Args:
            data: Data to parse
            model: Pydantic model class

        Returns:
            Model instance

        Raises:
            ValidationError: If validation fails
        """
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError as e:
                raise ValidationError(f"Invalid JSON: {e}") from e
        else:
            parsed = data

        try:
            return model.model_validate(parsed)
        except Exception as e:
            raise ValidationError(str(e)) from e

    def _validate_pydantic(self, data: dict[str, Any]) -> ValidationResult:
        """Validate against Pydantic model.

        Args:
            data: Data to validate

        Returns:
            ValidationResult
        """
        model = self._pydantic_model
        if model is None:
            return ValidationResult(valid=False, errors=["No Pydantic model configured"])
        try:
            validated = model.model_validate(data)
            return ValidationResult(valid=True, data=validated)
        except Exception as e:
            return ValidationResult(
                valid=False,
                errors=[str(e)],
            )

    def _validate_json_schema(self, data: dict[str, Any]) -> ValidationResult:
        """Validate against JSON schema.

        Args:
            data: Data to validate

        Returns:
            ValidationResult
        """
        errors: list[str] = []

        schema = self._json_schema
        if schema is None:
            return ValidationResult(valid=False, errors=["No JSON schema configured"])

        # Basic type checking
        schema_type = schema.get("type")
        if schema_type == "object" and not isinstance(data, dict):
            return ValidationResult(
                valid=False,
                errors=[f"Expected object, got {type(data).__name__}"],
            )

        # Validate required properties
        required = schema.get("required", [])
        for prop in required:
            if prop not in data:
                errors.append(f"Missing required property: {prop}")

        # Validate property types
        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if prop_name in data:
                prop_errors = self._validate_property(
                    data[prop_name], prop_schema, prop_name
                )
                errors.extend(prop_errors)

        # Check additional properties
        additional_props = schema.get("additionalProperties", True)
        if additional_props is False:
            extra_props = set(data.keys()) - set(properties.keys())
            for prop in extra_props:
                errors.append(f"Additional property not allowed: {prop}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            data=data if len(errors) == 0 else None,
        )

    def _validate_property(
        self,
        value: Any,
        schema: dict[str, Any],
        path: str,
    ) -> list[str]:
        """Validate a single property.

        Args:
            value: Property value
            schema: Property schema
            path: Property path

        Returns:
            List of error messages
        """
        errors: list[str] = []
        prop_type = schema.get("type")

        # Type validation
        type_checks = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
            "null": lambda v: v is None,
        }

        if prop_type and prop_type in type_checks:
            # Handle nullable
            if schema.get("nullable") and value is None:
                return []

            if not type_checks[prop_type](value):
                errors.append(f"{path}: Expected {prop_type}, got {type(value).__name__}")
                return errors

        # String constraints
        if prop_type == "string" and isinstance(value, str):
            if "minLength" in schema and len(value) < schema["minLength"]:
                errors.append(f"{path}: String too short (min {schema['minLength']})")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                errors.append(f"{path}: String too long (max {schema['maxLength']})")
            if "pattern" in schema:
                import re

                if not re.match(schema["pattern"], value):
                    errors.append(f"{path}: String does not match pattern")

        # Number constraints
        if prop_type in ("integer", "number") and isinstance(value, (int, float)):
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{path}: Value below minimum ({schema['minimum']})")
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"{path}: Value above maximum ({schema['maximum']})")

        # Enum constraint
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{path}: Value not in allowed enum values")

        # Array validation
        if prop_type == "array" and isinstance(value, list):
            if "minItems" in schema and len(value) < schema["minItems"]:
                errors.append(f"{path}: Array too short (min {schema['minItems']})")
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                errors.append(f"{path}: Array too long (max {schema['maxItems']})")
            if "items" in schema:
                for i, item in enumerate(value):
                    item_errors = self._validate_property(
                        item, schema["items"], f"{path}[{i}]"
                    )
                    errors.extend(item_errors)

        return errors
