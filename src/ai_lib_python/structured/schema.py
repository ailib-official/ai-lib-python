"""
JSON Schema generation utilities.

Provides schema generation from Python types and Pydantic models.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, get_args, get_origin

if TYPE_CHECKING:
    from pydantic import BaseModel


def json_schema_from_type(python_type: type) -> dict[str, Any]:
    """Generate JSON schema from a Python type.

    Args:
        python_type: Python type to convert

    Returns:
        JSON schema dictionary

    Example:
        >>> schema = json_schema_from_type(str)
        >>> print(schema)
        {"type": "string"}

        >>> from typing import List
        >>> schema = json_schema_from_type(List[int])
        >>> print(schema)
        {"type": "array", "items": {"type": "integer"}}
    """
    # Handle None
    if python_type is type(None):
        return {"type": "null"}

    # Handle basic types
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        bytes: {"type": "string", "format": "byte"},
    }

    if python_type in type_mapping:
        return type_mapping[python_type]

    # Handle generic types
    origin = get_origin(python_type)
    args = get_args(python_type)

    # Handle list/List
    if origin is list:
        if args:
            return {
                "type": "array",
                "items": json_schema_from_type(args[0]),
            }
        return {"type": "array"}

    # Handle dict/Dict
    if origin is dict:
        schema: dict[str, Any] = {"type": "object"}
        if len(args) >= 2:
            schema["additionalProperties"] = json_schema_from_type(args[1])
        return schema

    # Handle tuple/Tuple
    if origin is tuple:
        if args:
            return {
                "type": "array",
                "items": [json_schema_from_type(arg) for arg in args],
                "minItems": len(args),
                "maxItems": len(args),
            }
        return {"type": "array"}

    # Handle Union types (including | syntax)
    if origin is type(int | str):  # Python 3.10+ union
        schemas = [json_schema_from_type(arg) for arg in args]
        # Check if it's Optional (Union with None)
        none_schemas = [s for s in schemas if s.get("type") == "null"]
        other_schemas = [s for s in schemas if s.get("type") != "null"]

        if none_schemas and len(other_schemas) == 1:
            # Optional type
            return {**other_schemas[0], "nullable": True}
        return {"anyOf": schemas}

    # Handle Any
    if python_type is Any:
        return {}

    # Try to handle as Pydantic model
    if hasattr(python_type, "model_json_schema"):
        return json_schema_from_pydantic(python_type)

    # Default to object
    return {"type": "object"}


def json_schema_from_pydantic(model: type[BaseModel]) -> dict[str, Any]:
    """Generate JSON schema from a Pydantic model.

    Args:
        model: Pydantic model class

    Returns:
        JSON schema dictionary

    Raises:
        ValueError: If model is not a Pydantic model

    Example:
        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>> schema = json_schema_from_pydantic(User)
    """
    if not hasattr(model, "model_json_schema"):
        raise ValueError(f"{model} is not a Pydantic model")

    schema = model.model_json_schema()
    if isinstance(schema, dict):
        return schema
    raise ValueError(f"{model} did not return a valid schema object")


class SchemaGenerator:
    """Generator for JSON schemas with customization options.

    Example:
        >>> generator = SchemaGenerator()
        >>> generator.add_property("name", str, description="User's name")
        >>> generator.add_property("age", int, minimum=0)
        >>> schema = generator.build()
    """

    def __init__(
        self,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize schema generator.

        Args:
            title: Schema title
            description: Schema description
        """
        self._title = title
        self._description = description
        self._properties: dict[str, dict[str, Any]] = {}
        self._required: list[str] = []
        self._additional_properties: bool | dict[str, Any] = False

    def add_property(
        self,
        name: str,
        python_type: type,
        *,
        description: str | None = None,
        required: bool = True,
        default: Any = None,
        enum: list[Any] | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
    ) -> SchemaGenerator:
        """Add a property to the schema.

        Args:
            name: Property name
            python_type: Property type
            description: Property description
            required: Whether property is required
            default: Default value
            enum: Allowed values
            minimum: Minimum value (for numbers)
            maximum: Maximum value (for numbers)
            min_length: Minimum length (for strings)
            max_length: Maximum length (for strings)
            pattern: Regex pattern (for strings)

        Returns:
            Self for chaining
        """
        prop_schema = json_schema_from_type(python_type)

        if description:
            prop_schema["description"] = description
        if default is not None:
            prop_schema["default"] = default
        if enum:
            prop_schema["enum"] = enum
        if minimum is not None:
            prop_schema["minimum"] = minimum
        if maximum is not None:
            prop_schema["maximum"] = maximum
        if min_length is not None:
            prop_schema["minLength"] = min_length
        if max_length is not None:
            prop_schema["maxLength"] = max_length
        if pattern:
            prop_schema["pattern"] = pattern

        self._properties[name] = prop_schema

        if required:
            self._required.append(name)

        return self

    def add_object_property(
        self,
        name: str,
        nested_schema: dict[str, Any],
        *,
        description: str | None = None,
        required: bool = True,
    ) -> SchemaGenerator:
        """Add a nested object property.

        Args:
            name: Property name
            nested_schema: Nested JSON schema
            description: Property description
            required: Whether property is required

        Returns:
            Self for chaining
        """
        prop_schema = nested_schema.copy()
        if description:
            prop_schema["description"] = description

        self._properties[name] = prop_schema

        if required:
            self._required.append(name)

        return self

    def allow_additional_properties(
        self, allowed: bool | type = True
    ) -> SchemaGenerator:
        """Configure additional properties.

        Args:
            allowed: True to allow any, False to disallow, or type to restrict

        Returns:
            Self for chaining
        """
        if isinstance(allowed, bool):
            self._additional_properties = allowed
        else:
            self._additional_properties = json_schema_from_type(allowed)
        return self

    def build(self) -> dict[str, Any]:
        """Build the JSON schema.

        Returns:
            JSON schema dictionary
        """
        schema: dict[str, Any] = {
            "type": "object",
            "properties": self._properties,
        }

        if self._title:
            schema["title"] = self._title
        if self._description:
            schema["description"] = self._description
        if self._required:
            schema["required"] = self._required
        if self._additional_properties is not True:
            schema["additionalProperties"] = self._additional_properties

        return schema

    def to_json(self, indent: int = 2) -> str:
        """Convert schema to JSON string.

        Args:
            indent: JSON indentation

        Returns:
            JSON string
        """
        return json.dumps(self.build(), indent=indent)

    @classmethod
    def from_pydantic(cls, model: type[BaseModel]) -> SchemaGenerator:
        """Create generator from Pydantic model.

        Args:
            model: Pydantic model class

        Returns:
            SchemaGenerator instance
        """
        generator = cls(
            title=getattr(model, "__name__", None),
            description=getattr(model, "__doc__", None),
        )

        # Get schema from Pydantic
        schema = json_schema_from_pydantic(model)

        # Copy properties
        generator._properties = schema.get("properties", {})
        generator._required = schema.get("required", [])

        return generator
