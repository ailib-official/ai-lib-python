"""
Tool types for function calling support.

Based on AI-Protocol standard_schema for tool definitions and calls.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ai_lib_python.types.message import Message


class ToolChoice(str, Enum):
    """Tool choice policy for requests."""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


class FunctionDefinition(BaseModel):
    """Function definition within a tool.

    Defines the schema for a callable function including:
    - name: Function identifier
    - description: What the function does
    - parameters: JSON Schema for function parameters
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(description="Function name")
    description: str | None = Field(default=None, description="Function description")
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema for parameters",
    )
    strict: bool | None = Field(
        default=None,
        description="Whether to enforce strict schema validation (provider-specific)",
    )


class ToolDefinition(BaseModel):
    """Tool definition for function calling.

    Wraps a FunctionDefinition with the standard tool format.

    Example:
        >>> tool = ToolDefinition.from_function(
        ...     name="get_weather",
        ...     description="Get weather for a city",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {
        ...             "city": {"type": "string", "description": "City name"},
        ...             "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        ...         },
        ...         "required": ["city"]
        ...     }
        ... )
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(default="function", description="Tool type")
    function: FunctionDefinition = Field(description="Function definition")

    @classmethod
    def from_function(
        cls,
        name: str,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> ToolDefinition:
        """Create a tool definition from function details.

        Args:
            name: Function name
            description: Function description
            parameters: JSON Schema for parameters
            strict: Whether to enforce strict validation

        Returns:
            ToolDefinition instance
        """
        func_def = FunctionDefinition(
            name=name,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            strict=strict,
        )
        return cls(function=func_def)

    @property
    def name(self) -> str:
        """Get the function name."""
        return self.function.name

    @property
    def description(self) -> str | None:
        """Get the function description."""
        return self.function.description


class ToolCall(BaseModel):
    """A tool call from the model response.

    Represents a request from the model to invoke a specific tool.

    Attributes:
        id: Unique identifier for this tool call
        type: Tool type (typically "function")
        function_name: Name of the function to call
        arguments: Parsed arguments for the function

    Example:
        >>> tool_call = ToolCall(
        ...     id="call_abc123",
        ...     function_name="get_weather",
        ...     arguments={"city": "Beijing", "unit": "celsius"}
        ... )
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(description="Unique tool call identifier")
    type: str = Field(default="function", description="Tool type")
    function_name: str = Field(description="Name of the function to call")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed function arguments",
    )

    # Raw arguments string (useful for debugging/logging)
    arguments_raw: str | None = Field(
        default=None,
        description="Raw arguments string before parsing",
    )

    @classmethod
    def from_openai_format(
        cls,
        id: str,
        function_name: str,
        arguments: str | dict[str, Any],
    ) -> ToolCall:
        """Create from OpenAI-style tool call format.

        Args:
            id: Tool call ID
            function_name: Function name
            arguments: Arguments (string or dict)

        Returns:
            ToolCall instance
        """
        import json

        if isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                parsed_args = {}
            return cls(
                id=id,
                function_name=function_name,
                arguments=parsed_args,
                arguments_raw=arguments,
            )
        return cls(
            id=id,
            function_name=function_name,
            arguments=arguments,
        )

    def to_content_block(self) -> dict[str, Any]:
        """Convert to content block format for message construction.

        Returns:
            Dictionary suitable for use as a tool_use content block.
        """
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.function_name,
            "input": self.arguments,
        }
    @staticmethod
    def to_message(tool_call: ToolCall, result: Any) -> Message:
        """Create a TOOL-role message from a tool call and its result.

        Args:
            tool_call: The ToolCall instance
            result: The execution result (will be stringified if not a string)

        Returns:
            Message with TOOL role
        """
        from ai_lib_python.types.message import Message

        content = result if isinstance(result, str) else str(result)
        return Message.tool(tool_call_id=tool_call.id, content=content)
