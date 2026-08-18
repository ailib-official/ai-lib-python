"""
Utility functions and helpers.

This module contains:
- JSONPath utilities
- Tool decorator
- ToolCallAssembler for streaming tool calls
- Other helper functions
"""

from ai_lib_python.utils.thinking_extract import (
    OPENAI_COMPAT_THINKING_KEYS,
    first_nonempty_string_field,
    thinking_from_openai_compat_delta,
    thinking_from_openai_compat_message,
)
from ai_lib_python.utils.tool_call_assembler import (
    MultiToolCallAssembler,
    ToolCallAssembler,
    ToolCallFragment,
)

__all__ = [
    "MultiToolCallAssembler",
    "OPENAI_COMPAT_THINKING_KEYS",
    "ToolCallAssembler",
    "ToolCallFragment",
    "first_nonempty_string_field",
    "thinking_from_openai_compat_delta",
    "thinking_from_openai_compat_message",
]
