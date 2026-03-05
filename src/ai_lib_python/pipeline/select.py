"""
Frame selector using JSONPath-like expressions.

Filters streaming frames based on conditions defined in protocol manifests.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from ai_lib_python.pipeline.base import Transform

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class JsonPathSelector(Transform):
    """Selector that filters frames using JSONPath-like expressions.

    Supports expressions like:
    - "exists($.choices)" - Check if path exists
    - "$.type == 'content_block_delta'" - Equality check
    - "exists($.choices) || exists($.error)" - Logical OR

    Example:
        >>> selector = JsonPathSelector("exists($.choices) || exists($.error)")
        >>> async for frame in selector.transform(frames):
        ...     process(frame)  # Only frames matching the condition
    """

    def __init__(self, expression: str) -> None:
        """Initialize the selector.

        Args:
            expression: JSONPath-like filter expression
        """
        self._expression = expression
        self._compiled = self._compile_expression(expression)

    def _compile_expression(self, expr: str) -> Any:
        """Compile the expression for efficient evaluation.

        Args:
            expr: Filter expression

        Returns:
            Compiled expression (callable)
        """
        # Store original for reference
        self._original = expr

        # Parse the expression into an evaluator function
        return self._create_evaluator(expr)

    def _create_evaluator(self, expr: str) -> Any:
        """Create an evaluator function for the expression.

        Args:
            expr: Filter expression

        Returns:
            Callable that evaluates the expression against a frame
        """
        expr = expr.strip()

        # Handle logical OR
        if "||" in expr:
            parts = [p.strip() for p in expr.split("||")]
            evaluators = [self._create_evaluator(p) for p in parts]
            return lambda frame: any(e(frame) for e in evaluators)

        # Handle logical AND
        if "&&" in expr:
            parts = [p.strip() for p in expr.split("&&")]
            evaluators = [self._create_evaluator(p) for p in parts]
            return lambda frame: all(e(frame) for e in evaluators)

        # Handle exists() function
        exists_match = re.match(r"exists\((.+)\)", expr)
        if exists_match:
            path = exists_match.group(1).strip()
            return lambda frame: self._path_exists(frame, path)

        # Handle equality: $.path == 'value'
        eq_match = re.match(r"(.+?)\s*==\s*['\"](.+?)['\"]", expr)
        if eq_match:
            path = eq_match.group(1).strip()
            value = eq_match.group(2)
            return lambda frame: self._get_value(frame, path) == value

        # Handle inequality: $.path != 'value'
        neq_match = re.match(r"(.+?)\s*!=\s*['\"](.+?)['\"]", expr)
        if neq_match:
            path = neq_match.group(1).strip()
            value = neq_match.group(2)
            return lambda frame: self._get_value(frame, path) != value

        # Handle null check: $.path != null
        null_neq_match = re.match(r"(.+?)\s*!=\s*null", expr)
        if null_neq_match:
            path = null_neq_match.group(1).strip()
            return lambda frame: self._get_value(frame, path) is not None

        # Default: treat as path existence check
        return lambda frame: self._path_exists(frame, expr)

    def _path_exists(self, frame: dict[str, Any], path: str) -> bool:
        """Check if a path exists in the frame.

        Args:
            frame: JSON frame to check
            path: JSONPath expression

        Returns:
            True if path exists and has a value
        """
        value = self._get_value(frame, path)
        return value is not None

    def _get_value(self, frame: dict[str, Any], path: str) -> Any:
        """Get value at a JSONPath.

        Args:
            frame: JSON frame
            path: JSONPath expression (e.g., "$.choices[0].delta.content")

        Returns:
            Value at path, or None if not found
        """
        # Remove leading $. if present
        if path.startswith("$."):
            path = path[2:]
        elif path.startswith("$"):
            path = path[1:]

        # Handle array wildcard notation: $.choices[*].delta
        if "[*]" in path:
            return self._get_wildcard_value(frame, path)

        current: Any = frame

        # Split path by dots and brackets
        parts = self._split_path(path)

        for part in parts:
            if current is None:
                return None

            # Handle array index
            if part.isdigit():
                idx = int(part)
                if isinstance(current, list) and 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return None
            # Handle dict key
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                return None

        return current

    def _get_wildcard_value(self, frame: dict[str, Any], path: str) -> Any:
        """Handle wildcard array access like $.choices[*].delta.content.

        Args:
            frame: JSON frame
            path: Path with [*] wildcard

        Returns:
            First matching value, or None
        """
        # Split at [*]
        parts = path.split("[*]")
        if len(parts) != 2:
            return None

        before = parts[0].rstrip(".")
        after = parts[1].lstrip(".")

        # Get the array
        array = self._get_value(frame, before) if before else frame
        if not isinstance(array, list):
            return None

        # Try to get value from each element
        for item in array:
            if after:
                value = self._get_value(item, after)
                if value is not None:
                    return value
            else:
                return item

        return None

    def _split_path(self, path: str) -> list[str]:
        """Split a JSONPath into parts.

        Args:
            path: JSONPath string

        Returns:
            List of path parts
        """
        parts: list[str] = []
        current = ""

        i = 0
        while i < len(path):
            char = path[i]

            if char == ".":
                if current:
                    parts.append(current)
                    current = ""
            elif char == "[":
                if current:
                    parts.append(current)
                    current = ""
                # Find closing bracket
                j = path.index("]", i)
                parts.append(path[i + 1 : j])
                i = j
            else:
                current += char

            i += 1

        if current:
            parts.append(current)

        return parts

    def matches(self, frame: dict[str, Any]) -> bool:
        """Check if a frame matches the expression.

        Args:
            frame: JSON frame to check

        Returns:
            True if frame matches
        """
        try:
            return bool(self._compiled(frame))
        except Exception:
            return False

    async def transform(
        self, frames: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[dict[str, Any]]:
        """Filter frames based on the expression.

        Args:
            frames: Async iterator of JSON frames

        Yields:
            Frames that match the expression
        """
        async for frame in frames:
            if self.matches(frame):
                yield frame


class PassThroughSelector(Transform):
    """Selector that passes all frames through unchanged."""

    async def transform(
        self, frames: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[dict[str, Any]]:
        """Pass all frames through.

        Args:
            frames: Async iterator of JSON frames

        Yields:
            All frames unchanged
        """
        async for frame in frames:
            yield frame


def create_selector(expression: str | None) -> Transform | None:
    """Create a selector from an expression.

    Args:
        expression: Filter expression, or None for no filtering

    Returns:
        Selector instance, or None if no filtering needed
    """
    if not expression:
        return None

    return JsonPathSelector(expression)
