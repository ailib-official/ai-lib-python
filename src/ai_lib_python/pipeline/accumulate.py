"""
Accumulators for stateful stream processing.

Handles accumulation of partial data, particularly for tool call arguments
that may be split across multiple streaming chunks.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ai_lib_python.pipeline.base import Transform

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class ToolCallAccumulator(Transform):
    """Accumulates tool call arguments from streaming chunks.

    Tool calls in streaming often arrive as partial JSON strings:
    - Chunk 1: {"name": "get_weather", "arguments": '{"city":'}
    - Chunk 2: {"arguments": ' "Tokyo"'}
    - Chunk 3: {"arguments": '}'}

    This accumulator collects these fragments and emits complete tool calls.
    """

    def __init__(
        self,
        tool_call_id_path: str = "$.choices[*].delta.tool_calls[*].id",
        tool_name_path: str = "$.choices[*].delta.tool_calls[*].function.name",
        arguments_path: str = "$.choices[*].delta.tool_calls[*].function.arguments",
        index_path: str = "$.choices[*].delta.tool_calls[*].index",
    ) -> None:
        """Initialize the accumulator.

        Args:
            tool_call_id_path: Path to tool call ID
            tool_name_path: Path to tool name
            arguments_path: Path to arguments fragment
            index_path: Path to tool call index
        """
        self._tool_call_id_path = tool_call_id_path
        self._tool_name_path = tool_name_path
        self._arguments_path = arguments_path
        self._index_path = index_path

        # State: accumulated arguments per tool call index
        self._accumulated: dict[int, dict[str, Any]] = {}

    def _get_path_value(self, frame: dict[str, Any], path: str) -> Any:
        """Extract value at a simplified JSONPath.

        Args:
            frame: JSON frame
            path: Simplified JSONPath

        Returns:
            Value at path, or None
        """
        # Remove $. prefix
        if path.startswith("$."):
            path = path[2:]

        current: Any = frame

        # Handle array wildcards by getting first match
        parts = path.replace("[*]", ".0.").split(".")
        parts = [p for p in parts if p]

        for part in parts:
            if current is None:
                return None

            if part.isdigit():
                idx = int(part)
                if isinstance(current, list) and len(current) > idx:
                    current = current[idx]
                else:
                    return None
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                return None

        return current

    async def transform(
        self, frames: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[dict[str, Any]]:
        """Process frames and accumulate tool call arguments.

        This transform:
        1. Passes through non-tool-call frames unchanged
        2. Accumulates tool call argument fragments
        3. Annotates frames with accumulated state

        Args:
            frames: Async iterator of JSON frames

        Yields:
            Frames with tool call state annotated
        """
        async for frame in frames:
            # Try to extract tool call data
            tool_calls = self._extract_tool_calls(frame)

            if tool_calls:
                # Process each tool call in the frame
                for tc in tool_calls:
                    index = tc.get("index", 0)
                    if isinstance(index, str):
                        index = int(index) if index.isdigit() else 0

                    # Initialize accumulator for this index if needed
                    if index not in self._accumulated:
                        self._accumulated[index] = {
                            "id": tc.get("id"),
                            "name": tc.get("name"),
                            "arguments": "",
                            "index": index,
                        }

                    # Update with new data
                    if tc.get("id"):
                        self._accumulated[index]["id"] = tc["id"]
                    if tc.get("name"):
                        self._accumulated[index]["name"] = tc["name"]
                    if tc.get("arguments"):
                        self._accumulated[index]["arguments"] += tc["arguments"]

                    # Annotate frame with accumulated state
                    frame["_accumulated_tool_call"] = {
                        "index": index,
                        "id": self._accumulated[index]["id"],
                        "name": self._accumulated[index]["name"],
                        "arguments": self._accumulated[index]["arguments"],
                        "is_complete": self._is_complete_json(
                            self._accumulated[index]["arguments"]
                        ),
                    }

            yield frame

    def _extract_tool_calls(self, frame: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract tool call data from a frame.

        Args:
            frame: JSON frame

        Returns:
            List of tool call data dicts
        """
        tool_calls: list[dict[str, Any]] = []

        # Try OpenAI format: choices[*].delta.tool_calls[*]
        choices: Any = frame.get("choices", [])
        for choice in choices:
            delta = choice.get("delta", {})
            tcs = delta.get("tool_calls", [])
            for tc in tcs:
                tool_calls.append(
                    {
                        "index": tc.get("index", 0),
                        "id": tc.get("id"),
                        "name": tc.get("function", {}).get("name"),
                        "arguments": tc.get("function", {}).get("arguments", ""),
                    }
                )

        # Try Anthropic format: content_block with type tool_use
        if frame.get("type") == "content_block_start":
            content_block = frame.get("content_block", {})
            if content_block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "index": frame.get("index", 0),
                        "id": content_block.get("id"),
                        "name": content_block.get("name"),
                        "arguments": "",
                    }
                )

        # Anthropic input_json_delta
        if frame.get("type") == "content_block_delta":
            delta = frame.get("delta", {})
            if delta.get("type") == "input_json_delta":
                tool_calls.append(
                    {
                        "index": frame.get("index", 0),
                        "arguments": delta.get("partial_json", ""),
                    }
                )

        return tool_calls

    def _is_complete_json(self, s: str) -> bool:
        """Check if a string is complete JSON.

        Args:
            s: JSON string to check

        Returns:
            True if string is valid, complete JSON
        """
        if not s:
            return False

        try:
            json.loads(s)
            return True
        except json.JSONDecodeError:
            return False

    def reset(self) -> None:
        """Reset accumulated state."""
        self._accumulated.clear()

    def get_accumulated(self, index: int = 0) -> dict[str, Any] | None:
        """Get accumulated tool call data for an index.

        Args:
            index: Tool call index

        Returns:
            Accumulated data, or None if not found
        """
        return self._accumulated.get(index)


def create_accumulator(config: dict[str, Any] | None) -> Transform | None:
    """Create an accumulator from configuration.

    Args:
        config: Accumulator configuration

    Returns:
        Accumulator instance, or None if not configured
    """
    if not config:
        return None

    if config.get("stateful_tool_parsing"):
        return ToolCallAccumulator()

    return None
