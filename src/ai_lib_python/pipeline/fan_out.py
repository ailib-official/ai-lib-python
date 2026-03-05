"""
FanOut stream transform operator.

Provides stream splitting and array element expansion capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_lib_python.pipeline.base import Transform

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


class FanOutTransform(Transform):
    """Transform that expands array elements into separate events.

    When encountering an array in the stream, this transform emits each
    element as a separate event. Non-array values pass through unchanged.

    This is useful for processing multi-candidate responses where the
    API returns an array of choices/candidates.

    Example:
        >>> async def source():
        ...     yield {"choices": [{"text": "a"}, {"text": "b"}]}
        ...
        >>> fan_out = FanOutTransform(array_path="choices")
        >>> async for item in fan_out.transform(source()):
        ...     print(item)
        {"text": "a"}
        {"text": "b"}
    """

    def __init__(
        self,
        enabled: bool = True,
        array_path: str | None = None,
        preserve_metadata: bool = True,
    ) -> None:
        """Initialize FanOut transform.

        Args:
            enabled: Whether fan-out is enabled
            array_path: JSON path to the array to expand (e.g., "choices", "candidates")
            preserve_metadata: Whether to preserve non-array metadata from parent
        """
        self._enabled = enabled
        self._array_path = array_path
        self._preserve_metadata = preserve_metadata

    async def transform(
        self, input_stream: AsyncIterator[Any]
    ) -> AsyncIterator[Any]:
        """Transform input stream by expanding arrays.

        Args:
            input_stream: Input async iterator

        Yields:
            Expanded items
        """
        if not self._enabled:
            async for item in input_stream:
                yield item
            return

        async for item in input_stream:
            async for expanded in self._expand(item):
                yield expanded

    async def _expand(self, item: Any) -> AsyncIterator[Any]:
        """Expand a single item.

        Args:
            item: Item to potentially expand

        Yields:
            Expanded items
        """
        if not isinstance(item, dict):
            # Non-dict items pass through
            if isinstance(item, list):
                # Top-level array - expand directly
                for element in item:
                    yield element
            else:
                yield item
            return

        # Try to extract array from specified path
        if self._array_path:
            array = self._get_path(item, self._array_path)
            if isinstance(array, list) and array:
                # Get metadata (non-array fields)
                metadata = {}
                if self._preserve_metadata:
                    for key, value in item.items():
                        if key != self._array_path.split(".")[0]:
                            metadata[key] = value

                # Emit each array element
                for element in array:
                    if self._preserve_metadata and metadata:
                        if isinstance(element, dict):
                            # Merge metadata into element
                            yield {**metadata, **element}
                        else:
                            # Wrap element with metadata
                            yield {**metadata, "value": element}
                    else:
                        yield element
                return

        # No array found at path, check for common patterns
        for path in ["choices", "candidates", "results", "data", "items"]:
            array = item.get(path)
            if isinstance(array, list) and array:
                metadata = {}
                if self._preserve_metadata:
                    metadata = {k: v for k, v in item.items() if k != path}

                for element in array:
                    if self._preserve_metadata and metadata:
                        if isinstance(element, dict):
                            yield {**metadata, **element}
                        else:
                            yield {**metadata, "value": element}
                    else:
                        yield element
                return

        # No array found, pass through
        yield item

    def _get_path(self, obj: dict[str, Any], path: str) -> Any:
        """Get value at a dot-separated path.

        Args:
            obj: Dictionary object
            path: Dot-separated path (e.g., "response.choices")

        Returns:
            Value at path or None
        """
        current: Any = obj
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                idx = int(part)
                current = current[idx] if 0 <= idx < len(current) else None
            else:
                return None
        return current


class ReplicateTransform(Transform):
    """Transform that replicates each event multiple times.

    Useful for testing or creating multiple processing paths.

    Example:
        >>> replicate = ReplicateTransform(count=3)
        >>> async for item in replicate.transform(source()):
        ...     # Each input item yields 3 copies
        ...     process(item)
    """

    def __init__(self, count: int = 2, add_index: bool = False) -> None:
        """Initialize replicate transform.

        Args:
            count: Number of copies to create
            add_index: Whether to add replica index to each copy
        """
        self._count = max(1, count)
        self._add_index = add_index

    async def transform(
        self, input_stream: AsyncIterator[Any]
    ) -> AsyncIterator[Any]:
        """Replicate each item in the stream.

        Args:
            input_stream: Input async iterator

        Yields:
            Replicated items
        """
        async for item in input_stream:
            for i in range(self._count):
                if self._add_index:
                    if isinstance(item, dict):
                        yield {**item, "_replica_index": i}
                    else:
                        yield {"value": item, "_replica_index": i}
                else:
                    yield item


class SplitTransform(Transform):
    """Transform that splits stream based on a predicate.

    Routes items to different output paths based on a condition.
    Items matching the predicate are yielded; others are collected
    in a separate buffer that can be retrieved later.

    Example:
        >>> split = SplitTransform(predicate=lambda x: x.get("type") == "content")
        >>> async for item in split.transform(source()):
        ...     # Only content items are yielded
        ...     process(item)
        >>> # Get filtered items
        >>> metadata = split.get_filtered()
    """

    def __init__(
        self,
        predicate: Callable[[Any], bool] | None = None,
        collect_filtered: bool = True,
    ) -> None:
        """Initialize split transform.

        Args:
            predicate: Function that returns True for items to pass through
            collect_filtered: Whether to collect filtered items
        """
        self._predicate = predicate or (lambda _: True)
        self._collect_filtered = collect_filtered
        self._filtered: list[Any] = []

    async def transform(
        self, input_stream: AsyncIterator[Any]
    ) -> AsyncIterator[Any]:
        """Split stream based on predicate.

        Args:
            input_stream: Input async iterator

        Yields:
            Items matching predicate
        """
        self._filtered.clear()

        async for item in input_stream:
            try:
                if self._predicate(item):
                    yield item
                elif self._collect_filtered:
                    self._filtered.append(item)
            except Exception:
                # On predicate error, pass through
                yield item

    def get_filtered(self) -> list[Any]:
        """Get items that were filtered out.

        Returns:
            List of filtered items
        """
        return list(self._filtered)

    def clear_filtered(self) -> None:
        """Clear the filtered items buffer."""
        self._filtered.clear()


def create_fan_out(
    enabled: bool = True,
    array_path: str | None = None,
) -> FanOutTransform:
    """Create a FanOut transform.

    Args:
        enabled: Whether fan-out is enabled
        array_path: JSON path to array to expand

    Returns:
        FanOutTransform instance
    """
    return FanOutTransform(enabled=enabled, array_path=array_path)
