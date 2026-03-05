"""
Base abstractions for the pipeline layer.

Defines the core interfaces that all pipeline operators implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ai_lib_python.protocol.manifest import ProtocolManifest
    from ai_lib_python.types.events import StreamingEvent


class Decoder(ABC):
    """Abstract decoder that converts byte stream to JSON frames.

    Decoders handle the transport-level parsing of streaming responses,
    converting raw bytes into structured JSON objects.
    """

    @abstractmethod
    def decode(
        self, byte_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[dict[str, Any]]:
        """Decode a byte stream into JSON frames.

        Args:
            byte_stream: Async iterator of raw bytes

        Yields:
            Parsed JSON frames as dictionaries
        """
        ...


class Transform(ABC):
    """Abstract transform that processes JSON frames.

    Transforms operate on the stream of JSON frames, performing
    operations like filtering, accumulation, or fan-out.
    """

    @abstractmethod
    def transform(
        self, frames: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[dict[str, Any]]:
        """Transform a stream of frames.

        Args:
            frames: Async iterator of JSON frames

        Yields:
            Transformed JSON frames
        """
        ...


class EventMapper(ABC):
    """Abstract mapper that converts JSON frames to streaming events.

    EventMappers are the final stage of the pipeline, converting
    protocol-specific frames into unified StreamingEvent objects.
    """

    @abstractmethod
    def map_events(
        self, frames: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[StreamingEvent]:
        """Map frames to streaming events.

        Args:
            frames: Async iterator of JSON frames

        Yields:
            Unified streaming events
        """
        ...


class Pipeline:
    """Complete pipeline for processing streaming responses.

    A pipeline consists of:
    1. A decoder (bytes -> frames)
    2. Zero or more transforms (frames -> frames)
    3. An event mapper (frames -> events)

    Example:
        >>> pipeline = Pipeline.from_manifest(manifest)
        >>> async for event in pipeline.process(byte_stream):
        ...     print(event)
    """

    def __init__(
        self,
        decoder: Decoder,
        transforms: list[Transform] | None = None,
        event_mapper: EventMapper | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            decoder: The decoder to convert bytes to frames
            transforms: Optional list of transforms to apply
            event_mapper: Optional event mapper for final conversion
        """
        self._decoder = decoder
        self._transforms = transforms or []
        self._event_mapper = event_mapper

    async def process(
        self, byte_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[StreamingEvent]:
        """Process a byte stream through the complete pipeline.

        Args:
            byte_stream: Async iterator of raw bytes

        Yields:
            Unified streaming events
        """
        # Stage 1: Decode bytes to frames
        frames = self._decoder.decode(byte_stream)

        # Stage 2: Apply transforms in sequence
        for transform in self._transforms:
            frames = transform.transform(frames)

        # Stage 3: Map to events
        if self._event_mapper:
            async for event in self._event_mapper.map_events(frames):
                yield event
        else:
            # If no mapper, yield raw frames wrapped as events
            from ai_lib_python.types.events import StreamingEvent

            async for frame in frames:
                yield StreamingEvent.content_delta(str(frame))

    async def decode_only(
        self, byte_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[dict[str, Any]]:
        """Process only through decoder and transforms, without event mapping.

        Useful for debugging or when custom event handling is needed.

        Args:
            byte_stream: Async iterator of raw bytes

        Yields:
            JSON frames after decoding and transforms
        """
        frames = self._decoder.decode(byte_stream)

        for transform in self._transforms:
            frames = transform.transform(frames)

        async for frame in frames:
            yield frame

    @classmethod
    def from_manifest(cls, manifest: ProtocolManifest) -> Pipeline:
        """Create a pipeline from a protocol manifest.

        Args:
            manifest: The protocol manifest with streaming configuration

        Returns:
            Configured Pipeline instance
        """
        from ai_lib_python.pipeline.decode import create_decoder
        from ai_lib_python.pipeline.event_map import create_event_mapper
        from ai_lib_python.pipeline.select import create_selector

        streaming = manifest.streaming
        if not streaming:
            # Create a default SSE pipeline
            from ai_lib_python.pipeline.decode import SSEDecoder
            from ai_lib_python.pipeline.event_map import DefaultEventMapper

            return cls(
                decoder=SSEDecoder(),
                event_mapper=DefaultEventMapper(),
            )

        # Create decoder
        decoder = create_decoder(streaming.decoder)

        # Create transforms
        transforms: list[Transform] = []

        # Add selector if frame_selector is specified
        if streaming.frame_selector:
            selector = create_selector(streaming.frame_selector)
            if selector:
                transforms.append(selector)

        # Create event mapper
        event_mapper = create_event_mapper(streaming, manifest.tooling)

        return cls(
            decoder=decoder,
            transforms=transforms,
            event_mapper=event_mapper,
        )

    def with_transform(self, transform: Transform) -> Pipeline:
        """Add a transform to the pipeline.

        Args:
            transform: Transform to add

        Returns:
            New Pipeline with the transform added
        """
        return Pipeline(
            decoder=self._decoder,
            transforms=[*self._transforms, transform],
            event_mapper=self._event_mapper,
        )

    def with_event_mapper(self, mapper: EventMapper) -> Pipeline:
        """Set the event mapper.

        Args:
            mapper: Event mapper to use

        Returns:
            New Pipeline with the mapper set
        """
        return Pipeline(
            decoder=self._decoder,
            transforms=self._transforms,
            event_mapper=mapper,
        )
