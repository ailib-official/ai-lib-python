"""事件映射：将提供商特定的流帧转换为统一的 StreamingEvent 模型。

Event mappers for converting protocol frames to unified streaming events.

Maps provider-specific streaming formats to the unified StreamingEvent model.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ai_lib_python.pipeline.base import EventMapper
from ai_lib_python.types.events import StreamingEvent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ai_lib_python.protocol.manifest import (
        EventMapRule,
        StreamingConfig,
        ToolingConfig,
    )


class ProtocolEventMapper(EventMapper):
    """Event mapper driven by protocol manifest rules.

    Uses event_map rules from the protocol manifest to convert
    provider-specific frames to unified StreamingEvent objects.

    Example manifest rules:
        event_map:
          - match: "exists($.choices[*].delta.content)"
            emit: "PartialContentDelta"
            fields:
              content: "$.choices[*].delta.content"
    """

    def __init__(self, rules: list[EventMapRule]) -> None:
        """Initialize the mapper.

        Args:
            rules: Event mapping rules from manifest
        """
        self._rules = rules
        self._compiled_rules = [self._compile_rule(r) for r in rules]

    def _compile_rule(self, rule: EventMapRule) -> dict[str, Any]:
        """Compile a rule for efficient evaluation.

        Args:
            rule: Event map rule

        Returns:
            Compiled rule dict
        """
        return {
            "match": rule.match,
            "emit": rule.emit,
            "fields": rule.fields,
            "matcher": self._create_matcher(rule.match),
        }

    def _create_matcher(self, expr: str) -> Any:
        """Create a matcher function for an expression.

        Args:
            expr: Match expression

        Returns:
            Callable that checks if frame matches
        """
        from ai_lib_python.pipeline.select import JsonPathSelector

        selector = JsonPathSelector(expr)
        return selector.matches

    def _extract_field(self, frame: dict[str, Any], path: str) -> Any:
        """Extract a field value using JSONPath.

        Args:
            frame: JSON frame
            path: JSONPath expression

        Returns:
            Extracted value, or None
        """
        from ai_lib_python.pipeline.select import JsonPathSelector

        selector = JsonPathSelector(f"exists({path})")
        return selector._get_value(frame, path)

    async def map_events(
        self, frames: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[StreamingEvent]:
        """Map frames to streaming events using protocol rules.

        Args:
            frames: Async iterator of JSON frames

        Yields:
            Unified streaming events
        """
        async for frame in frames:
            # Try each rule in order
            for rule in self._compiled_rules:
                if rule["matcher"](frame):
                    event = self._create_event(frame, rule)
                    if event:
                        yield event
                        break  # Only emit one event per frame

    def _create_event(
        self, frame: dict[str, Any], rule: dict[str, Any]
    ) -> StreamingEvent | None:
        """Create an event from a frame using a rule.

        Args:
            frame: JSON frame
            rule: Compiled rule

        Returns:
            StreamingEvent, or None if creation fails
        """
        emit_type = rule["emit"]
        fields = rule["fields"]

        # Extract field values
        extracted = {}
        for field_name, path in fields.items():
            value = self._extract_field(frame, path)
            if value is not None:
                extracted[field_name] = value

        # Create appropriate event
        try:
            if emit_type == "PartialContentDelta":
                content = extracted.get("content", "")
                if content:
                    return StreamingEvent.content_delta(
                        content=str(content),
                        sequence_id=extracted.get("sequence_id"),
                    )

            elif emit_type == "ThinkingDelta":
                thinking = extracted.get("thinking", "")
                if thinking:
                    return StreamingEvent.thinking_delta(
                        thinking=str(thinking),
                        tool_consideration=extracted.get("tool_consideration"),
                    )

            elif emit_type == "ToolCallStarted":
                tool_call_id = extracted.get("tool_call_id", "")
                tool_name = extracted.get("tool_name", "")
                if tool_call_id or tool_name:
                    return StreamingEvent.tool_call_started(
                        tool_call_id=str(tool_call_id or ""),
                        tool_name=str(tool_name or ""),
                        index=extracted.get("index"),
                    )

            elif emit_type == "PartialToolCall":
                arguments = extracted.get("arguments", "")
                # Also check accumulated tool call
                accumulated = frame.get("_accumulated_tool_call", {})
                tool_call_id = accumulated.get("id") or extracted.get("tool_call_id", "")
                if arguments or tool_call_id:
                    return StreamingEvent.partial_tool_call(
                        tool_call_id=str(tool_call_id or ""),
                        arguments=str(arguments or ""),
                        index=extracted.get("index"),
                        is_complete=accumulated.get("is_complete"),
                    )

            elif emit_type == "ToolCallEnded":
                tool_call_id = extracted.get("tool_call_id", "")
                return StreamingEvent.tool_call_ended(
                    tool_call_id=str(tool_call_id),
                    index=extracted.get("index"),
                )

            elif emit_type == "Metadata":
                return StreamingEvent.metadata(
                    usage=extracted.get("usage"),
                    finish_reason=extracted.get("finish_reason"),
                    stop_reason=extracted.get("stop_reason"),
                )

            elif emit_type == "FinalCandidate":
                candidate_index = extracted.get("candidate_index", 0)
                finish_reason = extracted.get("finish_reason", "")
                return StreamingEvent.final_candidate(
                    candidate_index=int(candidate_index) if candidate_index else 0,
                    finish_reason=str(finish_reason or "stop"),
                )

            elif emit_type == "StreamEnd":
                return StreamingEvent.stream_end(
                    finish_reason=extracted.get("finish_reason"),
                )

            elif emit_type == "StreamError":
                return StreamingEvent.stream_error(
                    error=extracted.get("error", frame.get("error", {})),
                    event_id=extracted.get("event_id"),
                )

        except Exception:
            # Failed to create event, skip
            pass

        return None


class DefaultEventMapper(EventMapper):
    """Default event mapper for OpenAI-compatible streaming.

    Uses standard paths to extract events without explicit rules.
    """

    def __init__(
        self,
        content_path: str = "choices[0].delta.content",
        tool_call_path: str = "choices[0].delta.tool_calls",
        finish_reason_path: str = "choices[0].finish_reason",
        usage_path: str = "usage",
    ) -> None:
        """Initialize the default mapper.

        Args:
            content_path: Path to content field
            tool_call_path: Path to tool calls
            finish_reason_path: Path to finish reason
            usage_path: Path to usage data
        """
        self._content_path = content_path
        self._tool_call_path = tool_call_path
        self._finish_reason_path = finish_reason_path
        self._usage_path = usage_path

        # Tool call accumulation state
        self._tool_calls: dict[int, dict[str, Any]] = {}

    def _get_nested(self, data: dict[str, Any], path: str) -> Any:
        """Get nested value by dot path.

        Args:
            data: Data dict
            path: Dot-separated path

        Returns:
            Value at path, or None
        """
        current: Any = data

        for part in path.replace("[", ".").replace("]", "").split("."):
            if not part:
                continue

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

    async def map_events(
        self, frames: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[StreamingEvent]:
        """Map frames to streaming events.

        Args:
            frames: Async iterator of JSON frames

        Yields:
            Unified streaming events
        """
        async for frame in frames:
            # Check for error
            if "error" in frame:
                yield StreamingEvent.stream_error(error=frame["error"])
                continue

            # Extract content delta
            content = self._get_nested(frame, self._content_path)
            if content:
                yield StreamingEvent.content_delta(content=str(content))

            # Extract tool calls
            tool_calls = self._get_nested(frame, self._tool_call_path)
            if tool_calls and isinstance(tool_calls, list):
                for tc in tool_calls:
                    index = tc.get("index", 0)
                    tc_id = tc.get("id")
                    func = tc.get("function", {})
                    name = func.get("name")
                    args = func.get("arguments", "")

                    # Initialize or update accumulated state
                    if index not in self._tool_calls:
                        self._tool_calls[index] = {
                            "id": tc_id,
                            "name": name,
                            "arguments": "",
                        }

                    if tc_id:
                        self._tool_calls[index]["id"] = tc_id
                    if name:
                        self._tool_calls[index]["name"] = name

                    # Emit tool call started if we got the name
                    if name and not self._tool_calls[index].get("_started"):
                        self._tool_calls[index]["_started"] = True
                        yield StreamingEvent.tool_call_started(
                            tool_call_id=self._tool_calls[index].get("id", ""),
                            tool_name=name,
                            index=index,
                        )

                    # Accumulate and emit partial arguments
                    if args:
                        self._tool_calls[index]["arguments"] += args
                        accumulated = self._tool_calls[index]["arguments"]

                        # Check if complete JSON
                        is_complete = False
                        try:
                            json.loads(accumulated)
                            is_complete = True
                        except json.JSONDecodeError:
                            pass

                        yield StreamingEvent.partial_tool_call(
                            tool_call_id=self._tool_calls[index].get("id", ""),
                            arguments=args,
                            index=index,
                            is_complete=is_complete,
                        )

            # Extract finish reason
            finish_reason = self._get_nested(frame, self._finish_reason_path)
            if finish_reason:
                yield StreamingEvent.metadata(finish_reason=str(finish_reason))

            # Extract usage
            usage = self._get_nested(frame, self._usage_path)
            if usage:
                yield StreamingEvent.metadata(usage=usage)


class AnthropicEventMapper(EventMapper):
    """Event mapper for Anthropic's streaming format."""

    def __init__(self) -> None:
        """Initialize Anthropic mapper."""
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._current_block_type: str | None = None

    async def map_events(
        self, frames: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[StreamingEvent]:
        """Map Anthropic frames to streaming events.

        Args:
            frames: Async iterator of JSON frames

        Yields:
            Unified streaming events
        """
        async for frame in frames:
            event_type = frame.get("type", "")

            # Error event
            if event_type == "error":
                yield StreamingEvent.stream_error(error=frame.get("error", {}))
                continue

            # Content block start
            if event_type == "content_block_start":
                content_block = frame.get("content_block", {})
                block_type = content_block.get("type")
                index = frame.get("index", 0)

                self._current_block_type = block_type

                if block_type == "tool_use":
                    self._tool_calls[index] = {
                        "id": content_block.get("id", ""),
                        "name": content_block.get("name", ""),
                        "arguments": "",
                    }
                    yield StreamingEvent.tool_call_started(
                        tool_call_id=content_block.get("id", ""),
                        tool_name=content_block.get("name", ""),
                        index=index,
                    )

            # Content block delta
            elif event_type == "content_block_delta":
                delta = frame.get("delta", {})
                delta_type = delta.get("type", "")
                index = frame.get("index", 0)

                if delta_type == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        yield StreamingEvent.content_delta(content=text)

                elif delta_type == "thinking_delta":
                    thinking = delta.get("thinking", "")
                    if thinking:
                        yield StreamingEvent.thinking_delta(thinking=thinking)

                elif delta_type == "input_json_delta":
                    partial_json = delta.get("partial_json", "")
                    if index in self._tool_calls:
                        self._tool_calls[index]["arguments"] += partial_json

                        # Check if complete
                        accumulated = self._tool_calls[index]["arguments"]
                        is_complete = False
                        try:
                            json.loads(accumulated)
                            is_complete = True
                        except json.JSONDecodeError:
                            pass

                        yield StreamingEvent.partial_tool_call(
                            tool_call_id=self._tool_calls[index]["id"],
                            arguments=partial_json,
                            index=index,
                            is_complete=is_complete,
                        )

            # Content block stop
            elif event_type == "content_block_stop":
                index = frame.get("index", 0)
                if index in self._tool_calls:
                    yield StreamingEvent.tool_call_ended(
                        tool_call_id=self._tool_calls[index]["id"],
                        index=index,
                    )

            # Message delta (contains usage and stop reason)
            elif event_type == "message_delta":
                delta = frame.get("delta", {})
                usage = frame.get("usage", {})

                stop_reason = delta.get("stop_reason")
                if stop_reason:
                    yield StreamingEvent.metadata(
                        finish_reason=stop_reason,
                        stop_reason=stop_reason,
                    )

                if usage:
                    yield StreamingEvent.metadata(usage=usage)

            # Message stop
            elif event_type == "message_stop":
                yield StreamingEvent.stream_end()


def create_event_mapper(
    streaming_config: StreamingConfig | None,
    tooling_config: ToolingConfig | None = None,
) -> EventMapper:
    """Create an event mapper from configuration.

    Args:
        streaming_config: Streaming configuration from manifest
        tooling_config: Tooling configuration from manifest

    Returns:
        Appropriate EventMapper instance
    """
    if not streaming_config:
        return DefaultEventMapper()

    # If explicit event_map rules are provided, use protocol mapper
    if streaming_config.event_map:
        return ProtocolEventMapper(streaming_config.event_map)

    # Check decoder strategy for provider-specific mappers
    if streaming_config.decoder:
        strategy = streaming_config.decoder.strategy
        if strategy == "anthropic_event_stream":
            return AnthropicEventMapper()

    # Use default mapper with configured paths
    return DefaultEventMapper(
        content_path=streaming_config.content_path or "choices[0].delta.content",
        tool_call_path=streaming_config.tool_call_path or "choices[0].delta.tool_calls",
        usage_path=streaming_config.usage_path or "usage",
    )
