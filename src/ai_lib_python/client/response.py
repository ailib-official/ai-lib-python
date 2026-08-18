"""
Response types for client operations.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ai_lib_python.types.message import Message

if TYPE_CHECKING:
    from ai_lib_python.types.tool import ToolCall


@dataclass
class ChatResponse:
    """Response from a chat completion request.

    Attributes:
        content: The generated text content
        thinking: Aggregated extended thinking / reasoning (ALP-RSN-001).
            Empty when absent. Additive — does not change ``content`` semantics.
        tool_calls: List of tool calls requested by the model
        finish_reason: Why the model stopped generating
        usage: Token usage information
        model: Model that generated the response
        raw_response: Raw response data from the API
    """

    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    model: str | None = None
    raw_response: dict[str, Any] | None = None

    def to_message(self) -> Message:
        """Convert response to an assistant message.

        Returns:
            Message with assistant role and response content
        """
        return Message.assistant(self.content)

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0

    @property
    def prompt_tokens(self) -> int | None:
        """Get prompt token count from usage."""
        if self.usage:
            return self.usage.get("prompt_tokens")
        return None

    @property
    def completion_tokens(self) -> int | None:
        """Get completion token count from usage."""
        if self.usage:
            return self.usage.get("completion_tokens")
        return None

    @property
    def total_tokens(self) -> int | None:
        """Get total token count from usage."""
        if self.usage:
            return self.usage.get("total_tokens")
        return None

    @property
    def reasoning_tokens(self) -> int | None:
        """Get reasoning token count from usage (OpenAI completion_tokens_details)."""
        if not self.usage:
            return None
        details = self.usage.get("completion_tokens_details")
        if isinstance(details, dict):
            return details.get("reasoning_tokens")
        return None

    @property
    def input_tokens(self) -> int | None:
        """Get input token count (Anthropic-style)."""
        if self.usage:
            return self.usage.get("input_tokens")
        return None

    @property
    def output_tokens(self) -> int | None:
        """Get output token count (Anthropic-style)."""
        if self.usage:
            return self.usage.get("output_tokens")
        return None

    @property
    def cache_read_tokens(self) -> int | None:
        """Get cache read tokens (Anthropic-style)."""
        if self.usage:
            return self.usage.get("cache_read_input_tokens")
        return None

    @property
    def cache_creation_tokens(self) -> int | None:
        """Get cache creation tokens (Anthropic-style)."""
        if self.usage:
            return self.usage.get("cache_creation_input_tokens")
        return None


@dataclass
class CallStats:
    """Statistics for a single API call.

    Attributes:
        client_request_id: Client-generated request ID for tracking
        latency_ms: Total latency in milliseconds
        time_to_first_token_ms: Time to first token (streaming only)
        retry_count: Number of retries performed
        model: Model used for the call
        provider: Provider ID
        endpoint: API endpoint used
        prompt_tokens: Prompt token count
        completion_tokens: Completion token count
    """

    client_request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    latency_ms: float = 0.0
    time_to_first_token_ms: float | None = None
    retry_count: int = 0
    model: str | None = None
    provider: str | None = None
    endpoint: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    # Internal timing
    _start_time: float = field(default_factory=time.time, repr=False)
    _first_token_time: float | None = field(default=None, repr=False)

    def record_start(self) -> None:
        """Record the start time."""
        self._start_time = time.time()

    def record_first_token(self) -> None:
        """Record time of first token."""
        if self._first_token_time is None:
            self._first_token_time = time.time()
            self.time_to_first_token_ms = (self._first_token_time - self._start_time) * 1000

    def record_end(self) -> None:
        """Record the end time and calculate latency."""
        self.latency_ms = (time.time() - self._start_time) * 1000

    def record_usage(self, usage: dict[str, Any] | None) -> None:
        """Record token usage.

        Args:
            usage: Usage dict from API response
        """
        if usage:
            self.prompt_tokens = usage.get("prompt_tokens")
            self.completion_tokens = usage.get("completion_tokens")

    @property
    def total_tokens(self) -> int | None:
        """Get total token count."""
        if self.prompt_tokens is not None and self.completion_tokens is not None:
            return self.prompt_tokens + self.completion_tokens
        return None
