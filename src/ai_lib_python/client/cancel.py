"""
Stream cancellation control.

Provides cancellation tokens and handles for controlling streaming requests.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class CancelReason(str, Enum):
    """Reasons for cancellation."""

    USER_REQUEST = "user_request"
    TIMEOUT = "timeout"
    ERROR = "error"
    RESOURCE_LIMIT = "resource_limit"
    SHUTDOWN = "shutdown"


@dataclass
class CancelState:
    """State of a cancellation token.

    Attributes:
        cancelled: Whether cancellation was requested
        reason: Reason for cancellation
        timestamp: Time of cancellation
    """

    cancelled: bool = False
    reason: CancelReason | None = None
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CancelToken:
    """Cancellation token for controlling async operations.

    A CancelToken can be passed to async operations and checked
    periodically to support cooperative cancellation.

    Example:
        >>> token = CancelToken()
        >>>
        >>> async def process_stream(stream, token):
        ...     async for item in stream:
        ...         if token.is_cancelled:
        ...             break
        ...         await process(item)
        >>>
        >>> # Cancel from another task
        >>> token.cancel(CancelReason.USER_REQUEST)
    """

    def __init__(self, timeout: float | None = None) -> None:
        """Initialize cancellation token.

        Args:
            timeout: Optional timeout in seconds
        """
        self._state = CancelState()
        self._event = asyncio.Event()
        self._callbacks: list[Any] = []  # Callable[[CancelReason], Any]
        self._timeout = timeout
        self._timeout_task: asyncio.Task[None] | None = None

        if timeout:
            self._start_timeout()

    def _start_timeout(self) -> None:
        """Start the timeout task."""
        timeout = self._timeout
        if timeout is None:
            return

        async def timeout_handler() -> None:
            await asyncio.sleep(timeout)
            if not self._state.cancelled:
                self.cancel(CancelReason.TIMEOUT)

        try:
            loop = asyncio.get_running_loop()
            self._timeout_task = loop.create_task(timeout_handler())
        except RuntimeError:
            # No running loop - timeout will be created when first awaited
            pass

    def cancel(
        self,
        reason: CancelReason = CancelReason.USER_REQUEST,
        **metadata: Any,
    ) -> bool:
        """Request cancellation.

        Args:
            reason: Reason for cancellation
            **metadata: Additional metadata

        Returns:
            True if cancellation was newly requested, False if already cancelled
        """
        if self._state.cancelled:
            return False

        import time

        self._state.cancelled = True
        self._state.reason = reason
        self._state.timestamp = time.time()
        self._state.metadata.update(metadata)

        # Signal the event
        self._event.set()

        # Stop timeout task if running
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()

        # Invoke callbacks
        for callback in self._callbacks:
            try:
                result = callback(reason)
                if asyncio.iscoroutine(result):
                    _ = asyncio.create_task(result)  # noqa: RUF006
            except Exception:
                pass

        return True

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._state.cancelled

    @property
    def reason(self) -> CancelReason | None:
        """Get cancellation reason."""
        return self._state.reason

    @property
    def state(self) -> CancelState:
        """Get full cancellation state."""
        return self._state

    async def wait(self) -> CancelReason:
        """Wait until cancellation is requested.

        Returns:
            Cancellation reason
        """
        await self._event.wait()
        return self._state.reason or CancelReason.USER_REQUEST

    async def wait_with_timeout(self, timeout: float) -> bool:
        """Wait for cancellation with a timeout.

        Args:
            timeout: Timeout in seconds

        Returns:
            True if cancelled, False if timeout occurred
        """
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def on_cancel(self, callback: Callable[[CancelReason], Any]) -> CancelToken:
        """Register a callback to be called on cancellation.

        Args:
            callback: Callback function

        Returns:
            Self for chaining
        """
        self._callbacks.append(callback)
        # If already cancelled, call immediately
        if self._state.cancelled and self._state.reason:
            try:
                result = callback(self._state.reason)
                if asyncio.iscoroutine(result):
                    _ = asyncio.create_task(result)  # noqa: RUF006
            except Exception:
                pass
        return self

    def raise_if_cancelled(self) -> None:
        """Raise CancelledError if cancelled.

        Raises:
            asyncio.CancelledError: If cancellation was requested
        """
        if self._state.cancelled:
            raise asyncio.CancelledError(str(self._state.reason))

    def reset(self) -> None:
        """Reset the token to uncancelled state."""
        self._state = CancelState()
        self._event.clear()


class CancelHandle:
    """Handle for cancelling streaming operations.

    Provides a public interface for users to cancel operations,
    while the token is used internally by the operation.

    Example:
        >>> async def streaming_operation():
        ...     handle, token = create_cancel_pair()
        ...     task = asyncio.create_task(process_with_token(token))
        ...     return stream, handle
        >>>
        >>> stream, handle = await streaming_operation()
        >>> # Later...
        >>> handle.cancel()  # Cancel the stream
    """

    def __init__(self, token: CancelToken) -> None:
        """Initialize cancel handle.

        Args:
            token: Associated cancel token
        """
        self._token = token

    def cancel(
        self,
        reason: CancelReason = CancelReason.USER_REQUEST,
        **metadata: Any,
    ) -> bool:
        """Request cancellation.

        Args:
            reason: Reason for cancellation
            **metadata: Additional metadata

        Returns:
            True if cancellation was newly requested
        """
        return self._token.cancel(reason, **metadata)

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._token.is_cancelled

    @property
    def reason(self) -> CancelReason | None:
        """Get cancellation reason."""
        return self._token.reason


def create_cancel_pair(
    timeout: float | None = None,
) -> tuple[CancelHandle, CancelToken]:
    """Create a cancel handle and token pair.

    Args:
        timeout: Optional timeout in seconds

    Returns:
        Tuple of (CancelHandle, CancelToken)
    """
    token = CancelToken(timeout=timeout)
    handle = CancelHandle(token)
    return handle, token


class CancellableStream:
    """Wrapper for making async iterators cancellable.

    Wraps an async iterator and checks for cancellation between items.

    Example:
        >>> token = CancelToken()
        >>> cancellable = CancellableStream(original_stream, token)
        >>> async for item in cancellable:
        ...     process(item)
    """

    def __init__(
        self,
        stream: Any,  # AsyncIterator
        token: CancelToken,
        on_cancel: Callable[[], Any] | None = None,
    ) -> None:
        """Initialize cancellable stream.

        Args:
            stream: Async iterator to wrap
            token: Cancellation token
            on_cancel: Optional callback when cancelled
        """
        self._stream = stream
        self._token = token
        self._on_cancel = on_cancel
        self._started = False
        self._finished = False

    def __aiter__(self) -> CancellableStream:
        """Return async iterator."""
        return self

    async def __anext__(self) -> Any:
        """Get next item, checking for cancellation."""
        if self._token.is_cancelled or self._finished:
            if self._on_cancel and not self._finished:
                self._finished = True
                try:
                    result = self._on_cancel()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    pass
            raise StopAsyncIteration

        try:
            item = await self._stream.__anext__()
            self._started = True
            return item
        except StopAsyncIteration:
            self._finished = True
            raise

    async def close(self) -> None:
        """Close the stream."""
        self._finished = True
        if hasattr(self._stream, "close"):
            close_result = self._stream.close()
            if asyncio.iscoroutine(close_result):
                await close_result
        elif hasattr(self._stream, "aclose"):
            await self._stream.aclose()

    @property
    def started(self) -> bool:
        """Check if stream has started (emitted any items)."""
        return self._started

    @property
    def finished(self) -> bool:
        """Check if stream has finished."""
        return self._finished


async def with_cancellation(
    stream: Any,  # AsyncIterator
    token: CancelToken,
    on_cancel: Callable[[], Any] | None = None,
) -> Any:  # AsyncIterator
    """Create a cancellable stream from an async iterator.

    Args:
        stream: Async iterator to wrap
        token: Cancellation token
        on_cancel: Optional callback when cancelled

    Returns:
        CancellableStream wrapping the input
    """
    return CancellableStream(stream, token, on_cancel)
