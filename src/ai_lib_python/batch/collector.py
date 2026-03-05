"""
Batch collector for grouping requests.

Collects requests and groups them for batch execution.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchConfig:
    """Configuration for batch collection.

    Attributes:
        max_batch_size: Maximum requests per batch
        max_wait_ms: Maximum wait time before flushing batch
        group_by: Function to determine grouping key
    """

    max_batch_size: int = 10
    max_wait_ms: float = 100.0
    group_by: Callable[[Any], str] | None = None

    @classmethod
    def default(cls) -> BatchConfig:
        """Create default configuration."""
        return cls()

    @classmethod
    def for_embeddings(cls) -> BatchConfig:
        """Create configuration optimized for embeddings."""
        return cls(
            max_batch_size=100,
            max_wait_ms=50.0,
        )

    @classmethod
    def for_chat(cls) -> BatchConfig:
        """Create configuration for chat completions."""
        return cls(
            max_batch_size=5,
            max_wait_ms=10.0,
        )


@dataclass
class PendingRequest(Generic[T]):
    """A pending request waiting for batch execution.

    Attributes:
        data: Request data
        future: Future to resolve with result
        added_at: Timestamp when added
        group_key: Grouping key
    """

    data: T
    future: asyncio.Future[Any]
    added_at: float = field(default_factory=time.time)
    group_key: str = "_default_"


class BatchCollector(Generic[T, R]):
    """Collects requests for batch processing.

    Accumulates requests until batch size or time limit is reached,
    then triggers batch execution.

    Example:
        >>> async def process_batch(items):
        ...     return [f"result_{i}" for i in range(len(items))]
        ...
        >>> collector = BatchCollector(
        ...     config=BatchConfig(max_batch_size=5),
        ...     executor=process_batch,
        ... )
        >>> result = await collector.add("request1")
    """

    def __init__(
        self,
        config: BatchConfig | None = None,
        executor: Callable[[list[T]], Awaitable[list[R]]] | None = None,
    ) -> None:
        """Initialize batch collector.

        Args:
            config: Batch configuration
            executor: Function to execute batches
        """
        self._config = config or BatchConfig.default()
        self._executor = executor
        self._pending: dict[str, list[PendingRequest[T]]] = {}
        self._lock = asyncio.Lock()
        self._timers: dict[str, asyncio.Task[None]] = {}
        self._running = True

    def set_executor(
        self, executor: Callable[[list[T]], Awaitable[list[R]]]
    ) -> None:
        """Set the batch executor function.

        Args:
            executor: Function to execute batches
        """
        self._executor = executor

    async def add(self, data: T) -> R:
        """Add a request to the batch.

        Args:
            data: Request data

        Returns:
            Result from batch execution
        """
        if not self._running:
            raise RuntimeError("Batch collector is stopped")

        if self._executor is None:
            raise RuntimeError("No executor set")

        # Determine group key
        group_key = (
            self._config.group_by(data) if self._config.group_by else "_default_"
        )

        # Create future for result
        loop = asyncio.get_event_loop()
        future: asyncio.Future[R] = loop.create_future()

        async with self._lock:
            # Add to pending
            if group_key not in self._pending:
                self._pending[group_key] = []

            self._pending[group_key].append(
                PendingRequest(data=data, future=future, group_key=group_key)
            )

            # Check if batch is full
            if len(self._pending[group_key]) >= self._config.max_batch_size:
                await self._flush_group(group_key)
            else:
                # Start timer if not already running
                if group_key not in self._timers or self._timers[group_key].done():
                    self._timers[group_key] = asyncio.create_task(
                        self._timer_flush(group_key)
                    )

        return await future

    async def _timer_flush(self, group_key: str) -> None:
        """Flush group after timeout."""
        await asyncio.sleep(self._config.max_wait_ms / 1000.0)

        async with self._lock:
            if self._pending.get(group_key):
                await self._flush_group(group_key)

    async def _flush_group(self, group_key: str) -> None:
        """Flush a specific group.

        Args:
            group_key: Group to flush
        """
        if group_key not in self._pending or not self._pending[group_key]:
            return

        # Get pending requests
        requests = self._pending.pop(group_key)

        # Cancel timer
        if group_key in self._timers:
            self._timers[group_key].cancel()
            del self._timers[group_key]

        # Extract data
        data_list = [r.data for r in requests]

        try:
            # Execute batch
            executor = self._executor
            if executor is None:
                raise RuntimeError("No executor set")
            results = await executor(data_list)

            # Resolve futures
            for request, result in zip(requests, results, strict=False):
                if not request.future.done():
                    request.future.set_result(result)

        except Exception as e:
            # Reject all futures
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(e)

    async def flush(self) -> None:
        """Flush all pending batches."""
        async with self._lock:
            for group_key in list(self._pending.keys()):
                await self._flush_group(group_key)

    async def stop(self) -> None:
        """Stop the collector and flush pending requests."""
        self._running = False
        await self.flush()

        # Cancel all timers
        for timer in self._timers.values():
            timer.cancel()
        self._timers.clear()

    def get_pending_count(self, group_key: str | None = None) -> int:
        """Get count of pending requests.

        Args:
            group_key: Group to count (all if None)

        Returns:
            Number of pending requests
        """
        if group_key:
            return len(self._pending.get(group_key, []))
        return sum(len(p) for p in self._pending.values())

    @property
    def config(self) -> BatchConfig:
        """Get batch configuration."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Check if collector is running."""
        return self._running
