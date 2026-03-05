"""
Connection pool for HTTP transport.

Provides connection pooling and management for efficient HTTP client reuse.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class PoolConfig:
    """Configuration for connection pool.

    Attributes:
        max_connections: Maximum total connections
        max_keepalive_connections: Maximum idle connections to keep
        keepalive_expiry: Seconds before idle connection expires
        connect_timeout: Connection timeout in seconds
        read_timeout: Read timeout in seconds
        write_timeout: Write timeout in seconds
        pool_timeout: Timeout waiting for available connection
    """

    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry: float = 30.0
    connect_timeout: float = 10.0
    read_timeout: float = 60.0
    write_timeout: float = 60.0
    pool_timeout: float = 30.0

    @classmethod
    def default(cls) -> PoolConfig:
        """Create default configuration."""
        return cls()

    @classmethod
    def high_throughput(cls) -> PoolConfig:
        """Create configuration optimized for high throughput."""
        return cls(
            max_connections=200,
            max_keepalive_connections=50,
            keepalive_expiry=60.0,
            pool_timeout=60.0,
        )

    @classmethod
    def low_latency(cls) -> PoolConfig:
        """Create configuration optimized for low latency."""
        return cls(
            max_connections=50,
            max_keepalive_connections=30,
            keepalive_expiry=120.0,
            connect_timeout=5.0,
        )

    def to_httpx_limits(self) -> httpx.Limits:
        """Convert to httpx Limits."""
        return httpx.Limits(
            max_connections=self.max_connections,
            max_keepalive_connections=self.max_keepalive_connections,
            keepalive_expiry=self.keepalive_expiry,
        )

    def to_httpx_timeout(self) -> httpx.Timeout:
        """Convert to httpx Timeout."""
        return httpx.Timeout(
            connect=self.connect_timeout,
            read=self.read_timeout,
            write=self.write_timeout,
            pool=self.pool_timeout,
        )


@dataclass
class PoolStats:
    """Statistics for connection pool.

    Attributes:
        total_connections_created: Total connections created
        total_connections_closed: Total connections closed
        active_connections: Current active connections
        idle_connections: Current idle connections
        requests_total: Total requests made
        requests_successful: Successful requests
        requests_failed: Failed requests
        avg_connection_time_ms: Average connection time
    """

    total_connections_created: int = 0
    total_connections_closed: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    requests_total: int = 0
    requests_successful: int = 0
    requests_failed: int = 0
    connection_times: list[float] = field(default_factory=list)

    @property
    def avg_connection_time_ms(self) -> float:
        """Get average connection time in milliseconds."""
        if not self.connection_times:
            return 0.0
        return sum(self.connection_times) / len(self.connection_times) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_connections_created": self.total_connections_created,
            "total_connections_closed": self.total_connections_closed,
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
            "requests_total": self.requests_total,
            "requests_successful": self.requests_successful,
            "requests_failed": self.requests_failed,
            "avg_connection_time_ms": self.avg_connection_time_ms,
        }


class ConnectionPool:
    """HTTP connection pool with provider isolation.

    Manages httpx.AsyncClient instances per provider for efficient
    connection reuse.

    Example:
        >>> pool = ConnectionPool(PoolConfig.default())
        >>> client = await pool.get_client("openai", "https://api.openai.com")
        >>> response = await client.get("/v1/models")
        >>> await pool.close()
    """

    def __init__(self, config: PoolConfig | None = None) -> None:
        """Initialize connection pool.

        Args:
            config: Pool configuration
        """
        self._config = config or PoolConfig.default()
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._stats: dict[str, PoolStats] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def get_client(
        self,
        provider: str,
        base_url: str,
        headers: dict[str, str] | None = None,
    ) -> httpx.AsyncClient:
        """Get or create a client for a provider.

        Args:
            provider: Provider identifier
            base_url: Base URL for the provider
            headers: Default headers

        Returns:
            httpx.AsyncClient instance
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")

        async with self._lock:
            key = f"{provider}:{base_url}"

            if key not in self._clients:
                start_time = time.time()

                # Check if h2 is available for HTTP/2
                try:
                    import h2  # noqa: F401

                    use_http2 = True
                except ImportError:
                    use_http2 = False

                client = httpx.AsyncClient(
                    base_url=base_url,
                    headers=headers or {},
                    limits=self._config.to_httpx_limits(),
                    timeout=self._config.to_httpx_timeout(),
                    http2=use_http2,
                    trust_env=False,
                )

                self._clients[key] = client

                # Track stats
                if provider not in self._stats:
                    self._stats[provider] = PoolStats()

                self._stats[provider].total_connections_created += 1
                self._stats[provider].connection_times.append(
                    time.time() - start_time
                )
                # Keep only last 100 samples
                if len(self._stats[provider].connection_times) > 100:
                    self._stats[provider].connection_times = (
                        self._stats[provider].connection_times[-100:]
                    )

            return self._clients[key]

    async def close_client(self, provider: str, base_url: str) -> None:
        """Close a specific client.

        Args:
            provider: Provider identifier
            base_url: Base URL for the provider
        """
        async with self._lock:
            key = f"{provider}:{base_url}"
            if key in self._clients:
                await self._clients[key].aclose()
                del self._clients[key]

                if provider in self._stats:
                    self._stats[provider].total_connections_closed += 1

    async def close(self) -> None:
        """Close all connections."""
        self._closed = True
        async with self._lock:
            for client in self._clients.values():
                await client.aclose()
            self._clients.clear()

    def record_request(
        self,
        provider: str,
        success: bool = True,
    ) -> None:
        """Record a request for statistics.

        Args:
            provider: Provider identifier
            success: Whether request succeeded
        """
        if provider not in self._stats:
            self._stats[provider] = PoolStats()

        self._stats[provider].requests_total += 1
        if success:
            self._stats[provider].requests_successful += 1
        else:
            self._stats[provider].requests_failed += 1

    def get_stats(self, provider: str | None = None) -> dict[str, Any]:
        """Get pool statistics.

        Args:
            provider: Provider to get stats for (all if None)

        Returns:
            Dict with statistics
        """
        if provider:
            stats = self._stats.get(provider, PoolStats())
            return {provider: stats.to_dict()}
        else:
            return {p: s.to_dict() for p, s in self._stats.items()}

    def get_active_providers(self) -> list[str]:
        """Get list of providers with active connections.

        Returns:
            List of provider identifiers
        """
        providers = set()
        for key in self._clients:
            provider = key.split(":")[0]
            providers.add(provider)
        return list(providers)

    @property
    def config(self) -> PoolConfig:
        """Get pool configuration."""
        return self._config

    @property
    def is_closed(self) -> bool:
        """Check if pool is closed."""
        return self._closed

    async def __aenter__(self) -> ConnectionPool:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self, exc_type: Any, exc_val: Any, exc_tb: Any
    ) -> None:
        """Async context manager exit."""
        await self.close()


class PooledTransport:
    """HTTP transport that uses connection pooling.

    Wraps httpx.AsyncClient with connection pooling and
    automatic provider isolation.

    Example:
        >>> transport = PooledTransport(pool)
        >>> response = await transport.request(
        ...     "openai",
        ...     "https://api.openai.com",
        ...     "POST",
        ...     "/v1/chat/completions",
        ...     json=payload,
        ... )
    """

    def __init__(self, pool: ConnectionPool) -> None:
        """Initialize pooled transport.

        Args:
            pool: Connection pool to use
        """
        self._pool = pool

    async def request(
        self,
        provider: str,
        base_url: str,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        content: bytes | None = None,
    ) -> httpx.Response:
        """Make an HTTP request.

        Args:
            provider: Provider identifier
            base_url: Base URL
            method: HTTP method
            path: Request path
            headers: Request headers
            json: JSON body
            params: Query parameters
            content: Raw content

        Returns:
            HTTP response
        """
        client = await self._pool.get_client(provider, base_url)

        try:
            response = await client.request(
                method,
                path,
                headers=headers,
                json=json,
                params=params,
                content=content,
            )
            self._pool.record_request(provider, success=True)
            return response

        except Exception:
            self._pool.record_request(provider, success=False)
            raise

    async def get(
        self,
        provider: str,
        base_url: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make a GET request."""
        return await self.request(provider, base_url, "GET", path, **kwargs)

    async def post(
        self,
        provider: str,
        base_url: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make a POST request."""
        return await self.request(provider, base_url, "POST", path, **kwargs)


# Global connection pool
_global_pool: ConnectionPool | None = None


def get_connection_pool() -> ConnectionPool:
    """Get the global connection pool.

    Returns:
        Global ConnectionPool instance
    """
    global _global_pool
    if _global_pool is None:
        _global_pool = ConnectionPool()
    return _global_pool


def set_connection_pool(pool: ConnectionPool) -> None:
    """Set the global connection pool.

    Args:
        pool: ConnectionPool instance
    """
    global _global_pool
    _global_pool = pool


async def close_global_pool() -> None:
    """Close the global connection pool."""
    global _global_pool
    if _global_pool is not None:
        await _global_pool.close()
        _global_pool = None
