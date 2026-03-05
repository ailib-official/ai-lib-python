"""
Cache manager for AI responses.

Provides unified cache management with multiple backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_lib_python.cache.backends import CacheBackend, MemoryCache, NullCache
from ai_lib_python.cache.key import CacheKey, CacheKeyGenerator


@dataclass
class CacheStats:
    """Cache statistics.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        sets: Number of cache sets
        evictions: Number of evictions
        total_requests: Total requests
    """

    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0

    @property
    def total_requests(self) -> int:
        """Get total number of requests."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate (0.0 to 1.0)."""
        total = self.total_requests
        if total == 0:
            return 0.0
        return self.hits / total

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "total_requests": self.total_requests,
            "hit_rate": self.hit_rate,
        }

    def reset(self) -> None:
        """Reset statistics."""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0


@dataclass
class CacheConfig:
    """Cache configuration.

    Attributes:
        enabled: Whether caching is enabled
        default_ttl: Default TTL in seconds
        max_size: Maximum cache size
        cache_streaming: Whether to cache streaming responses
        cache_errors: Whether to cache error responses
    """

    enabled: bool = True
    default_ttl: float = 3600.0  # 1 hour
    max_size: int = 1000
    cache_streaming: bool = False
    cache_errors: bool = False

    @classmethod
    def disabled(cls) -> CacheConfig:
        """Create disabled cache config."""
        return cls(enabled=False)

    @classmethod
    def short_ttl(cls, ttl: float = 300.0) -> CacheConfig:
        """Create config with short TTL (5 minutes default)."""
        return cls(default_ttl=ttl)

    @classmethod
    def long_ttl(cls, ttl: float = 86400.0) -> CacheConfig:
        """Create config with long TTL (24 hours default)."""
        return cls(default_ttl=ttl)


class CacheManager:
    """Manages caching for AI responses.

    Provides a high-level interface for caching with automatic
    key generation and statistics.

    Example:
        >>> manager = CacheManager(CacheConfig(default_ttl=3600))
        >>>
        >>> # Check cache before making request
        >>> cached = await manager.get_response(model, messages)
        >>> if cached:
        ...     return cached
        >>>
        >>> # Make request and cache result
        >>> response = await client.chat(model, messages)
        >>> await manager.cache_response(model, messages, response)
    """

    def __init__(
        self,
        config: CacheConfig | None = None,
        backend: CacheBackend | None = None,
    ) -> None:
        """Initialize cache manager.

        Args:
            config: Cache configuration
            backend: Cache backend (defaults to MemoryCache)
        """
        self._config = config or CacheConfig()
        self._key_generator = CacheKeyGenerator()
        self._stats = CacheStats()

        self._backend: CacheBackend
        if not self._config.enabled:
            self._backend = NullCache()
        else:
            self._backend = backend or MemoryCache(
                max_size=self._config.max_size,
                default_ttl=self._config.default_ttl,
            )

    async def get_response(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **params: Any,
    ) -> dict[str, Any] | None:
        """Get a cached response.

        Args:
            model: Model name
            messages: Chat messages
            **params: Additional parameters

        Returns:
            Cached response or None
        """
        if not self._config.enabled:
            self._stats.misses += 1
            return None

        key = self._key_generator.generate(model, messages, **params)
        value = await self._backend.get(key.key)

        if value is not None:
            self._stats.hits += 1
        else:
            self._stats.misses += 1

        return value

    async def cache_response(
        self,
        model: str,
        messages: list[dict[str, Any]],
        response: dict[str, Any],
        ttl: float | None = None,
        **params: Any,
    ) -> CacheKey:
        """Cache a response.

        Args:
            model: Model name
            messages: Chat messages
            response: Response to cache
            ttl: TTL override
            **params: Additional parameters

        Returns:
            Cache key used
        """
        key = self._key_generator.generate(model, messages, **params)

        if self._config.enabled:
            await self._backend.set(
                key.key,
                response,
                ttl=ttl or self._config.default_ttl,
            )
            self._stats.sets += 1

        return key

    async def get_embedding(
        self,
        model: str,
        input_text: str | list[str],
        dimensions: int | None = None,
    ) -> list[list[float]] | None:
        """Get cached embeddings.

        Args:
            model: Model name
            input_text: Input text(s)
            dimensions: Output dimensions

        Returns:
            Cached embeddings or None
        """
        if not self._config.enabled:
            self._stats.misses += 1
            return None

        key = self._key_generator.generate_for_embedding(model, input_text, dimensions)
        value = await self._backend.get(key.key)

        if value is not None:
            self._stats.hits += 1
        else:
            self._stats.misses += 1

        return value

    async def cache_embedding(
        self,
        model: str,
        input_text: str | list[str],
        embeddings: list[list[float]],
        dimensions: int | None = None,
        ttl: float | None = None,
    ) -> CacheKey:
        """Cache embeddings.

        Args:
            model: Model name
            input_text: Input text(s)
            embeddings: Embedding vectors
            dimensions: Output dimensions
            ttl: TTL override

        Returns:
            Cache key used
        """
        key = self._key_generator.generate_for_embedding(model, input_text, dimensions)

        if self._config.enabled:
            await self._backend.set(
                key.key,
                embeddings,
                ttl=ttl or self._config.default_ttl,
            )
            self._stats.sets += 1

        return key

    async def invalidate(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **params: Any,
    ) -> bool:
        """Invalidate a cached response.

        Args:
            model: Model name
            messages: Chat messages
            **params: Additional parameters

        Returns:
            True if invalidated, False if not found
        """
        key = self._key_generator.generate(model, messages, **params)
        return await self._backend.delete(key.key)

    async def clear(self) -> None:
        """Clear all cached entries."""
        await self._backend.clear()
        self._stats.reset()

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    @property
    def config(self) -> CacheConfig:
        """Get cache configuration."""
        return self._config

    @property
    def enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._config.enabled

    async def close(self) -> None:
        """Close the cache manager."""
        await self._backend.close()


# Global cache manager
_global_cache: CacheManager | None = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager.

    Returns:
        Global CacheManager instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache


def set_cache_manager(manager: CacheManager) -> None:
    """Set the global cache manager.

    Args:
        manager: CacheManager instance
    """
    global _global_cache
    _global_cache = manager
