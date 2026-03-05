"""
Health check utilities for ai-lib-python.

Provides health status tracking and provider availability detection.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check.

    Attributes:
        name: Check name
        status: Health status
        message: Status message
        latency_ms: Check latency in milliseconds
        timestamp: Check timestamp
        details: Additional details
    """

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "details": self.details,
        }


@dataclass
class AggregatedHealth:
    """Aggregated health status.

    Attributes:
        status: Overall health status
        checks: Individual check results
        timestamp: Aggregation timestamp
    """

    status: HealthStatus
    checks: list[HealthCheckResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "checks": [c.to_dict() for c in self.checks],
            "timestamp": self.timestamp,
        }


class HealthChecker:
    """Health checker for ai-lib-python components.

    Example:
        >>> checker = HealthChecker()
        >>> checker.register("openai", check_openai_health)
        >>> health = await checker.check_all()
        >>> print(health.status)
    """

    def __init__(self) -> None:
        """Initialize health checker."""
        self._checks: dict[str, Callable[[], Awaitable[HealthCheckResult]]] = {}
        self._last_results: dict[str, HealthCheckResult] = {}

    def register(
        self,
        name: str,
        check: Callable[[], Awaitable[HealthCheckResult]],
    ) -> None:
        """Register a health check.

        Args:
            name: Check name
            check: Async function returning HealthCheckResult
        """
        self._checks[name] = check

    def unregister(self, name: str) -> None:
        """Unregister a health check.

        Args:
            name: Check name
        """
        self._checks.pop(name, None)
        self._last_results.pop(name, None)

    async def check(self, name: str) -> HealthCheckResult:
        """Run a specific health check.

        Args:
            name: Check name

        Returns:
            HealthCheckResult

        Raises:
            KeyError: If check not found
        """
        if name not in self._checks:
            raise KeyError(f"Health check not found: {name}")

        check_fn = self._checks[name]
        start = time.time()

        try:
            result = await check_fn()
            result.latency_ms = (time.time() - start) * 1000
        except Exception as e:
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.time() - start) * 1000,
            )

        self._last_results[name] = result
        return result

    async def check_all(self, timeout: float = 30.0) -> AggregatedHealth:
        """Run all health checks.

        Args:
            timeout: Timeout for all checks in seconds

        Returns:
            AggregatedHealth with all results
        """
        if not self._checks:
            return AggregatedHealth(status=HealthStatus.UNKNOWN)

        # Run all checks concurrently
        tasks = [self.check(name) for name in self._checks]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            # Return unhealthy if timeout
            return AggregatedHealth(
                status=HealthStatus.UNHEALTHY,
                checks=[
                    HealthCheckResult(
                        name="_timeout",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check timeout after {timeout}s",
                    )
                ],
            )

        # Process results
        check_results: list[HealthCheckResult] = []
        for result in results:
            if isinstance(result, BaseException):
                check_results.append(
                    HealthCheckResult(
                        name="_error",
                        status=HealthStatus.UNHEALTHY,
                        message=str(result),
                    )
                )
            else:
                check_results.append(result)

        # Determine overall status
        statuses = [r.status for r in check_results]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.UNKNOWN

        return AggregatedHealth(status=overall, checks=check_results)

    def get_last_result(self, name: str) -> HealthCheckResult | None:
        """Get last result for a check.

        Args:
            name: Check name

        Returns:
            Last result or None
        """
        return self._last_results.get(name)

    def get_all_last_results(self) -> dict[str, HealthCheckResult]:
        """Get all last results.

        Returns:
            Dict of name to result
        """
        return dict(self._last_results)


class ProviderHealthTracker:
    """Tracks health of AI providers based on request outcomes.

    Automatically tracks success/failure rates and determines
    provider health status.

    Example:
        >>> tracker = ProviderHealthTracker()
        >>> tracker.record_success("openai")
        >>> tracker.record_failure("openai", "rate_limited")
        >>> status = tracker.get_status("openai")
    """

    def __init__(
        self,
        window_size: int = 100,
        unhealthy_threshold: float = 0.5,
        degraded_threshold: float = 0.1,
    ) -> None:
        """Initialize tracker.

        Args:
            window_size: Number of requests to consider
            unhealthy_threshold: Error rate for unhealthy status
            degraded_threshold: Error rate for degraded status
        """
        self._window_size = window_size
        self._unhealthy_threshold = unhealthy_threshold
        self._degraded_threshold = degraded_threshold

        # Sliding window of outcomes (True = success, False = failure)
        self._outcomes: dict[str, list[bool]] = {}
        self._last_error: dict[str, str] = {}
        self._last_success_time: dict[str, float] = {}
        self._last_failure_time: dict[str, float] = {}

    def record_success(self, provider: str) -> None:
        """Record a successful request.

        Args:
            provider: Provider name
        """
        if provider not in self._outcomes:
            self._outcomes[provider] = []

        self._outcomes[provider].append(True)
        if len(self._outcomes[provider]) > self._window_size:
            self._outcomes[provider].pop(0)

        self._last_success_time[provider] = time.time()

    def record_failure(self, provider: str, error: str = "") -> None:
        """Record a failed request.

        Args:
            provider: Provider name
            error: Error description
        """
        if provider not in self._outcomes:
            self._outcomes[provider] = []

        self._outcomes[provider].append(False)
        if len(self._outcomes[provider]) > self._window_size:
            self._outcomes[provider].pop(0)

        self._last_error[provider] = error
        self._last_failure_time[provider] = time.time()

    def get_status(self, provider: str) -> HealthStatus:
        """Get health status for a provider.

        Args:
            provider: Provider name

        Returns:
            HealthStatus
        """
        if provider not in self._outcomes or not self._outcomes[provider]:
            return HealthStatus.UNKNOWN

        outcomes = self._outcomes[provider]
        error_rate = 1 - (sum(outcomes) / len(outcomes))

        if error_rate >= self._unhealthy_threshold:
            return HealthStatus.UNHEALTHY
        elif error_rate >= self._degraded_threshold:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def get_error_rate(self, provider: str) -> float:
        """Get error rate for a provider.

        Args:
            provider: Provider name

        Returns:
            Error rate (0.0 to 1.0)
        """
        if provider not in self._outcomes or not self._outcomes[provider]:
            return 0.0

        outcomes = self._outcomes[provider]
        return 1 - (sum(outcomes) / len(outcomes))

    def get_details(self, provider: str) -> dict[str, Any]:
        """Get detailed health information for a provider.

        Args:
            provider: Provider name

        Returns:
            Dict with health details
        """
        outcomes = self._outcomes.get(provider, [])
        return {
            "provider": provider,
            "status": self.get_status(provider).value,
            "error_rate": self.get_error_rate(provider),
            "sample_count": len(outcomes),
            "last_error": self._last_error.get(provider),
            "last_success_time": self._last_success_time.get(provider),
            "last_failure_time": self._last_failure_time.get(provider),
        }

    def get_all_providers(self) -> list[str]:
        """Get all tracked providers.

        Returns:
            List of provider names
        """
        return list(self._outcomes.keys())

    def reset(self, provider: str | None = None) -> None:
        """Reset health tracking.

        Args:
            provider: Provider to reset (all if None)
        """
        if provider:
            self._outcomes.pop(provider, None)
            self._last_error.pop(provider, None)
            self._last_success_time.pop(provider, None)
            self._last_failure_time.pop(provider, None)
        else:
            self._outcomes.clear()
            self._last_error.clear()
            self._last_success_time.clear()
            self._last_failure_time.clear()


# Global health checker and tracker
_global_health_checker: HealthChecker | None = None
_global_health_tracker: ProviderHealthTracker | None = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker.

    Returns:
        Global HealthChecker instance
    """
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = HealthChecker()
    return _global_health_checker


def get_health_tracker() -> ProviderHealthTracker:
    """Get the global provider health tracker.

    Returns:
        Global ProviderHealthTracker instance
    """
    global _global_health_tracker
    if _global_health_tracker is None:
        _global_health_tracker = ProviderHealthTracker()
    return _global_health_tracker
