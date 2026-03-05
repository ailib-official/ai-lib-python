"""HTTP 传输层：基于 httpx 的异步 HTTP 客户端，支持流式传输和代理。

HTTP transport using httpx for async requests.

Provides:
- Async streaming support
- Configurable timeouts
- Proxy support
- Automatic header management
"""

from __future__ import annotations

import importlib.util
import os
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any

import httpx

from ai_lib_python.errors import RemoteError, TransportError
from ai_lib_python.transport.auth import get_auth_header

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ai_lib_python.protocol.manifest import ProtocolManifest


# Default timeouts
_DEFAULT_TIMEOUT = 30.0
_DEFAULT_CONNECT_TIMEOUT = 10.0


_UA_VERSION: str | None = None


def _http2_enabled() -> bool:
    """Enable HTTP/2 only when optional dependency is present."""
    return importlib.util.find_spec("h2") is not None


def _trust_env_enabled() -> bool:
    """Use env proxy settings only when explicitly enabled."""
    return os.getenv("AI_HTTP_TRUST_ENV", "0") == "1"


def _get_ua_version() -> str:
    """Get package version for User-Agent (cached)."""
    global _UA_VERSION
    if _UA_VERSION is None:
        try:
            from importlib.metadata import version

            _UA_VERSION = version("ai-lib-python")
        except Exception:
            _UA_VERSION = "0.5.0"
    return _UA_VERSION


class HttpTransport:
    """HTTP transport for API communication.

    Uses httpx for async HTTP requests with streaming support.

    Example:
        >>> transport = HttpTransport(manifest, model_id="gpt-4")
        >>> async with transport.stream_post("/chat/completions", payload) as response:
        ...     async for chunk in response.aiter_bytes():
        ...         process(chunk)
    """

    def __init__(
        self,
        manifest: ProtocolManifest,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        base_url_override: str | None = None,
        timeout: float | None = None,
        proxy: str | None = None,
    ) -> None:
        """Initialize HTTP transport.

        Args:
            manifest: Protocol manifest with endpoint configuration
            model_id: Model identifier (for logging/headers)
            api_key: Explicit API key (overrides env/keyring)
            base_url_override: Override base URL from manifest
            timeout: Request timeout in seconds
            proxy: Proxy URL
        """
        self._manifest = manifest
        self._model_id = model_id
        self._api_key = api_key

        # Resolve base URL
        self._base_url = base_url_override or manifest.endpoint.base_url

        # Resolve timeout
        self._timeout = timeout
        if self._timeout is None:
            # Check environment variable
            env_timeout = os.getenv("AI_HTTP_TIMEOUT_SECS") or os.getenv("AI_TIMEOUT_SECS")
            if env_timeout:
                with suppress(ValueError):
                    self._timeout = float(env_timeout)
        if self._timeout is None:
            # Use manifest timeout (convert from ms to s)
            self._timeout = manifest.endpoint.timeout_ms / 1000.0
        if self._timeout is None:
            self._timeout = _DEFAULT_TIMEOUT

        # Resolve proxy: default to direct connection unless trust_env is enabled.
        proxy_value: str | None
        if proxy is not None:
            proxy_value = proxy
        elif _trust_env_enabled():
            proxy_value = os.getenv("AI_PROXY_URL")
        else:
            proxy_value = None
        self._proxy = proxy_value

        # Build auth headers
        self._auth_headers = get_auth_header(
            manifest.id, manifest, self._api_key
        )

        # Client instance (lazy initialization)
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            timeout = httpx.Timeout(
                self._timeout,
                connect=_DEFAULT_CONNECT_TIMEOUT,
            )

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=timeout,
                proxy=self._proxy,
                http2=_http2_enabled(),
                trust_env=_trust_env_enabled(),
            )

        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_headers(self, extra_headers: dict[str, str] | None = None) -> dict[str, str]:
        """Build request headers.

        Args:
            extra_headers: Additional headers to include

        Returns:
            Complete headers dictionary
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"ai-lib-python/{_get_ua_version()}",
        }

        # Add auth headers
        headers.update(self._auth_headers)

        # Add extra headers
        if extra_headers:
            headers.update(extra_headers)

        return headers

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Make an HTTP request.

        Args:
            method: HTTP method
            path: Request path (relative to base URL)
            json: JSON body
            headers: Additional headers
            params: Query parameters

        Returns:
            HTTP response

        Raises:
            TransportError: On network/connection errors
            RemoteError: On API errors (4xx, 5xx)
        """
        client = self._get_client()
        request_headers = self._build_headers(headers)

        try:
            response = await client.request(
                method=method,
                url=path,
                json=json,
                headers=request_headers,
                params=params,
            )
        except httpx.ConnectError as e:
            raise TransportError(
                f"Connection failed: {e}",
                url=f"{self._base_url}{path}",
                cause=e,
            ) from e
        except httpx.TimeoutException as e:
            raise TransportError(
                f"Request timed out: {e}",
                url=f"{self._base_url}{path}",
                cause=e,
            ) from e
        except httpx.HTTPError as e:
            raise TransportError(
                f"HTTP error: {e}",
                url=f"{self._base_url}{path}",
                cause=e,
            ) from e

        # Check for error responses
        if response.status_code >= 400:
            body = None
            with suppress(Exception):
                body = response.json()

            raise RemoteError.from_response(
                status_code=response.status_code,
                body=body,
                headers=dict(response.headers),
                provider_classification=(
                    self._manifest.error_classification.model_dump()
                    if self._manifest.error_classification
                    else None
                ),
            )

        return response

    async def post(
        self,
        path: str,
        json: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Make a POST request.

        Args:
            path: Request path
            json: JSON body
            headers: Additional headers

        Returns:
            HTTP response
        """
        return await self.request("POST", path, json=json, headers=headers)

    async def get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Make a GET request.

        Args:
            path: Request path
            params: Query parameters
            headers: Additional headers

        Returns:
            HTTP response
        """
        return await self.request("GET", path, params=params, headers=headers)

    @asynccontextmanager
    async def stream_request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[httpx.Response]:
        """Make a streaming HTTP request.

        Args:
            method: HTTP method
            path: Request path
            json: JSON body
            headers: Additional headers

        Yields:
            HTTP response for streaming

        Example:
            >>> async with transport.stream_request("POST", "/chat", json=payload) as resp:
            ...     async for chunk in resp.aiter_bytes():
            ...         process(chunk)
        """
        client = self._get_client()
        request_headers = self._build_headers(headers)

        # For streaming, we need text/event-stream accept header
        request_headers["Accept"] = "text/event-stream"

        try:
            async with client.stream(
                method=method,
                url=path,
                json=json,
                headers=request_headers,
            ) as response:
                # Check for error responses
                if response.status_code >= 400:
                    # Need to read the body for error parsing
                    body_text = await response.aread()
                    body = None
                    try:
                        import json as json_module
                        body = json_module.loads(body_text)
                    except Exception:
                        pass

                    raise RemoteError.from_response(
                        status_code=response.status_code,
                        body=body,
                        headers=dict(response.headers),
                        provider_classification=(
                            self._manifest.error_classification.model_dump()
                            if self._manifest.error_classification
                            else None
                        ),
                    )

                yield response

        except httpx.ConnectError as e:
            raise TransportError(
                f"Connection failed: {e}",
                url=f"{self._base_url}{path}",
                cause=e,
            ) from e
        except httpx.TimeoutException as e:
            raise TransportError(
                f"Request timed out: {e}",
                url=f"{self._base_url}{path}",
                cause=e,
            ) from e
        except httpx.HTTPError as e:
            raise TransportError(
                f"HTTP error: {e}",
                url=f"{self._base_url}{path}",
                cause=e,
            ) from e

    @asynccontextmanager
    async def stream_post(
        self,
        path: str,
        json: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[httpx.Response]:
        """Make a streaming POST request.

        Args:
            path: Request path
            json: JSON body
            headers: Additional headers

        Yields:
            HTTP response for streaming
        """
        async with self.stream_request("POST", path, json=json, headers=headers) as resp:
            yield resp

    async def __aenter__(self) -> HttpTransport:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
