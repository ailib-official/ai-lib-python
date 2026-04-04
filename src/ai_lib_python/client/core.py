"""核心客户端实现：提供协议驱动的统一 AI 模型交互接口。

Core AiClient implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ai_lib_python.client.builder import AiClientBuilder, ChatRequestBuilder
from ai_lib_python.client.response import CallStats, ChatResponse
from ai_lib_python.pipeline import Pipeline
from ai_lib_python.protocol import ProtocolLoader
from ai_lib_python.transport import HttpTransport
from ai_lib_python.types.tool import ToolCall

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ai_lib_python.protocol.manifest import ProtocolManifest
    from ai_lib_python.types.events import StreamingEvent

# Contact-layer resilience is loaded only via importlib when users opt in (Paper1 §3.2).
_RES_PKG = "ai_lib_python.resilience"


def _load_resilience_module() -> Any:
    import importlib

    return importlib.import_module(_RES_PKG)


class AiClient:
    """Unified client for AI model interaction.

    AiClient is the main entry point for interacting with AI models.
    It provides a protocol-driven, provider-agnostic interface with
    built-in resilience patterns.

    Example:
        >>> client = await AiClient.create("anthropic/claude-3-5-sonnet")
        >>> response = await client.chat().messages([Message.user("Hello!")]).execute()
        >>> print(response.content)

        >>> # With resilience
        >>> client = await (
        ...     AiClient.builder()
        ...     .model("openai/gpt-4o")
        ...     .production_ready()  # Enable all resilience patterns
        ...     .build()
        ... )

        >>> # Streaming
        >>> async for event in client.chat().messages(msgs).stream():
        ...     if event.is_content_delta:
        ...         print(event.as_content_delta.content, end="")
    """

    def __init__(
        self,
        manifest: ProtocolManifest,
        transport: HttpTransport,
        pipeline: Pipeline,
        model_id: str,
        fallbacks: list[str] | None = None,
        executor: Any | None = None,
        loader: ProtocolLoader | None = None,
        api_keys: dict[str, str] | None = None,
        base_url_override: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize the client (internal use).

        Use AiClient.create() or AiClientBuilder for public construction.
        """
        self._manifest = manifest
        self._transport = transport
        self._pipeline = pipeline
        self._model_id = model_id
        self._fallbacks = fallbacks or []
        self._executor = executor
        self._loader = loader
        self._api_keys = api_keys or {}
        self._base_url_override = base_url_override
        self._timeout = timeout

    @classmethod
    async def create(
        cls,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> AiClient:
        """Create a new AiClient instance.

        Args:
            model: Model identifier (e.g., "anthropic/claude-3-5-sonnet")
            api_key: Optional explicit API key
            base_url: Optional base URL override
            timeout: Optional timeout in seconds

        Returns:
            Configured AiClient instance

        Example:
            >>> client = await AiClient.create("openai/gpt-4o")
            >>> client = await AiClient.create(
            ...     "anthropic/claude-3-5-sonnet",
            ...     api_key="sk-...",
            ... )
        """
        return await cls._create(
            model=model,
            api_key=api_key,
            base_url_override=base_url,
            timeout=timeout,
        )

    @classmethod
    def builder(cls) -> AiClientBuilder:
        """Get a builder for advanced configuration.

        Returns:
            AiClientBuilder instance

        Example:
            >>> client = await (
            ...     AiClient.builder()
            ...     .model("anthropic/claude-3-5-sonnet")
            ...     .with_fallbacks(["openai/gpt-4o"])
            ...     .max_inflight(10)
            ...     .build()
            ... )
        """
        return AiClientBuilder()

    @classmethod
    async def _create(
        cls,
        model: str,
        protocol_path: str | None = None,
        fallbacks: list[str] | None = None,
        api_key: str | None = None,
        base_url_override: str | None = None,
        timeout: float | None = None,
        hot_reload: bool = False,
        resilient_config: Any | None = None,
        api_keys: dict[str, str] | None = None,
    ) -> AiClient:
        """Internal creation method.

        Args:
            model: Model identifier
            protocol_path: Custom protocol directory
            fallbacks: Fallback model IDs
            api_key: Explicit API key
            base_url_override: Base URL override
            timeout: Request timeout
            hot_reload: Enable hot reload
            resilient_config: Resilience configuration

        Returns:
            Configured AiClient
        """
        # Load protocol
        loader = ProtocolLoader(
            base_path=protocol_path,
            hot_reload=hot_reload,
        )
        manifest = await loader.load_model(model)

        # Extract model ID from model string
        parts = model.split("/")
        model_id = parts[1] if len(parts) >= 2 else model

        # Create transport
        transport = HttpTransport(
            manifest=manifest,
            model_id=model_id,
            api_key=api_key,
            base_url_override=base_url_override,
            timeout=timeout,
        )

        # Create pipeline
        pipeline = Pipeline.from_manifest(manifest)

        # Create executor if resilience is configured (policy-layer module; dynamic import).
        executor = None
        if resilient_config is not None:
            rm = _load_resilience_module()
            ResilientExecutor = getattr(rm, "ResilientExecutor")
            executor = ResilientExecutor(resilient_config, name=f"{manifest.id}/{model_id}")

        return cls(
            manifest=manifest,
            transport=transport,
            pipeline=pipeline,
            model_id=model,  # Keep the full model name including provider
            fallbacks=fallbacks,
            executor=executor,
            loader=loader,
            api_keys=api_keys,
            base_url_override=base_url_override,
            timeout=timeout,
        )

    def chat(self) -> ChatRequestBuilder:
        """Start building a chat request.

        Returns:
            ChatRequestBuilder for fluent configuration

        Example:
            >>> response = await (
            ...     client.chat()
            ...     .messages([Message.user("Hello!")])
            ...     .temperature(0.7)
            ...     .execute()
            ... )
        """
        return ChatRequestBuilder(self)

    async def _execute_chat(self, builder: ChatRequestBuilder) -> ChatResponse:
        """Execute a non-streaming chat request with fallback support.

        Args:
            builder: Configured request builder

        Returns:
            ChatResponse with the completion
        """
        models_to_try = [self._model_id, *self._fallbacks]
        last_error = None

        for model in models_to_try:
            try:
                # 1. Resolve manifest and transport for this model
                if model == self._model_id:
                    manifest = self._manifest
                    transport = self._transport
                    pipeline = self._pipeline
                else:
                    # Dynamic load for fallback
                    if not self._loader:
                        raise ValueError("ProtocolLoader missing for fallback")
                    manifest = await self._loader.load_model(model)

                    parts = model.split("/")
                    m_id = parts[1] if len(parts) >= 2 else model

                    # Resolve key for this model
                    m_key = self._api_keys.get(model)

                    from ai_lib_python.transport import HttpTransport

                    transport = HttpTransport(
                        manifest=manifest,
                        model_id=m_id,
                        api_key=m_key,
                        base_url_override=self._base_url_override,
                        timeout=self._timeout,
                    )
                    pipeline = Pipeline.from_manifest(manifest)

                async def do_request(
                    m: ProtocolManifest = manifest,
                    t: HttpTransport = transport,
                    p: Pipeline = pipeline,
                    mid: str = model,
                ) -> ChatResponse:
                    # Debug print for model being used
                    print(f"DEBUG: Executing request for model: {mid}, manifest ID: {m.id}")

                    # Update builder's client temporary context?
                    # Actually builder.build_payload() uses self._client._model_id
                    # This is tricky as builder is bound to the primary client.
                    # We need to temporarily override the client context in the builder.

                    # Create a temporary builder/payload
                    # For simplicity, we'll just manually build the payload here or
                    # temporarily swap self._model_id (hacky but it works for this pattern)
                    original_model_id = self._model_id
                    original_manifest = self._manifest
                    try:
                        self._model_id = mid
                        self._manifest = m
                        payload = builder.build_payload()
                        print(f"DEBUG: Payload model: {payload.get('model')}")
                    finally:
                        self._model_id = original_model_id
                        self._manifest = original_manifest

                    endpoint = m.get_chat_endpoint()
                    print(f"DEBUG: Endpoint: {endpoint}")
                    response = await t.post(endpoint, json=payload)
                    data = response.json()

                    # Parse using the correct pipeline
                    return self._parse_response(data)

                # Use executor if available for resilience
                if self._executor:
                    return cast(
                        ChatResponse,
                        await self._executor.execute(do_request),
                    )
                return await do_request()

            except Exception as e:
                from ai_lib_python.errors import is_fallbackable

                # Check if we should fallback
                error_class = getattr(e, "error_class", None)
                if model != models_to_try[-1] and (
                    error_class is None or is_fallbackable(error_class)
                ):
                    last_error = e
                    continue
                raise e

        raise last_error or RuntimeError("Fallback exhausted")

    async def _execute_chat_with_stats(
        self, builder: ChatRequestBuilder
    ) -> tuple[ChatResponse, CallStats]:
        """Execute chat request and return stats.

        Args:
            builder: Configured request builder

        Returns:
            Tuple of (ChatResponse, CallStats)
        """
        stats = CallStats(
            model=self._model_id,
            provider=self._manifest.id,
            endpoint=self._manifest.get_chat_endpoint(),
        )
        stats.record_start()

        try:
            response = await self._execute_chat(builder)
            stats.record_end()
            stats.record_usage(response.usage)
            return response, stats
        except Exception:
            stats.record_end()
            raise

    async def _execute_stream(self, builder: ChatRequestBuilder) -> AsyncIterator[StreamingEvent]:
        """Execute a streaming chat request.

        Args:
            builder: Configured request builder

        Yields:
            StreamingEvent objects
        """
        payload = builder.build_payload()
        endpoint = self._manifest.get_chat_endpoint()

        async with self._transport.stream_post(endpoint, json=payload) as response:
            # Create byte stream from response
            async def byte_stream() -> AsyncIterator[bytes]:
                async for chunk in response.aiter_bytes():
                    yield chunk

            # Process through pipeline
            async for event in self._pipeline.process(byte_stream()):
                yield event

    async def _execute_stream_with_stats(
        self, builder: ChatRequestBuilder
    ) -> tuple[AsyncIterator[StreamingEvent], CallStats]:
        """Execute streaming request and return stats.

        Args:
            builder: Configured request builder

        Returns:
            Tuple of (event iterator, CallStats)
        """
        stats = CallStats(
            model=self._model_id,
            provider=self._manifest.id,
            endpoint=self._manifest.get_chat_endpoint(),
        )
        stats.record_start()

        async def wrapped_stream() -> AsyncIterator[StreamingEvent]:
            first_event = True
            try:
                async for event in self._execute_stream(builder):
                    if first_event:
                        stats.record_first_token()
                        first_event = False

                    # Capture usage from metadata events
                    if event.is_metadata:
                        meta = event.as_metadata
                        if meta.usage:
                            stats.record_usage(meta.usage)

                    yield event
            finally:
                stats.record_end()

        return wrapped_stream(), stats

    def _parse_response(self, data: dict[str, Any]) -> ChatResponse:
        """Parse API response to ChatResponse.

        Args:
            data: Raw API response

        Returns:
            Parsed ChatResponse
        """
        response = ChatResponse(raw_response=data)

        # OpenAI-style response
        choices = data.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})

            response.content = message.get("content", "") or ""
            response.finish_reason = choice.get("finish_reason")

            # Parse tool calls
            tool_calls_data = message.get("tool_calls", [])
            for tc in tool_calls_data:
                response.tool_calls.append(
                    ToolCall.from_openai_format(
                        id=tc.get("id", ""),
                        function_name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", "{}"),
                    )
                )

        # Anthropic-style response
        elif "content" in data:
            content_blocks = data.get("content", [])
            text_parts = []
            for block in content_blocks:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    response.tool_calls.append(
                        ToolCall(
                            id=block.get("id", ""),
                            function_name=block.get("name", ""),
                            arguments=block.get("input", {}),
                        )
                    )

            response.content = "".join(text_parts)
            response.finish_reason = data.get("stop_reason")

        # Usage
        response.usage = data.get("usage")
        response.model = data.get("model")

        return response

    @property
    def model_id(self) -> str:
        """Get the model ID."""
        return self._model_id

    @property
    def provider_id(self) -> str:
        """Get the provider ID."""
        return self._manifest.id

    @property
    def manifest(self) -> ProtocolManifest:
        """Get the protocol manifest."""
        return self._manifest

    @property
    def is_resilient(self) -> bool:
        """Check if resilience is enabled."""
        return self._executor is not None

    @property
    def circuit_state(self) -> str:
        """Get current circuit breaker state."""
        if self._executor:
            return cast(str, self._executor.circuit_state)
        return "disabled"

    @property
    def current_inflight(self) -> int:
        """Get current number of in-flight requests."""
        if self._executor:
            return cast(int, self._executor.current_inflight)
        return 0

    def get_resilience_stats(self) -> dict[str, Any]:
        """Get resilience statistics.

        Returns:
            Dict with resilience stats, or empty dict if not resilient
        """
        if self._executor:
            return cast(dict[str, Any], self._executor.get_stats())
        return {}

    def reset_resilience(self) -> None:
        """Reset all resilience components to initial state."""
        if self._executor:
            self._executor.reset()

    async def close(self) -> None:
        """Close the client and release resources."""
        await self._transport.close()

    async def __aenter__(self) -> AiClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
