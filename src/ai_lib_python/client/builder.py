"""
Builder classes for fluent API construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Policy-layer resilience (contact) must not be statically imported here (PT-067 / Paper1 §3).
_RES_PKG = "ai_lib_python.resilience"


def _resilience_mod() -> Any:
    import importlib

    return importlib.import_module(_RES_PKG)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ai_lib_python.client.core import AiClient
    from ai_lib_python.client.response import CallStats, ChatResponse
    from ai_lib_python.types.events import StreamingEvent
    from ai_lib_python.types.message import Message
    from ai_lib_python.types.tool import ToolChoice, ToolDefinition


class AiClientBuilder:
    """Builder for creating AiClient instances with custom configuration.

    Example:
        >>> client = await (
        ...     AiClientBuilder()
        ...     .model("anthropic/claude-3-5-sonnet")
        ...     .with_fallbacks(["openai/gpt-4o"])
        ...     .protocol_path("./custom-protocols")
        ...     .with_retry(RetryConfig(max_retries=3))
        ...     .with_circuit_breaker(CircuitBreakerConfig())
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._model: str | None = None
        self._protocol_path: str | None = None
        self._fallbacks: list[str] = []
        self._api_key: str | None = None
        self._base_url_override: str | None = None
        self._timeout: float | None = None
        self._hot_reload: bool = False
        self._max_inflight: int | None = None
        # Resilience configuration (types live in ai_lib_python.resilience; loaded at build time).
        self._retry_config: Any | None = None
        self._rate_limit_config: Any | None = None
        self._circuit_breaker_config: Any | None = None
        self._resilient_config: Any | None = None
        self._api_keys: dict[str, str] = {}

    def model(self, model_id: str) -> AiClientBuilder:
        """Set the model to use.

        Args:
            model_id: Model identifier (e.g., "anthropic/claude-3-5-sonnet")

        Returns:
            Self for chaining
        """
        self._model = model_id
        return self

    def protocol_path(self, path: str) -> AiClientBuilder:
        """Set custom protocol directory path.

        Args:
            path: Path to protocol directory

        Returns:
            Self for chaining
        """
        self._protocol_path = path
        return self

    def with_fallbacks(self, fallbacks: list[str]) -> AiClientBuilder:
        """Set fallback models.

        Args:
            fallbacks: List of fallback model IDs

        Returns:
            Self for chaining
        """
        self._fallbacks = fallbacks
        return self

    def api_key(self, key: str) -> AiClientBuilder:
        """Set explicit API key.

        Args:
            key: API key

        Returns:
            Self for chaining
        """
        self._api_key = key
        return self

    def base_url(self, url: str) -> AiClientBuilder:
        """Override base URL.

        Args:
            url: Base URL for API requests

        Returns:
            Self for chaining
        """
        self._base_url_override = url
        return self

    def timeout(self, seconds: float) -> AiClientBuilder:
        """Set request timeout.

        Args:
            seconds: Timeout in seconds

        Returns:
            Self for chaining
        """
        self._timeout = seconds
        return self

    def hot_reload(self, enable: bool = True) -> AiClientBuilder:
        """Enable hot reload of protocol files.

        Args:
            enable: Whether to enable hot reload

        Returns:
            Self for chaining
        """
        self._hot_reload = enable
        return self

    def max_inflight(self, n: int) -> AiClientBuilder:
        """Set maximum concurrent requests.

        Args:
            n: Maximum number of concurrent requests

        Returns:
            Self for chaining
        """
        self._max_inflight = n
        return self

    def retry(self, max_attempts: int = 3, backoff: float = 1.0) -> AiClientBuilder:
        """Configure retry policy with simple parameters.

        Args:
            max_attempts: Maximum number of attempts (including initial)
            backoff: Backoff base in seconds

        Returns:
            Self for chaining
        """
        rm = _resilience_mod()
        RetryConfig = getattr(rm, "RetryConfig")

        self._retry_config = RetryConfig(
            max_retries=max_attempts - 1,
            min_delay_ms=int(backoff * 1000),
        )
        return self

    def api_key_for(self, model_id: str, key: str) -> AiClientBuilder:
        """Set API key for a specific fallback model.

        Args:
            model_id: Model identifier
            key: API key to use for this model

        Returns:
            Self for chaining
        """
        self._api_keys[model_id] = key
        return self

    def with_retry(self, config: Any) -> AiClientBuilder:
        """Configure retry policy.

        Args:
            config: Retry configuration

        Returns:
            Self for chaining
        """
        self._retry_config = config
        return self

    def with_rate_limit(self, config: Any) -> AiClientBuilder:
        """Configure rate limiting.

        Args:
            config: Rate limiter configuration

        Returns:
            Self for chaining
        """
        self._rate_limit_config = config
        return self

    def with_circuit_breaker(self, config: Any) -> AiClientBuilder:
        """Configure circuit breaker.

        Args:
            config: Circuit breaker configuration

        Returns:
            Self for chaining
        """
        self._circuit_breaker_config = config
        return self

    def with_resilience(self, config: Any) -> AiClientBuilder:
        """Configure all resilience patterns at once.

        Args:
            config: Combined resilience configuration

        Returns:
            Self for chaining
        """
        self._resilient_config = config
        return self

    def production_ready(self) -> AiClientBuilder:
        """Enable production-ready resilience defaults.

        Returns:
            Self for chaining
        """
        rm = _resilience_mod()
        ResilientConfig = getattr(rm, "ResilientConfig")

        self._resilient_config = ResilientConfig.production()
        return self

    async def build(self) -> AiClient:
        """Build the AiClient instance.

        Returns:
            Configured AiClient

        Raises:
            ValueError: If model is not set
        """
        if not self._model:
            raise ValueError("Model must be set before building")

        from ai_lib_python.client.core import AiClient

        # Build resilient config
        resilient_config = self._resilient_config
        if resilient_config is None and any(
            [
                self._retry_config,
                self._rate_limit_config,
                self._circuit_breaker_config,
                self._max_inflight,
            ]
        ):
            rm = _resilience_mod()
            BackpressureConfig = getattr(rm, "BackpressureConfig")
            ResilientConfig = getattr(rm, "ResilientConfig")

            resilient_config = ResilientConfig(
                retry=self._retry_config,
                rate_limit=self._rate_limit_config,
                circuit_breaker=self._circuit_breaker_config,
                backpressure=(
                    BackpressureConfig(max_concurrent=self._max_inflight)
                    if self._max_inflight
                    else None
                ),
            )

        return await AiClient._create(
            model=self._model,
            protocol_path=self._protocol_path,
            fallbacks=self._fallbacks,
            api_key=self._api_key,
            base_url_override=self._base_url_override,
            timeout=self._timeout,
            hot_reload=self._hot_reload,
            resilient_config=resilient_config,
            api_keys=self._api_keys,
        )


class ChatRequestBuilder:
    """Builder for chat completion requests.

    Provides a fluent API for configuring chat requests:

    Example:
        >>> response = await (
        ...     client.chat()
        ...     .messages([Message.user("Hello!")])
        ...     .temperature(0.7)
        ...     .max_tokens(1024)
        ...     .execute()
        ... )

        >>> async for event in client.chat().messages(msgs).stream():
        ...     print(event)
    """

    def __init__(self, client: AiClient) -> None:
        """Initialize the builder.

        Args:
            client: Parent AiClient instance
        """
        self._client = client
        self._messages: list[Message] = []
        self._temperature: float | None = None
        self._max_tokens: int | None = None
        self._top_p: float | None = None
        self._stop_sequences: list[str] | None = None
        self._tools: list[ToolDefinition] | None = None
        self._tool_choice: ToolChoice | str | None = None
        self._stream: bool = False
        self._extra_params: dict[str, Any] = {}

    def messages(self, messages: list[Message]) -> ChatRequestBuilder:
        """Set the messages for the request.

        Args:
            messages: List of messages

        Returns:
            Self for chaining
        """
        self._messages = messages
        return self

    def add_message(self, message: Message) -> ChatRequestBuilder:
        """Add a message to the request.

        Args:
            message: Message to add

        Returns:
            Self for chaining
        """
        self._messages.append(message)
        return self

    def system(self, content: str) -> ChatRequestBuilder:
        """Add a system message.

        Args:
            content: System message content

        Returns:
            Self for chaining
        """
        from ai_lib_python.types.message import Message

        self._messages.append(Message.system(content))
        return self

    def user(self, content: str) -> ChatRequestBuilder:
        """Add a user message.

        Args:
            content: User message content

        Returns:
            Self for chaining
        """
        from ai_lib_python.types.message import Message

        self._messages.append(Message.user(content))
        return self

    def temperature(self, value: float) -> ChatRequestBuilder:
        """Set the temperature.

        Args:
            value: Temperature (0.0 to 2.0)

        Returns:
            Self for chaining
        """
        self._temperature = value
        return self

    def max_tokens(self, value: int) -> ChatRequestBuilder:
        """Set maximum tokens to generate.

        Args:
            value: Maximum tokens

        Returns:
            Self for chaining
        """
        self._max_tokens = value
        return self

    def top_p(self, value: float) -> ChatRequestBuilder:
        """Set nucleus sampling parameter.

        Args:
            value: Top-p value (0.0 to 1.0)

        Returns:
            Self for chaining
        """
        self._top_p = value
        return self

    def stop(self, sequences: list[str]) -> ChatRequestBuilder:
        """Set stop sequences.

        Args:
            sequences: List of stop sequences

        Returns:
            Self for chaining
        """
        self._stop_sequences = sequences
        return self

    def tools(self, tools: list[ToolDefinition]) -> ChatRequestBuilder:
        """Set tools for function calling.

        Args:
            tools: List of tool definitions

        Returns:
            Self for chaining
        """
        self._tools = tools
        return self

    def tool_choice(self, choice: ToolChoice | str) -> ChatRequestBuilder:
        """Set tool choice policy.

        Args:
            choice: Tool choice (auto, none, required, or specific)

        Returns:
            Self for chaining
        """
        self._tool_choice = choice
        return self

    def param(self, key: str, value: Any) -> ChatRequestBuilder:
        """Set an extra parameter.

        Args:
            key: Parameter name
            value: Parameter value

        Returns:
            Self for chaining
        """
        self._extra_params[key] = value
        return self

    def build_payload(self) -> dict[str, Any]:
        """Build the request payload.

        Returns:
            Request payload dictionary
        """
        manifest = self._client._manifest

        # Build messages
        messages_payload = []
        for msg in self._messages:
            msg_dict: dict[str, Any] = {
                "role": msg.role.value if hasattr(msg.role, "value") else msg.role
            }
            if isinstance(msg.content, str):
                msg_dict["content"] = msg.content
            else:
                # Content blocks
                msg_dict["content"] = [block.model_dump(exclude_none=True) for block in msg.content]
            messages_payload.append(msg_dict)

        payload: dict[str, Any] = {
            manifest.get_parameter_name("messages") if manifest else "messages": messages_payload,
            "model": self._client._model_id,
        }

        # Add optional parameters
        if self._temperature is not None:
            key = manifest.get_parameter_name("temperature") if manifest else "temperature"
            payload[key] = self._temperature

        if self._max_tokens is not None:
            key = manifest.get_parameter_name("max_tokens") if manifest else "max_tokens"
            payload[key] = self._max_tokens

        if self._top_p is not None:
            key = manifest.get_parameter_name("top_p") if manifest else "top_p"
            payload[key] = self._top_p

        if self._stop_sequences:
            key = manifest.get_parameter_name("stop_sequences") if manifest else "stop"
            payload[key] = self._stop_sequences

        if self._tools:
            key = manifest.get_parameter_name("tools") if manifest else "tools"
            payload[key] = [t.model_dump(exclude_none=True) for t in self._tools]

        if self._tool_choice:
            key = manifest.get_parameter_name("tool_choice") if manifest else "tool_choice"
            if hasattr(self._tool_choice, "value"):
                payload[key] = self._tool_choice.value
            else:
                payload[key] = self._tool_choice

        if self._stream:
            key = manifest.get_parameter_name("stream") if manifest else "stream"
            payload[key] = True

        # Add extra parameters
        payload.update(self._extra_params)

        return payload

    async def execute(self) -> ChatResponse:
        """Execute the chat request (non-streaming).

        Returns:
            ChatResponse with the completion
        """
        self._stream = False
        return await self._client._execute_chat(self)

    async def execute_with_stats(self) -> tuple[ChatResponse, CallStats]:
        """Execute the chat request and return stats.

        Returns:
            Tuple of (ChatResponse, CallStats)
        """
        self._stream = False
        return await self._client._execute_chat_with_stats(self)

    async def stream(self) -> AsyncIterator[StreamingEvent]:
        """Execute the chat request with streaming.

        Yields:
            Streaming events as they arrive
        """
        self._stream = True
        async for event in self._client._execute_stream(self):
            yield event

    async def stream_with_stats(self) -> tuple[AsyncIterator[StreamingEvent], CallStats]:
        """Execute streaming request and return stats object.

        Returns:
            Tuple of (event iterator, CallStats)
        """
        self._stream = True
        return await self._client._execute_stream_with_stats(self)
