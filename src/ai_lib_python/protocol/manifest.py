"""
Protocol manifest models based on AI-Protocol specification.

These Pydantic models represent the provider manifest structure
defined in the AI-Protocol v1.x specification.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AuthConfig(BaseModel):
    """Authentication configuration."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(default="bearer", description="Auth type: bearer, api_key, etc.")
    token_env: str | None = Field(default=None, description="Environment variable for API key")
    header_name: str | None = Field(default=None, description="Custom header name for API key")


class EndpointConfig(BaseModel):
    """Endpoint configuration."""

    model_config = ConfigDict(extra="allow")

    base_url: str = Field(description="Base URL for API requests")
    protocol: str = Field(default="https", description="Protocol: https, http, ws, wss")
    timeout_ms: int = Field(default=10000, description="Default timeout in milliseconds")


class HealthCheckConfig(BaseModel):
    """Health check configuration."""

    model_config = ConfigDict(extra="allow")

    method: str = Field(default="GET", description="HTTP method")
    path: str = Field(description="Health check path")
    expected_status: list[int] = Field(
        default_factory=lambda: [200], description="Expected HTTP status codes"
    )
    timeout_ms: int = Field(default=3000, description="Timeout in milliseconds")


class AvailabilityConfig(BaseModel):
    """Availability and health check configuration."""

    model_config = ConfigDict(extra="allow")

    required: bool = Field(
        default=False, description="Whether provider must be reachable at startup"
    )
    regions: list[str] = Field(default_factory=lambda: ["global"], description="Available regions")
    check: HealthCheckConfig | None = Field(default=None, description="Health check config")


class CapabilitiesConfig(BaseModel):
    """Provider capabilities."""

    model_config = ConfigDict(extra="allow")

    streaming: bool = Field(default=True, description="Supports streaming")
    tools: bool = Field(default=False, description="Supports tool/function calling")
    vision: bool = Field(default=False, description="Supports image/vision input")
    agentic: bool = Field(default=False, description="Supports agentic reasoning")
    reasoning: bool = Field(default=False, description="Supports thinking/reasoning blocks")
    parallel_tools: bool = Field(default=False, description="Supports parallel tool calls")


class DecoderConfig(BaseModel):
    """Stream decoder configuration."""

    model_config = ConfigDict(extra="allow")

    format: str = Field(default="sse", description="Stream format: sse, json_lines")
    delimiter: str = Field(default="\n\n", description="Frame delimiter")
    prefix: str = Field(default="data: ", description="Data line prefix")
    done_signal: str = Field(default="[DONE]", description="Stream end signal")
    strategy: str | None = Field(default=None, description="Provider-specific strategy")


class EventMapRule(BaseModel):
    """Event mapping rule."""

    model_config = ConfigDict(extra="allow")

    match: str = Field(description="JSONPath-like match expression")
    emit: str = Field(description="Event type to emit")
    fields: dict[str, str] = Field(default_factory=dict, description="Field extraction mappings")


class CandidateConfig(BaseModel):
    """Multi-candidate configuration."""

    model_config = ConfigDict(extra="allow")

    candidate_id_path: str | None = Field(default=None, description="JSONPath to candidate ID")
    fan_out: bool | None = Field(default=None, description="Whether to fan out candidates")


class AccumulatorConfig(BaseModel):
    """Accumulator configuration for stateful parsing."""

    model_config = ConfigDict(extra="allow")

    stateful_tool_parsing: bool = Field(default=False, description="Enable stateful tool parsing")
    key_path: str | None = Field(default=None, description="JSONPath for accumulation key")
    flush_on: str | None = Field(default=None, description="Condition to flush accumulator")


class StreamingConfig(BaseModel):
    """Streaming configuration."""

    model_config = ConfigDict(extra="allow")

    event_format: str | None = Field(default=None, description="Event format type")
    decoder: DecoderConfig | None = Field(default=None, description="Decoder configuration")
    frame_selector: str | None = Field(default=None, description="Frame selection expression")
    candidate: CandidateConfig | None = Field(default=None, description="Candidate config")
    accumulator: AccumulatorConfig | None = Field(default=None, description="Accumulator config")
    event_map: list[EventMapRule] = Field(default_factory=list, description="Event mapping rules")
    stop_condition: str | None = Field(default=None, description="Stop condition expression")
    extra_metadata_path: str | None = Field(default=None, description="Path to extra metadata")
    content_path: str | None = Field(default=None, description="Path to content field")
    tool_call_path: str | None = Field(default=None, description="Path to tool calls")
    usage_path: str | None = Field(default=None, description="Path to usage data")


class RetryPolicy(BaseModel):
    """Retry policy configuration."""

    model_config = ConfigDict(extra="allow")

    strategy: str = Field(default="exponential_backoff", description="Retry strategy")
    max_retries: int | None = Field(default=None, description="Maximum retry attempts")
    min_delay_ms: int = Field(default=1000, description="Minimum delay in ms")
    max_delay_ms: int | None = Field(default=None, description="Maximum delay in ms")
    jitter: str = Field(default="full", description="Jitter strategy: none, full, equal")
    retry_on_http_status: list[int] = Field(
        default_factory=lambda: [429, 500], description="HTTP status codes to retry"
    )
    retry_on_error_status: list[str] = Field(
        default_factory=list, description="Error status strings to retry"
    )
    notes: list[str] = Field(default_factory=list, description="Implementation notes")


class RateLimitHeaders(BaseModel):
    """Rate limit header configuration."""

    model_config = ConfigDict(extra="allow")

    requests_limit: str | None = Field(default=None, description="Requests limit header")
    requests_remaining: str | None = Field(default=None, description="Remaining requests header")
    requests_reset: str | None = Field(default=None, description="Requests reset time header")
    tokens_limit: str | None = Field(default=None, description="Tokens limit header")
    tokens_remaining: str | None = Field(default=None, description="Remaining tokens header")
    tokens_reset: str | None = Field(default=None, description="Tokens reset time header")
    retry_after: str | None = Field(default=None, description="Retry-After header name")


class ErrorClassification(BaseModel):
    """Error classification configuration."""

    model_config = ConfigDict(extra="allow")

    by_http_status: dict[str, str] = Field(
        default_factory=dict, description="Status code to error class mapping"
    )
    notes: list[str] = Field(default_factory=list, description="Classification notes")


class ToolUseConfig(BaseModel):
    """Tool use field extraction configuration."""

    model_config = ConfigDict(extra="allow")

    id_path: str | None = Field(default=None, description="Path to tool call ID")
    name_path: str | None = Field(default=None, description="Path to tool name")
    input_path: str | None = Field(default=None, description="Path to tool input")
    input_format: str = Field(
        default="json_object", description="Input format: json_object, json_string"
    )


class ToolingConfig(BaseModel):
    """Tooling/function calling configuration."""

    model_config = ConfigDict(extra="allow")

    source_model: str | None = Field(default=None, description="Source model for tool format")
    tool_use: ToolUseConfig | None = Field(default=None, description="Tool use extraction")
    notes: list[str] = Field(default_factory=list, description="Tooling notes")


class ServiceConfig(BaseModel):
    """Service endpoint configuration."""

    model_config = ConfigDict(extra="allow")

    path: str = Field(description="Service endpoint path")
    method: str = Field(default="GET", description="HTTP method")
    response_binding: str | None = Field(default=None, description="Response data binding path")


class TerminationConfig(BaseModel):
    """Termination reason mapping configuration."""

    model_config = ConfigDict(extra="allow")

    source_field: str = Field(
        default="finish_reason", description="Source field for termination reason"
    )
    mapping: dict[str, str] = Field(
        default_factory=dict, description="Provider to standard mapping"
    )
    notes: list[str] = Field(default_factory=list, description="Mapping notes")


class ResponsePathsConfig(BaseModel):
    """Response field paths configuration."""

    model_config = ConfigDict(extra="allow")

    content: str | None = Field(default=None, description="Path to content")
    tool_calls: str | None = Field(default=None, description="Path to tool calls")
    usage: str | None = Field(default=None, description="Path to usage")
    finish_reason: str | None = Field(default=None, description="Path to finish reason")
    reasoning_content: str | None = Field(
        default=None, description="Path to reasoning / thinking text (non-streaming)"
    )
    reasoning: str | None = Field(
        default=None, description="Alternate path for reasoning content"
    )


class ProtocolManifest(BaseModel):
    """Complete protocol manifest for a provider.

    This is the typed representation of a provider's YAML/JSON manifest file.
    It contains all configuration needed to interact with the provider's API.

    Example:
        >>> manifest = ProtocolManifest.model_validate(yaml_data)
        >>> print(manifest.id)  # "openai"
        >>> print(manifest.endpoint.base_url)  # "https://api.openai.com/v1"
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # Identity
    id: str = Field(description="Provider identifier")
    protocol_version: str = Field(default="1.0", description="Protocol version")
    name: str | None = Field(default=None, description="Human-readable provider name")
    version: str | None = Field(default=None, description="Provider version")
    status: str = Field(default="stable", description="Provider status")
    category: str | None = Field(default=None, description="Provider category")
    official_url: str | None = Field(default=None, description="Official documentation URL")
    support_contact: str | None = Field(default=None, description="Support contact URL")

    # Configuration
    endpoint: EndpointConfig = Field(description="Endpoint configuration")
    auth: AuthConfig | None = Field(default=None, description="Authentication config")
    payload_format: str | None = Field(default=None, description="Payload format style")

    # API configuration
    api_families: list[str] = Field(default_factory=list, description="Supported API families")
    default_api_family: str | None = Field(default=None, description="Default API family")
    endpoints: dict[str, Any] = Field(default_factory=dict, description="Named endpoints")
    services: dict[str, ServiceConfig] = Field(
        default_factory=dict, description="Service endpoints"
    )

    # Parameter mappings
    parameter_mappings: dict[str, str] = Field(
        default_factory=dict, description="Standard to provider parameter mapping"
    )

    # Response configuration
    response_format: str | None = Field(default=None, description="Response format style")
    response_paths: ResponsePathsConfig | None = Field(
        default=None, description="Response field paths"
    )

    # Streaming
    streaming: StreamingConfig | None = Field(default=None, description="Streaming configuration")

    # Capabilities
    capabilities: CapabilitiesConfig = Field(
        default_factory=CapabilitiesConfig, description="Provider capabilities"
    )

    # Availability
    availability: AvailabilityConfig | None = Field(
        default=None, description="Availability configuration"
    )

    # Resilience
    retry_policy: RetryPolicy | None = Field(default=None, description="Retry policy")
    rate_limit_headers: RateLimitHeaders | None = Field(
        default=None, description="Rate limit headers"
    )
    error_classification: ErrorClassification | None = Field(
        default=None, description="Error classification"
    )

    # Tooling
    tooling: ToolingConfig | None = Field(default=None, description="Tooling configuration")
    termination: TerminationConfig | None = Field(
        default=None, description="Termination reason mapping"
    )

    # Features
    features: dict[str, Any] = Field(default_factory=dict, description="Additional features")
    experimental_features: list[str] = Field(
        default_factory=list, description="Experimental features"
    )
    capability_profile: dict[str, Any] | None = Field(
        default=None, description="Structured capability profile (IOS/IOSPC phase)"
    )

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        return self.capabilities.streaming

    def supports_tools(self) -> bool:
        """Check if provider supports tool calling."""
        return self.capabilities.tools

    def supports_vision(self) -> bool:
        """Check if provider supports vision/image input."""
        return self.capabilities.vision

    def get_chat_endpoint(self) -> str:
        """Get the chat completions endpoint path."""
        chat_config = self.endpoints.get("chat")
        if isinstance(chat_config, dict):
            path = chat_config.get("path")
            if isinstance(path, str):
                return path
        return "/chat/completions"

    def get_service_endpoint(self, service_name: str) -> ServiceConfig | None:
        """Get a service endpoint configuration."""
        return self.services.get(service_name)

    def get_parameter_name(self, standard_name: str) -> str:
        """Map standard parameter name to provider-specific name."""
        return self.parameter_mappings.get(standard_name, standard_name)

    def get_error_class(self, status_code: int) -> str | None:
        """Get error class for HTTP status code."""
        if self.error_classification:
            return self.error_classification.by_http_status.get(str(status_code))
        return None
