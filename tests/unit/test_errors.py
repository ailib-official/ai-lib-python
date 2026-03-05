"""Tests for error module."""

from ai_lib_python.errors import (
    AiLibError,
    ErrorClass,
    ErrorContext,
    ProtocolError,
    RemoteError,
    TransportError,
    classify_http_error,
    is_fallbackable,
    is_retryable,
)


class TestErrorContext:
    """Tests for ErrorContext."""

    def test_empty_context(self) -> None:
        """Test empty context string representation."""
        ctx = ErrorContext()
        assert str(ctx) == ""

    def test_context_with_source(self) -> None:
        """Test context with source."""
        ctx = ErrorContext(source="protocol")
        assert "[protocol]" in str(ctx)

    def test_context_with_field_path(self) -> None:
        """Test context with field path."""
        ctx = ErrorContext(field_path="messages[0].content")
        assert "at 'messages[0].content'" in str(ctx)

    def test_context_with_hint(self) -> None:
        """Test context with hint."""
        ctx = ErrorContext(hint="Check your API key")
        assert "(hint: Check your API key)" in str(ctx)


class TestAiLibError:
    """Tests for base error class."""

    def test_basic_error(self) -> None:
        """Test basic error creation."""
        error = AiLibError("Something went wrong")
        assert error.message == "Something went wrong"
        assert str(error) == "Something went wrong"

    def test_error_with_context(self) -> None:
        """Test error with context."""
        ctx = ErrorContext(source="test", hint="Try again")
        error = AiLibError("Failed", ctx)
        assert "[test]" in str(error)
        assert "(hint: Try again)" in str(error)

    def test_with_hint(self) -> None:
        """Test adding hint to error."""
        error = AiLibError("Failed").with_hint("Check the config")
        assert error.context.hint == "Check the config"


class TestProtocolError:
    """Tests for ProtocolError."""

    def test_protocol_error(self) -> None:
        """Test protocol error creation."""
        error = ProtocolError(
            "Invalid manifest",
            protocol_path="/path/to/manifest.yaml",
            version="1.0",
        )
        assert error.protocol_path == "/path/to/manifest.yaml"
        assert error.version == "1.0"
        assert "protocol_path" in error.context.details


class TestTransportError:
    """Tests for TransportError."""

    def test_transport_error(self) -> None:
        """Test transport error creation."""
        error = TransportError(
            "Connection failed",
            url="https://api.example.com",
            status_code=500,
        )
        assert error.url == "https://api.example.com"
        assert error.status_code == 500


class TestRemoteError:
    """Tests for RemoteError."""

    def test_remote_error(self) -> None:
        """Test remote error creation."""
        error = RemoteError(
            message="Rate limited",
            status_code=429,
            error_class=ErrorClass.RATE_LIMITED,
            retryable=True,
            fallbackable=True,
            retry_after=60.0,
        )
        assert error.status_code == 429
        assert error.error_class == ErrorClass.RATE_LIMITED
        assert error.retryable is True
        assert error.fallbackable is True
        assert error.retry_after == 60.0

    def test_from_response_rate_limited(self) -> None:
        """Test creating error from rate limited response."""
        error = RemoteError.from_response(
            status_code=429,
            body={"error": {"message": "Too many requests"}},
            headers={"retry-after": "30"},
        )
        assert error.status_code == 429
        assert error.error_class == ErrorClass.RATE_LIMITED
        assert error.retryable is True
        assert error.retry_after == 30.0

    def test_from_response_quota_exhausted(self) -> None:
        """Test detecting quota exhausted from 429."""
        error = RemoteError.from_response(
            status_code=429,
            body={"error": {"message": "You have exceeded your quota limit"}},
        )
        assert error.error_class == ErrorClass.QUOTA_EXHAUSTED
        assert error.retryable is False

    def test_from_response_auth_error(self) -> None:
        """Test creating error from auth failure."""
        error = RemoteError.from_response(
            status_code=401,
            body={"error": {"message": "Invalid API key"}},
        )
        assert error.error_class == ErrorClass.AUTHENTICATION
        assert error.retryable is False


class TestErrorClassification:
    """Tests for error classification functions."""

    def test_classify_400(self) -> None:
        """Test classifying 400 error."""
        assert classify_http_error(400) == ErrorClass.INVALID_REQUEST

    def test_classify_401(self) -> None:
        """Test classifying 401 error."""
        assert classify_http_error(401) == ErrorClass.AUTHENTICATION

    def test_classify_403(self) -> None:
        """Test classifying 403 error."""
        assert classify_http_error(403) == ErrorClass.PERMISSION_DENIED

    def test_classify_404(self) -> None:
        """Test classifying 404 error."""
        assert classify_http_error(404) == ErrorClass.NOT_FOUND

    def test_classify_429(self) -> None:
        """Test classifying 429 error."""
        assert classify_http_error(429) == ErrorClass.RATE_LIMITED

    def test_classify_500(self) -> None:
        """Test classifying 500 error."""
        assert classify_http_error(500) == ErrorClass.SERVER_ERROR

    def test_classify_503(self) -> None:
        """Test classifying 503 error."""
        assert classify_http_error(503) == ErrorClass.OVERLOADED

    def test_classify_with_provider_config(self) -> None:
        """Test classifying with provider-specific config."""
        config = {"by_http_status": {"418": "other"}}
        assert classify_http_error(418, None, config) == ErrorClass.OTHER

    def test_is_retryable(self) -> None:
        """Test retryable check."""
        assert is_retryable(ErrorClass.RATE_LIMITED) is True
        assert is_retryable(ErrorClass.SERVER_ERROR) is True
        assert is_retryable(ErrorClass.TIMEOUT) is True
        assert is_retryable(ErrorClass.AUTHENTICATION) is False
        assert is_retryable(ErrorClass.INVALID_REQUEST) is False

    def test_is_fallbackable(self) -> None:
        """Test fallbackable check."""
        assert is_fallbackable(ErrorClass.RATE_LIMITED) is True
        assert is_fallbackable(ErrorClass.QUOTA_EXHAUSTED) is True
        # Authentication failures can fallback to alternate providers/keys.
        assert is_fallbackable(ErrorClass.AUTHENTICATION) is True
