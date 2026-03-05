"""
Structured logging for ai-lib-python.

Provides context-aware logging with sensitive data masking.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

# Context variable for request-scoped logging context
_log_context: ContextVar[dict[str, Any] | None] = ContextVar("log_context", default=None)


class LogLevel(str, Enum):
    """Log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def to_logging_level(self) -> int:
        """Convert to standard logging level."""
        level = getattr(logging, self.value, None)
        if isinstance(level, int):
            return level
        return logging.INFO


@dataclass
class LogContext:
    """Request-scoped logging context.

    Attributes:
        request_id: Unique request identifier
        trace_id: Distributed trace ID
        span_id: Current span ID
        provider: AI provider name
        model: Model name
        extra: Additional context fields
    """

    request_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    provider: str | None = None
    model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        result: dict[str, Any] = {}
        if self.request_id:
            result["request_id"] = self.request_id
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.provider:
            result["provider"] = self.provider
        if self.model:
            result["model"] = self.model
        result.update(self.extra)
        return result

    def with_extra(self, **kwargs: Any) -> LogContext:
        """Create new context with additional fields."""
        new_extra = {**self.extra, **kwargs}
        return LogContext(
            request_id=self.request_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
            provider=self.provider,
            model=self.model,
            extra=new_extra,
        )


def get_log_context() -> LogContext:
    """Get current logging context."""
    data = _log_context.get()
    return LogContext(**data) if data else LogContext()


def set_log_context(context: LogContext) -> None:
    """Set logging context for current async context."""
    _log_context.set(context.to_dict())


def clear_log_context() -> None:
    """Clear logging context."""
    _log_context.set(None)


class SensitiveDataMasker:
    """Masks sensitive data in log messages."""

    # Patterns for sensitive data
    DEFAULT_PATTERNS: ClassVar[list[tuple[str, str]]] = [
        # API keys
        (r"(sk-[a-zA-Z0-9]{20,})", r"sk-***REDACTED***"),
        (r"(api[_-]?key[\"']?\s*[:=]\s*[\"']?)([^\"'\s]+)", r"\1***REDACTED***"),
        # Bearer tokens
        (r"(Bearer\s+)([^\s]+)", r"\1***REDACTED***"),
        # Authorization headers
        (r"(Authorization[\"']?\s*[:=]\s*[\"']?)([^\"'\s]+)", r"\1***REDACTED***"),
        # Environment variable patterns
        (r"(OPENAI_API_KEY=)([^\s]+)", r"\1***REDACTED***"),
        (r"(ANTHROPIC_API_KEY=)([^\s]+)", r"\1***REDACTED***"),
        (r"(GOOGLE_API_KEY=)([^\s]+)", r"\1***REDACTED***"),
    ]

    def __init__(self, patterns: list[tuple[str, str]] | None = None) -> None:
        """Initialize masker with patterns.

        Args:
            patterns: List of (pattern, replacement) tuples
        """
        self._patterns = [
            (re.compile(p, re.IGNORECASE), r)
            for p, r in (patterns or self.DEFAULT_PATTERNS)
        ]

    def mask(self, text: str) -> str:
        """Mask sensitive data in text.

        Args:
            text: Text to mask

        Returns:
            Masked text
        """
        result = text
        for pattern, replacement in self._patterns:
            result = pattern.sub(replacement, result)
        return result

    def mask_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Mask sensitive data in dictionary.

        Args:
            data: Dictionary to mask

        Returns:
            Masked dictionary
        """
        result: dict[str, Any] = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(
                sensitive in key_lower
                for sensitive in ["key", "token", "secret", "password", "auth"]
            ):
                result[key] = "***REDACTED***"
            elif isinstance(value, str):
                result[key] = self.mask(value)
            elif isinstance(value, dict):
                result[key] = self.mask_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.mask_dict(v) if isinstance(v, dict) else v for v in value
                ]
            else:
                result[key] = value
        return result


class JsonFormatter(logging.Formatter):
    """JSON log formatter."""

    def __init__(
        self,
        masker: SensitiveDataMasker | None = None,
        include_timestamp: bool = True,
    ) -> None:
        """Initialize formatter.

        Args:
            masker: Sensitive data masker
            include_timestamp: Whether to include timestamp
        """
        super().__init__()
        self._masker = masker or SensitiveDataMasker()
        self._include_timestamp = include_timestamp

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": self._masker.mask(record.getMessage()),
        }

        if self._include_timestamp:
            log_data["timestamp"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            ) + f".{int(record.msecs):03d}Z"

        # Add context
        context = get_log_context()
        if context_dict := context.to_dict():
            log_data["context"] = context_dict

        # Add extra fields
        if hasattr(record, "extra_fields"):
            extra = self._masker.mask_dict(record.extra_fields)
            log_data.update(extra)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter."""

    def __init__(
        self,
        masker: SensitiveDataMasker | None = None,
        include_context: bool = True,
    ) -> None:
        """Initialize formatter.

        Args:
            masker: Sensitive data masker
            include_context: Whether to include context
        """
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._masker = masker or SensitiveDataMasker()
        self._include_context = include_context

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as text."""
        # Mask message
        original_msg = record.msg
        record.msg = self._masker.mask(str(record.msg))

        result = super().format(record)

        # Restore original
        record.msg = original_msg

        # Add context
        if self._include_context:
            context = get_log_context()
            if context_dict := context.to_dict():
                context_str = " ".join(f"{k}={v}" for k, v in context_dict.items())
                result = f"{result} | {context_str}"

        return result


class AiLibLogger:
    """Logger for ai-lib-python with structured logging support.

    Example:
        >>> logger = AiLibLogger.get_logger("ai_lib_python.client")
        >>> logger.info("Request started", model="gpt-4o", tokens=100)
    """

    _loggers: ClassVar[dict[str, logging.Logger]] = {}
    _level: ClassVar[LogLevel] = LogLevel.INFO
    _formatter: ClassVar[logging.Formatter | None] = None
    _handler: ClassVar[logging.Handler | None] = None

    @classmethod
    def configure(
        cls,
        level: LogLevel = LogLevel.INFO,
        format: str = "json",
        stream: Any = None,
        masker: SensitiveDataMasker | None = None,
    ) -> None:
        """Configure global logging settings.

        Args:
            level: Log level
            format: Output format ('json' or 'text')
            stream: Output stream (default: stderr)
            masker: Sensitive data masker
        """
        cls._level = level

        # Create formatter
        if format == "json":
            cls._formatter = JsonFormatter(masker=masker)
        else:
            cls._formatter = TextFormatter(masker=masker)

        # Create handler
        cls._handler = logging.StreamHandler(stream or sys.stderr)
        cls._handler.setFormatter(cls._formatter)
        cls._handler.setLevel(level.to_logging_level())

        # Update existing loggers
        for logger in cls._loggers.values():
            logger.handlers.clear()
            logger.addHandler(cls._handler)
            logger.setLevel(level.to_logging_level())

    @classmethod
    def get_logger(cls, name: str) -> AiLibLogger:
        """Get or create a logger.

        Args:
            name: Logger name

        Returns:
            Logger instance
        """
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(cls._level.to_logging_level())

            # Add handler if configured
            if cls._handler:
                logger.handlers.clear()
                logger.addHandler(cls._handler)
            elif not logger.handlers:
                # Default configuration
                handler = logging.StreamHandler(sys.stderr)
                handler.setFormatter(TextFormatter())
                logger.addHandler(handler)

            logger.propagate = False
            cls._loggers[name] = logger

        return cls(cls._loggers[name])

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize with underlying logger."""
        self._logger = logger

    def _log(
        self, level: int, msg: str, exc_info: bool = False, **kwargs: Any
    ) -> None:
        """Internal log method."""
        # Create record with extra fields
        extra = {"extra_fields": kwargs} if kwargs else {}
        self._logger.log(level, msg, exc_info=exc_info, extra=extra)

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log error message."""
        self._log(logging.ERROR, msg, exc_info=exc_info, **kwargs)

    def critical(self, msg: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, msg, exc_info=exc_info, **kwargs)

    def exception(self, msg: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self._log(logging.ERROR, msg, exc_info=True, **kwargs)


# Convenience function
def get_logger(name: str) -> AiLibLogger:
    """Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return AiLibLogger.get_logger(name)
