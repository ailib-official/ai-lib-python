# E/P boundary: execution result types (Paper1 section 3.1-3.2).
# E/P 边界: 执行层返回结果类型 (Paper1 3.1-3.2).
#
# The execution layer (E) returns ExecutionResult with ExecutionMetadata.
# The contact / policy layer (P) consumes metadata for routing, retry, and degradation.

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ExecutionUsage(BaseModel):
    """Token usage aligned with driver usage fields (all optional per schema)."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_creation_tokens: int | None = Field(default=None, ge=0)


class ExecutionMetadata(BaseModel):
    """Metadata returned with every E-layer call for P-layer policy decisions."""

    model_config = ConfigDict(extra="forbid")

    provider_id: str
    model_id: str
    execution_latency_ms: int = Field(ge=0)
    translation_latency_ms: int = Field(ge=0)
    micro_retry_count: int = Field(ge=0, le=255)
    error_code: str | None = Field(default=None, pattern=r"^E[0-9]{4}$")
    usage: ExecutionUsage | None = None


class ExecutionResult(BaseModel, Generic[T]):
    """Successful execution envelope from E: payload plus metadata for P."""

    model_config = ConfigDict(extra="forbid")

    data: T
    metadata: ExecutionMetadata
