"""ExecutionMetadata schema alignment tests (PT-073f)."""

from __future__ import annotations

import json
from pathlib import Path

import fastjsonschema
import pytest

from ai_lib_python.types.execution_result import ExecutionMetadata, ExecutionUsage


def _schema_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    env = __import__("os").environ.get("AI_PROTOCOL_DIR")
    if env:
        return Path(env) / "schemas" / "v2" / "execution-metadata.json"
    candidate = root.parent / "ai-protocol" / "schemas" / "v2" / "execution-metadata.json"
    if candidate.is_file():
        return candidate
    pytest.skip("AI_PROTOCOL_DIR or sibling ai-protocol repo required for schema test")


def test_execution_metadata_serializes_to_schema() -> None:
    meta = ExecutionMetadata(
        provider_id="mock-openai",
        model_id="gpt-test",
        execution_latency_ms=8,
        translation_latency_ms=2,
        micro_retry_count=0,
    )
    payload = json.loads(meta.model_dump_json(exclude_none=True))
    schema = json.loads(_schema_path().read_text(encoding="utf-8"))
    fastjsonschema.validate(schema, payload)


def test_execution_metadata_with_usage_matches_schema() -> None:
    meta = ExecutionMetadata(
        provider_id="mock-openai",
        model_id="gpt-test",
        execution_latency_ms=10,
        translation_latency_ms=2,
        micro_retry_count=1,
        error_code="E1003",
        usage=ExecutionUsage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
    )
    payload = json.loads(meta.model_dump_json(exclude_none=True))
    schema = json.loads(_schema_path().read_text(encoding="utf-8"))
    fastjsonschema.validate(schema, payload)
