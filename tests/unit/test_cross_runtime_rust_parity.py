"""Parity checks with ai-lib-rust / aitest debug notes (cross-runtime alignment)."""

from __future__ import annotations

from ai_lib_python.client.core import AiClient
from ai_lib_python.client.response import ChatResponse
from ai_lib_python.pipeline.event_map import DefaultEventMapper, ProtocolEventMapper, create_event_mapper
from ai_lib_python.protocol.manifest import (
    DecoderConfig,
    EndpointConfig,
    EventMapRule,
    ProtocolManifest,
    StreamingConfig,
)


def _minimal_manifest(**kwargs: object) -> ProtocolManifest:
    data: dict = {"id": "test", "endpoint": {"base_url": "https://api.example.com"}}
    data.update(kwargs)
    return ProtocolManifest.model_validate(data)


def test_openai_chat_decoder_uses_path_mapper_even_with_event_map() -> None:
    """Rust: prefer PathEventMapper when decoder.strategy == openai_chat."""
    streaming = StreamingConfig(
        decoder=DecoderConfig(format="sse", strategy="openai_chat"),
        event_map=[
            EventMapRule(
                match="exists($.this_rule_should_not_be_used)",
                emit="PartialContentDelta",
                fields={"content": "$.bad"},
            )
        ],
        content_path="choices[0].delta.content",
    )
    mapper = create_event_mapper(streaming)
    assert isinstance(mapper, DefaultEventMapper)
    assert not isinstance(mapper, ProtocolEventMapper)


def test_non_openai_chat_still_uses_protocol_event_mapper_when_rules_present() -> None:
    streaming = StreamingConfig(
        decoder=DecoderConfig(format="sse", strategy=None),
        event_map=[
            EventMapRule(
                match="exists($.choices[*].delta.content)",
                emit="PartialContentDelta",
                fields={"content": "$.choices[*].delta.content"},
            )
        ],
    )
    mapper = create_event_mapper(streaming)
    assert isinstance(mapper, ProtocolEventMapper)


def test_parse_response_manifest_content_path_before_openai_shape() -> None:
    manifest = _minimal_manifest(
        response_paths={"content": "custom.text", "usage": "u"},
    )
    client = AiClient.__new__(AiClient)
    client._manifest = manifest
    data = {
        "choices": [{"message": {"content": "ignored"}, "finish_reason": "stop"}],
        "custom": {"text": "from_manifest"},
        "u": {"prompt_tokens": 1},
    }
    r: ChatResponse = AiClient._parse_response(client, data)
    assert r.content == "from_manifest"
    assert r.usage == {"prompt_tokens": 1}


def test_parse_response_v2_openai_fallback_when_no_response_paths() -> None:
    manifest = _minimal_manifest()
    client = AiClient.__new__(AiClient)
    client._manifest = manifest
    data = {"choices": [{"message": {"content": "hello"}, "finish_reason": "length"}]}
    r = AiClient._parse_response(client, data)
    assert r.content == "hello"
    assert r.finish_reason == "length"


def test_parse_response_reasoning_fallback_when_content_empty() -> None:
    manifest = _minimal_manifest()
    client = AiClient.__new__(AiClient)
    client._manifest = manifest
    data = {
        "choices": [
            {
                "message": {"content": "", "reasoning_content": "think first"},
                "finish_reason": "stop",
            }
        ]
    }
    r = AiClient._parse_response(client, data)
    assert r.content == "think first"
