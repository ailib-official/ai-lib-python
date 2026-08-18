"""ALP-RSN-001: Thinking channel parity with ALR-RSN-001 (Rust fixtures)."""

from __future__ import annotations

import json

from ai_lib_python.client.core import AiClient
from ai_lib_python.client.response import ChatResponse
from ai_lib_python.drivers import OpenAiDriver
from ai_lib_python.pipeline.event_map import DefaultEventMapper
from ai_lib_python.protocol.manifest import ProtocolManifest
from ai_lib_python.utils.thinking_extract import (
    thinking_from_openai_compat_delta,
    thinking_from_openai_compat_message,
)


def _minimal_manifest() -> ProtocolManifest:
    return ProtocolManifest.model_validate(
        {
            "id": "openai",
            "endpoint": {"base_url": "https://api.openai.com/v1", "chat": "/chat/completions"},
        }
    )


class TestThinkingExtract:
    def test_delta_prefers_reasoning_content(self) -> None:
        frame = {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "a",
                        "thinking": "b",
                        "content": "c",
                    }
                }
            ]
        }
        assert thinking_from_openai_compat_delta(frame) == "a"

    def test_delta_alias_thinking(self) -> None:
        frame = {"choices": [{"delta": {"thinking": "plan"}}]}
        assert thinking_from_openai_compat_delta(frame) == "plan"

    def test_message_reasoning_not_content(self) -> None:
        frame = {"choices": [{"message": {"content": "", "reasoning_content": "only think"}}]}
        assert thinking_from_openai_compat_message(frame) == "only think"


class TestDefaultMapperThinkingAliases:
    def test_same_frame_emits_thinking_then_content(self) -> None:
        mapper = DefaultEventMapper()
        frame = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "plan",
                        "content": "answer",
                    },
                }
            ]
        }
        events = mapper.map_frame(frame)
        assert len(events) == 2
        assert events[0].is_thinking_delta
        assert events[0].as_thinking_delta.thinking == "plan"
        assert events[1].is_content_delta
        assert events[1].as_content_delta.content == "answer"

    def test_alias_thinking_key(self) -> None:
        mapper = DefaultEventMapper()
        frame = {"choices": [{"delta": {"thinking": "via-alias"}}]}
        events = mapper.map_frame(frame)
        assert len(events) == 1
        assert events[0].is_thinking_delta
        assert events[0].as_thinking_delta.thinking == "via-alias"


class TestNonStreamThinkingSeparation:
    def test_parse_response_keeps_thinking_separate(self) -> None:
        client = AiClient.__new__(AiClient)
        client._manifest = _minimal_manifest()
        data = {
            "choices": [
                {
                    "message": {"content": "", "reasoning_content": "only think"},
                    "finish_reason": "stop",
                }
            ]
        }
        r: ChatResponse = AiClient._parse_response(client, data)
        assert r.content == ""
        assert r.thinking == "only think"

    def test_parse_response_content_and_thinking_both(self) -> None:
        client = AiClient.__new__(AiClient)
        client._manifest = _minimal_manifest()
        data = {
            "choices": [
                {
                    "message": {
                        "content": "final",
                        "reasoning_content": "scratch",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        r = AiClient._parse_response(client, data)
        assert r.content == "final"
        assert r.thinking == "scratch"


class TestOpenAiDriverThinking:
    def test_parse_response_thinking_field(self) -> None:
        driver = OpenAiDriver("openai", [])
        body = {
            "choices": [
                {
                    "message": {"content": "", "reasoning_content": "only think"},
                    "finish_reason": "stop",
                }
            ]
        }
        resp = driver.parse_response(body)
        assert resp.content == "" or resp.content is None
        assert resp.thinking == "only think"

    def test_parse_stream_same_frame_both(self) -> None:
        driver = OpenAiDriver("openai", [])
        data = json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "t",
                            "content": "c",
                        },
                        "index": 0,
                    }
                ]
            }
        )
        events = driver.parse_stream_event(data)
        assert len(events) == 2
        assert events[0].is_thinking_delta
        assert events[0].as_thinking_delta.thinking == "t"
        assert events[1].is_content_delta
        assert events[1].as_content_delta.content == "c"
