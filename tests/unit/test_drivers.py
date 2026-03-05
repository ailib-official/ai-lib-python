"""Driver 单元测试。"""

from __future__ import annotations

import json

from ai_lib_python.drivers import (
    AnthropicDriver,
    GeminiDriver,
    OpenAiDriver,
    create_driver,
)
from ai_lib_python.protocol.v2.capabilities import Capability
from ai_lib_python.protocol.v2.manifest import ApiStyle
from ai_lib_python.types.message import Message


class TestOpenAiDriver:
    def setup_method(self) -> None:
        self.driver = OpenAiDriver("openai", [Capability.TEXT, Capability.STREAMING])

    def test_build_request(self) -> None:
        messages = [Message.user("Hello")]
        req = self.driver.build_request(
            messages, "gpt-4", temperature=0.7, max_tokens=1024, stream=True
        )
        assert req.stream
        assert req.body["model"] == "gpt-4"
        assert req.body["temperature"] == 0.7

    def test_parse_response(self) -> None:
        body = {
            "choices": [{"message": {"content": "Hi!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        resp = self.driver.parse_response(body)
        assert resp.content == "Hi!"
        assert resp.finish_reason == "stop"
        assert resp.usage is not None
        assert resp.usage.total_tokens == 15

    def test_parse_stream_event(self) -> None:
        data = json.dumps({"choices": [{"delta": {"content": "World"}, "index": 0}]})
        event = self.driver.parse_stream_event(data)
        assert event is not None
        assert event.is_content_delta
        assert event.as_content_delta.content == "World"

    def test_stream_done(self) -> None:
        assert self.driver.is_stream_done("[DONE]")
        assert not self.driver.is_stream_done('{"choices":[]}')


class TestAnthropicDriver:
    def setup_method(self) -> None:
        self.driver = AnthropicDriver("anthropic", [Capability.TEXT])

    def test_system_extraction(self) -> None:
        messages = [Message.system("Be concise."), Message.user("Hi")]
        req = self.driver.build_request(messages, "claude-sonnet-4-20250514")
        assert req.body["system"] == "Be concise."
        assert len(req.body["messages"]) == 1
        assert req.headers.get("anthropic-version") == "2023-06-01"

    def test_max_tokens_default(self) -> None:
        req = self.driver.build_request([Message.user("Hi")], "claude-sonnet-4-20250514")
        assert req.body["max_tokens"] == 4096

    def test_parse_response(self) -> None:
        body = {
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        resp = self.driver.parse_response(body)
        assert resp.content == "Hello!"
        assert resp.finish_reason == "stop"
        assert resp.usage is not None
        assert resp.usage.total_tokens == 15

    def test_stop_reason_normalization(self) -> None:
        body = {
            "content": [{"type": "text", "text": ""}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        resp = self.driver.parse_response(body)
        assert resp.finish_reason == "tool_calls"

    def test_parse_stream_delta(self) -> None:
        data = json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hi"},
        })
        event = self.driver.parse_stream_event(data)
        assert event is not None
        assert event.is_content_delta
        assert event.as_content_delta.content == "Hi"


class TestGeminiDriver:
    def setup_method(self) -> None:
        self.driver = GeminiDriver("google", [Capability.TEXT])

    def test_system_instruction(self) -> None:
        messages = [Message.system("Be concise."), Message.user("Explain Rust.")]
        req = self.driver.build_request(messages, "gemini-2.0-flash")
        assert req.body["system_instruction"]["parts"][0]["text"] == "Be concise."
        assert len(req.body["contents"]) == 1
        assert req.body["contents"][0]["role"] == "user"

    def test_role_mapping(self) -> None:
        messages = [
            Message.user("Hi"),
            Message.assistant("Hello!"),
            Message.user("How?"),
        ]
        req = self.driver.build_request(messages, "gemini-2.0-flash")
        assert req.body["contents"][0]["role"] == "user"
        assert req.body["contents"][1]["role"] == "model"
        assert req.body["contents"][2]["role"] == "user"

    def test_generation_config(self) -> None:
        req = self.driver.build_request(
            [Message.user("Hi")], "gemini-2.0-flash", temperature=0.5, max_tokens=2048
        )
        assert req.body["generationConfig"]["temperature"] == 0.5
        assert req.body["generationConfig"]["maxOutputTokens"] == 2048

    def test_parse_response(self) -> None:
        body = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hi!"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 3,
                "totalTokenCount": 8,
            },
        }
        resp = self.driver.parse_response(body)
        assert resp.content == "Hi!"
        assert resp.finish_reason == "stop"
        assert resp.usage is not None
        assert resp.usage.total_tokens == 8

    def test_finish_reason_normalization(self) -> None:
        body = {
            "candidates": [
                {
                    "content": {"parts": [{"text": ""}], "role": "model"},
                    "finishReason": "SAFETY",
                }
            ],
        }
        resp = self.driver.parse_response(body)
        assert resp.finish_reason == "content_filter"

    def test_parse_stream_delta(self) -> None:
        data = json.dumps({
            "candidates": [{"content": {"parts": [{"text": "World"}], "role": "model"}}]
        })
        event = self.driver.parse_stream_event(data)
        assert event is not None
        assert event.is_content_delta
        assert event.as_content_delta.content == "World"


class TestCreateDriver:
    def test_factory_openai(self) -> None:
        d = create_driver(ApiStyle.OPENAI_COMPATIBLE, "openai")
        assert isinstance(d, OpenAiDriver)

    def test_factory_anthropic(self) -> None:
        d = create_driver(ApiStyle.ANTHROPIC_MESSAGES, "anthropic")
        assert isinstance(d, AnthropicDriver)

    def test_factory_gemini(self) -> None:
        d = create_driver(ApiStyle.GEMINI_GENERATE, "google")
        assert isinstance(d, GeminiDriver)

    def test_factory_custom_fallback(self) -> None:
        d = create_driver(ApiStyle.CUSTOM, "custom-provider")
        assert isinstance(d, OpenAiDriver)
