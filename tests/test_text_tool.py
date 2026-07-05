"""Unit tests for text tool call parsing."""

from ai_lib_python.types.text_tool import (
    PromptLevel,
    StandardTextToolParser,
    TextToolConfig,
)
from ai_lib_python.types.tool import FunctionDefinition, ToolDefinition

_DSML = "\uff5c\uff5cDSML\uff5c\uff5c"


def test_strict_parse_standard_format() -> None:
    parser = StandardTextToolParser(config=TextToolConfig(lenient_parsing=False))
    text = (
        "I'll list the files for you.\n"
        '<tool_call>\n{"name": "shell", "arguments": {"command": "ls -la"}}\n</tool_call>'
    )
    remaining, calls = parser.parse(text)
    assert remaining == "I'll list the files for you."
    assert len(calls) == 1
    assert calls[0].name == "shell"
    assert calls[0].arguments["command"] == "ls -la"


def test_lenient_shell_dialect() -> None:
    parser = StandardTextToolParser(config=TextToolConfig(lenient_parsing=True))
    remaining, calls = parser.parse("Running command:\n<shell><command>ls</command></shell>")
    assert remaining == "Running command:"
    assert calls[0].name == "shell"
    assert calls[0].arguments["command"] == "ls"


def test_lenient_deepseek_dsml_dialect() -> None:
    parser = StandardTextToolParser(config=TextToolConfig(lenient_parsing=True))
    text = (
        "我来检查 piubt 服务器上 pifan 服务的概况。\n\n"
        f"<{_DSML}tool_calls>\n"
        f'<{_DSML}invoke name="shell">\n'
        f'<{_DSML}parameter name="command" string="true">'
        'ssh piubt "systemctl status pifan" 2>&1'
        f"</{_DSML}parameter>\n"
        f"</{_DSML}invoke>\n"
        f"</{_DSML}tool_calls>"
    )
    remaining, calls = parser.parse(text)
    assert remaining == "我来检查 piubt 服务器上 pifan 服务的概况。"
    assert calls[0].name == "shell"
    assert calls[0].arguments["command"] == 'ssh piubt "systemctl status pifan" 2>&1'


def test_hybrid_falls_back_to_text_when_native_empty() -> None:
    from ai_lib_python.types.text_tool import parse_hybrid_tool_calls

    parser = StandardTextToolParser(config=TextToolConfig(lenient_parsing=True))
    text = "<shell><command>ls</command></shell>"
    remaining, calls = parse_hybrid_tool_calls(parser, text, [])
    assert remaining == ""
    assert calls[0].name == "shell"


def test_hybrid_prefers_native_calls() -> None:
    from ai_lib_python.types.text_tool import TextParsedToolCall, parse_hybrid_tool_calls

    parser = StandardTextToolParser(config=TextToolConfig(lenient_parsing=True))
    native = [TextParsedToolCall(id="call_abc", name="shell", arguments={"command": "ls -la"})]
    text = "<shell><command>ignored</command></shell>"
    remaining, calls = parse_hybrid_tool_calls(parser, text, native)
    assert remaining.strip() == text.strip()
    assert calls[0].arguments["command"] == "ls -la"


def test_prompt_l2_contains_counterexamples() -> None:
    parser = StandardTextToolParser(config=TextToolConfig(prompt_level=PromptLevel.L2, locale="en"))
    tools = [
        ToolDefinition(
            function=FunctionDefinition(name="shell", description="Execute shell commands")
        )
    ]
    prompt = parser.prompt_instructions(tools)
    assert "<tool_call>" in prompt
    assert "WILL BE IGNORED" in prompt
    assert "shell" in prompt


def test_lenient_plain_shell_body_dialect() -> None:
    text = (
        '让我检查一下。\n<shell>\nwhich opencode 2>/dev/null || echo "not found"\n</shell>'
    )
    parser = StandardTextToolParser.from_manifest_tool_calling(
        {
            "native": {"supported": True, "reliability": "partial"},
            "text_fallback": {
                "prompt_level": "L2",
                "known_dialects": [{"tag": "shell", "map_to": "shell"}],
            },
        }
    )
    remaining, calls = parser.parse(text)
    assert len(calls) == 1
    assert calls[0].name == "shell"
    assert calls[0].arguments["command"] == 'which opencode 2>/dev/null || echo "not found"'
    assert "让我检查一下" in remaining


def test_tool_calling_policy_deepseek_partial_is_hybrid() -> None:
    from ai_lib_python.types.text_tool import NativeStrategy, ToolCallingPolicy

    policy = ToolCallingPolicy.from_tool_calling(
        {
            "native": {"supported": True, "reliability": "partial"},
            "text_fallback": {
                "prompt_level": "L2",
                "known_dialects": [{"tag": "shell", "map_to": "shell"}],
            },
        }
    )
    assert policy.native_strategy == NativeStrategy.HYBRID
    assert policy.send_native_tool_specs()
    assert policy.prefer_native_dispatcher()


def test_tool_calling_policy_text_only_when_no_native() -> None:
    from ai_lib_python.types.text_tool import NativeStrategy, ToolCallingPolicy

    policy = ToolCallingPolicy.from_tool_calling(
        {
            "native": {"supported": False},
            "text_fallback": {"prompt_level": "L2"},
        }
    )
    assert policy.native_strategy == NativeStrategy.TEXT_ONLY
    assert not policy.send_native_tool_specs()


def test_capabilities_v2_preserves_tool_calling() -> None:
    from ai_lib_python.protocol.v2.capabilities import CapabilitiesV2

    caps = CapabilitiesV2.model_validate(
        {
            "required": ["text"],
            "tool_calling": {
                "native": {"supported": True, "reliability": "partial"},
                "text_fallback": {"prompt_level": "L2"},
            },
        }
    )
    assert caps.tool_calling is not None
    assert caps.tool_calling["native"]["reliability"] == "partial"
