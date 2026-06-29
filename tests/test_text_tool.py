"""Unit tests for text tool call parsing."""

from ai_lib_python.types.text_tool import PromptLevel, StandardTextToolParser, TextToolConfig
from ai_lib_python.types.tool import FunctionDefinition, ToolDefinition


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
