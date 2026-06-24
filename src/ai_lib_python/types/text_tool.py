"""
Text-based tool call parsing for LLMs without reliable native function calling.

文本工具调用解析：适用于不支持或不稳定 native function calling 的 provider。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from ai_lib_python.types.tool import ToolDefinition


class PromptLevel(str, Enum):
    """Prompt strategy level (L1 / L2 / L3)."""

    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


@dataclass
class TextToolConfig:
    """Configuration for text tool call parsing and prompt generation."""

    lenient_parsing: bool = False
    max_call_depth: int = 1
    include_counterexamples: bool = True
    prompt_level: PromptLevel = PromptLevel.L1
    locale: str = "en"
    args_key: str | None = None


@dataclass
class TextParsedToolCall:
    """A tool call extracted from LLM text output."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextToolResult:
    """Tool execution result for text round-trip formatting."""

    tool_use_id: str
    content: Any
    is_error: bool = False


class TextToolParser(Protocol):
    """Cross-LLM text tool call parser protocol."""

    def parse(self, response_text: str) -> tuple[str, list[TextParsedToolCall]]: ...

    def prompt_instructions(self, tools: list[ToolDefinition]) -> str: ...

    def format_results(self, results: list[TextToolResult]) -> str: ...


_TOOL_CALL_BLOCK_RE = re.compile(r"(?s)<tool_call(?:\s+[^>]*)?>(.*?)</tool_call>")
_SHELL_DIALECT_RE = re.compile(r"(?s)<shell>\s*<command>(.*?)</command>\s*</shell>")
_BASH_DIALECT_RE = re.compile(r"(?s)<bash>(.*?)</bash>")
_OUTER_WRAPPER_RE = re.compile(r"(?s)<tool_calls>\s*(.*?)\s*</tool_calls>")
_NAME_ATTR_RE = re.compile(r'name="([^"]+)"')


def _unwrap_tool_calls_wrapper(text: str) -> str:
    match = _OUTER_WRAPPER_RE.search(text)
    return match.group(1) if match else text


def _extract_name_from_open_tag(full_match: str) -> str | None:
    match = _NAME_ATTR_RE.search(full_match)
    return match.group(1) if match else None


def _normalize_arguments(obj: dict[str, Any]) -> dict[str, Any]:
    if "arguments" in obj:
        val = obj["arguments"]
        return val if isinstance(val, dict) else {}
    for key in ("parameters", "params", "args"):
        if key in obj:
            val = obj[key]
            return val if isinstance(val, dict) else {}
    args = dict(obj)
    for key in ("name", "id", "type"):
        args.pop(key, None)
    return args


def _parse_json_body(body: str, attr_name: str | None) -> tuple[str, dict[str, Any]] | None:
    trimmed = body.strip()
    if not trimmed:
        return None
    try:
        value = json.loads(trimmed)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    if not isinstance(name, str) or not name:
        name = attr_name
    if not name:
        return None
    return name, _normalize_arguments(value)


def _parse_text_tool_calls(text: str, config: TextToolConfig) -> tuple[str, list[TextParsedToolCall]]:
    tool_calls: list[TextParsedToolCall] = []
    remaining = text

    if config.lenient_parsing:
        remaining = _unwrap_tool_calls_wrapper(remaining)

    spans_to_remove: list[tuple[int, int]] = []

    for match in _TOOL_CALL_BLOCK_RE.finditer(remaining):
        full = match.group(0)
        body = match.group(1) or ""
        attr_name = _extract_name_from_open_tag(full) if config.lenient_parsing else None
        parsed = _parse_json_body(body, attr_name)
        if parsed is None:
            continue
        name, arguments = parsed
        idx = len(tool_calls)
        tool_calls.append(TextParsedToolCall(id=f"text_tool_{idx}", name=name, arguments=arguments))
        spans_to_remove.append((match.start(), match.end()))

    if config.lenient_parsing and not tool_calls:
        shell_match = _SHELL_DIALECT_RE.search(remaining)
        if shell_match:
            cmd = (shell_match.group(1) or "").strip()
            tool_calls.append(
                TextParsedToolCall(id="text_tool_0", name="shell", arguments={"command": cmd})
            )
            spans_to_remove.append((shell_match.start(), shell_match.end()))
        else:
            bash_match = _BASH_DIALECT_RE.search(remaining)
            if bash_match:
                cmd = (bash_match.group(1) or "").strip()
                tool_calls.append(
                    TextParsedToolCall(id="text_tool_0", name="shell", arguments={"command": cmd})
                )
                spans_to_remove.append((bash_match.start(), bash_match.end()))

    chars = list(remaining)
    for start, end in sorted(spans_to_remove, key=lambda x: x[0], reverse=True):
        del chars[start:end]
    remaining_text = "".join(chars)
    remaining_text = "\n".join(line.strip() for line in remaining_text.splitlines() if line.strip()).strip()

    return remaining_text, tool_calls


def _generate_prompt_instructions(tools: list[ToolDefinition], config: TextToolConfig) -> str:
    tool_list = "\n".join(
        f"- {t.function.name}: {t.function.description or ''}" for t in tools
    )
    is_zh = config.locale.startswith("zh")

    if config.prompt_level == PromptLevel.L1 and is_zh:
        return (
            "## 工具调用协议\n\n"
            '<tool_call>\n{"name": "工具名", "arguments": {"参数": "值"}}\n</tool_call>\n\n'
            f"可用工具：\n{tool_list}"
        )
    if config.prompt_level == PromptLevel.L1:
        return (
            "## Tool Use Protocol\n\n"
            '<tool_call>\n{"name": "tool_name", "arguments": {"param": "value"}}\n</tool_call>\n\n'
            f"Available tools:\n{tool_list}"
        )
    if config.prompt_level == PromptLevel.L2 and is_zh:
        return (
            "## 工具调用协议\n\n"
            '<tool_call>\n{"name": "工具名", "arguments": {"参数": "值"}}\n</tool_call>\n\n'
            "关键规则：\n"
            "- 只能使用 <tool_call>。<shell>、<bash>、<function> 将被忽略。\n"
            '- JSON 必须包含 "name" 和 "arguments"。\n\n'
            f"可用工具：\n{tool_list}"
        )
    if config.prompt_level == PromptLevel.L2:
        return (
            "## Tool Use Protocol\n\n"
            '<tool_call>\n{"name": "tool_name", "arguments": {"param": "value"}}\n</tool_call>\n\n'
            "CRITICAL RULES:\n"
            "- Use <tool_call> ONLY. <shell>, <bash>, <function> WILL BE IGNORED.\n"
            '- JSON must contain "name" (string) and "arguments" (object).\n'
            "- Do NOT wrap in <tool_calls> or any other tag.\n\n"
            f"Available tools:\n{tool_list}"
        )
    return (
        "## Tool Use Protocol — Example\n\n"
        '<tool_call>\n{"name": "shell", "arguments": {"command": "ls -la"}}\n</tool_call>\n\n'
        "CRITICAL: <shell>, <bash>, <function> formats WILL BE IGNORED.\n\n"
        f"Available tools:\n{tool_list}"
    )


@dataclass
class StandardTextToolParser:
    """Default AI-Protocol `<tool_call>` text parser."""

    config: TextToolConfig = field(default_factory=TextToolConfig)

    def parse(self, response_text: str) -> tuple[str, list[TextParsedToolCall]]:
        return _parse_text_tool_calls(response_text, self.config)

    def prompt_instructions(self, tools: list[ToolDefinition]) -> str:
        return _generate_prompt_instructions(tools, self.config)

    def format_results(self, results: list[TextToolResult]) -> str:
        blocks: list[str] = []
        for result in results:
            body = json.dumps(
                {
                    "tool_use_id": result.tool_use_id,
                    "content": result.content,
                    "is_error": result.is_error,
                },
                ensure_ascii=False,
            )
            blocks.append(f"<tool_result>\n{body}\n</tool_result>")
        return "\n".join(blocks)

    @classmethod
    def from_manifest_tool_calling(cls, tool_calling: dict[str, Any]) -> StandardTextToolParser:
        config = TextToolConfig(lenient_parsing=True, prompt_level=PromptLevel.L2)
        fallback = tool_calling.get("text_fallback") or {}
        level = str(fallback.get("prompt_level", "L2")).upper()
        config.prompt_level = PromptLevel(level) if level in PromptLevel.__members__ else PromptLevel.L2
        if isinstance(fallback.get("args_key"), str):
            config.args_key = fallback["args_key"]
        config.include_counterexamples = config.prompt_level != PromptLevel.L1
        native = tool_calling.get("native") or {}
        if native.get("reliability") == "full":
            config.lenient_parsing = False
        return cls(config=config)
