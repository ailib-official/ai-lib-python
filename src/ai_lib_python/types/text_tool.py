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


class NativeStrategy(str, Enum):
    """Native tool-calling strategy derived from manifest ``tool_calling``."""

    FULL = "full"
    HYBRID = "hybrid"
    TEXT_ONLY = "text_only"


class TextToolDeviation(str, Enum):
    """Observed text-tool markup categories (align with ai-protocol format.yaml)."""

    STANDARD_TOOL_CALL = "standard_tool_call"
    SHELL = "shell"
    BASH = "bash"
    DSML = "dsml"


@dataclass
class KnownDialect:
    """Manifest ``known_dialects`` entry: XML tag → tool name."""

    tag: str
    map_to: str = ""


@dataclass
class TextToolConfig:
    """Configuration for text tool call parsing and prompt generation."""

    lenient_parsing: bool = False
    max_call_depth: int = 1
    include_counterexamples: bool = True
    prompt_level: PromptLevel = PromptLevel.L1
    locale: str = "en"
    args_key: str | None = None
    dialects: list[KnownDialect] = field(default_factory=list)


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
_SHELL_PLAIN_BODY_RE = re.compile(r"(?s)<shell>\s*(.*?)\s*</shell>")
_BASH_DIALECT_RE = re.compile(r"(?s)<bash>(.*?)</bash>")
_OUTER_WRAPPER_RE = re.compile(r"(?s)<tool_calls>\s*(.*?)\s*</tool_calls>")
_NAME_ATTR_RE = re.compile(r'name="([^"]+)"')
# DeepSeek DSML: U+FF5C fullwidth vertical line delimiter (see _DSML_TAG)
_DSML_TAG = "\uff5c\uff5cDSML\uff5c\uff5c"
_DSML_INVOKE_RE = re.compile(
    rf"(?s)<{_DSML_TAG}invoke\s+name=\"([^\"]+)\">(.*?)</{_DSML_TAG}invoke>"
)
_DSML_PARAMETER_RE = re.compile(
    rf"(?s)<{_DSML_TAG}parameter\s+name=\"([^\"]+)\"[^>]*>(.*?)</{_DSML_TAG}parameter>"
)
_DSML_WRAPPER_RE = re.compile(rf"(?s)<{_DSML_TAG}tool_calls>\s*(.*?)\s*</{_DSML_TAG}tool_calls>")


def _default_lenient_parser() -> StandardTextToolParser:
    return StandardTextToolParser(
        config=TextToolConfig(
            lenient_parsing=True,
            prompt_level=PromptLevel.L2,
            include_counterexamples=True,
        )
    )


def _infer_native_strategy(tool_calling: dict[str, Any]) -> NativeStrategy:
    native = tool_calling.get("native") or {}
    native_supported = bool(native.get("supported", False))
    if not native_supported:
        return NativeStrategy.TEXT_ONLY

    reliability = str(native.get("reliability", "unreliable"))
    has_text_fallback = tool_calling.get("text_fallback") is not None

    if reliability == "full":
        return NativeStrategy.FULL
    if reliability == "partial" and has_text_fallback:
        return NativeStrategy.HYBRID
    if reliability == "partial":
        return NativeStrategy.FULL
    if has_text_fallback:
        return NativeStrategy.TEXT_ONLY
    return NativeStrategy.FULL


def detect_text_tool_deviation(text: str) -> TextToolDeviation | None:
    """Detect the first recognizable text-tool markup in LLM output."""
    if _DSML_TAG in text:
        return TextToolDeviation.DSML
    if _SHELL_DIALECT_RE.search(text) or _SHELL_PLAIN_BODY_RE.search(text):
        return TextToolDeviation.SHELL
    if _BASH_DIALECT_RE.search(text):
        return TextToolDeviation.BASH
    if _TOOL_CALL_BLOCK_RE.search(text):
        return TextToolDeviation.STANDARD_TOOL_CALL
    return None


def parse_hybrid_tool_calls(
    parser: TextToolParser,
    content: str,
    native_calls: list[TextParsedToolCall],
) -> tuple[str, list[TextParsedToolCall]]:
    """Prefer native structured calls; fall back to lenient text parse when empty."""
    if native_calls:
        return content, list(native_calls)
    return parser.parse(content)


def _shell_tool_call(command: str, map_to: str, idx: int) -> TextParsedToolCall:
    name = map_to if map_to else "shell"
    return TextParsedToolCall(
        id=f"text_tool_{idx}",
        name=name,
        arguments={"command": command},
    )


def _try_parse_configured_dialects(
    text: str, dialects: list[KnownDialect]
) -> tuple[TextParsedToolCall, tuple[int, int]] | None:
    for dialect in dialects:
        if dialect.tag == "shell":
            match = _SHELL_DIALECT_RE.search(text)
            if match:
                cmd = (match.group(1) or "").strip()
                return _shell_tool_call(cmd, dialect.map_to, 0), (match.start(), match.end())
            plain = _SHELL_PLAIN_BODY_RE.search(text)
            if plain:
                body = (plain.group(1) or "").strip()
                if body.startswith("<command>"):
                    continue
                return _shell_tool_call(body, dialect.map_to, 0), (plain.start(), plain.end())
        elif dialect.tag == "bash":
            match = _BASH_DIALECT_RE.search(text)
            if match:
                cmd = (match.group(1) or "").strip()
                return _shell_tool_call(cmd, dialect.map_to, 0), (match.start(), match.end())
    return None


def _try_parse_legacy_dialects(text: str) -> tuple[TextParsedToolCall, tuple[int, int]] | None:
    match = _SHELL_DIALECT_RE.search(text)
    if match:
        cmd = (match.group(1) or "").strip()
        return _shell_tool_call(cmd, "shell", 0), (match.start(), match.end())
    plain = _SHELL_PLAIN_BODY_RE.search(text)
    if plain:
        body = (plain.group(1) or "").strip()
        if not body.startswith("<command>"):
            return _shell_tool_call(body, "shell", 0), (plain.start(), plain.end())
    bash = _BASH_DIALECT_RE.search(text)
    if bash:
        cmd = (bash.group(1) or "").strip()
        return _shell_tool_call(cmd, "shell", 0), (bash.start(), bash.end())
    return None


def _parse_dsml_dialect(text: str) -> tuple[list[TextParsedToolCall], list[tuple[int, int]]]:
    tool_calls: list[TextParsedToolCall] = []
    spans_to_remove: list[tuple[int, int]] = []

    for match in _DSML_INVOKE_RE.finditer(text):
        tool_name = (match.group(1) or "").strip()
        if not tool_name:
            continue
        body = match.group(2) or ""
        arguments: dict[str, Any] = {}
        for param in _DSML_PARAMETER_RE.finditer(body):
            key = param.group(1) or ""
            value = (param.group(2) or "").strip()
            if key:
                arguments[key] = value
        idx = len(tool_calls)
        tool_calls.append(
            TextParsedToolCall(id=f"text_tool_{idx}", name=tool_name, arguments=arguments)
        )
        spans_to_remove.append((match.start(), match.end()))

    if tool_calls:
        wrapper = _DSML_WRAPPER_RE.search(text)
        if wrapper:
            spans_to_remove = [(wrapper.start(), wrapper.end())]

    return tool_calls, spans_to_remove


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


def _parse_text_tool_calls(
    text: str, config: TextToolConfig
) -> tuple[str, list[TextParsedToolCall]]:
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
        dsml_calls, dsml_spans = _parse_dsml_dialect(remaining)
        if dsml_calls:
            tool_calls.extend(dsml_calls)
            spans_to_remove.extend(dsml_spans)
        else:
            dialect_result = (
                _try_parse_configured_dialects(remaining, config.dialects)
                if config.dialects
                else _try_parse_legacy_dialects(remaining)
            )
            if dialect_result:
                call, span = dialect_result
                tool_calls.append(call)
                spans_to_remove.append(span)

    chars = list(remaining)
    for start, end in sorted(spans_to_remove, key=lambda x: x[0], reverse=True):
        del chars[start:end]
    remaining_text = "".join(chars)
    remaining_text = "\n".join(
        line.strip() for line in remaining_text.splitlines() if line.strip()
    ).strip()

    return remaining_text, tool_calls


def _generate_prompt_instructions(tools: list[ToolDefinition], config: TextToolConfig) -> str:
    tool_list = "\n".join(f"- {t.function.name}: {t.function.description or ''}" for t in tools)
    is_zh = config.locale.startswith("zh")

    if config.prompt_level == PromptLevel.L1 and is_zh:
        return (
            "## 工具调用协议\n\n"
            '<tool_call>\n{"name": "工具名", "arguments": {"参数": "值"}}\n</tool_call>\n\n'
            f"可用工具:\n{tool_list}"
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
            "关键规则:\n"
            "- 只能使用 <tool_call>。<shell>、<bash>、<function> 将被忽略。\n"
            '- JSON 必须包含 "name" 和 "arguments"。\n\n'
            f"可用工具:\n{tool_list}"
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
        fallback = tool_calling.get("text_fallback")
        if fallback is not None and fallback is not False:
            if not isinstance(fallback, dict):
                fallback = {}
            level = str(fallback.get("prompt_level", "L2")).upper()
            config.prompt_level = (
                PromptLevel(level) if level in PromptLevel.__members__ else PromptLevel.L2
            )
            if isinstance(fallback.get("args_key"), str):
                config.args_key = fallback["args_key"]
            known = fallback.get("known_dialects")
            if isinstance(known, list):
                for entry in known:
                    if not isinstance(entry, dict):
                        continue
                    tag = entry.get("tag")
                    if not isinstance(tag, str) or not tag:
                        continue
                    map_to = entry.get("map_to")
                    config.dialects.append(
                        KnownDialect(tag=tag, map_to=map_to if isinstance(map_to, str) else "")
                    )
            config.include_counterexamples = config.prompt_level != PromptLevel.L1
        native = tool_calling.get("native") or {}
        if native.get("reliability") == "full":
            config.lenient_parsing = False
        return cls(config=config)


@dataclass
class ToolCallingPolicy:
    """Runtime policy: dispatcher selection + manifest-backed parser."""

    native_strategy: NativeStrategy
    parser: StandardTextToolParser

    @classmethod
    def from_tool_calling(cls, tool_calling: dict[str, Any] | None) -> ToolCallingPolicy:
        parser = (
            StandardTextToolParser.from_manifest_tool_calling(tool_calling)
            if tool_calling is not None
            else _default_lenient_parser()
        )
        strategy = (
            _infer_native_strategy(tool_calling)
            if tool_calling is not None
            else NativeStrategy.TEXT_ONLY
        )
        return cls(native_strategy=strategy, parser=parser)

    def send_native_tool_specs(self) -> bool:
        return self.native_strategy in (NativeStrategy.FULL, NativeStrategy.HYBRID)

    def prefer_native_dispatcher(self) -> bool:
        return self.send_native_tool_specs()
