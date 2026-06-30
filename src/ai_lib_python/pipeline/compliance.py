"""Conformance helpers for ai-protocol compliance YAML runners.

合规测试辅助：SSE 解码与 event_map / tool 累积。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from ai_lib_python.pipeline.decode import SSEDecoder
from ai_lib_python.utils.tool_call_assembler import ToolCallAssembler


def compliance_events_from_openai_frame(frame: dict[str, Any]) -> list[dict[str, Any]]:
    """Map a decoded OpenAI chat frame to compliance event records."""
    choices = frame.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    choice = choices[0] if isinstance(choices[0], dict) else {}
    delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}

    out: list[dict[str, Any]] = []
    if "content" in delta:
        out.append({"type": "PartialContentDelta", "content": delta.get("content")})
    if "tool_calls" in delta:
        out.append({"type": "PartialToolCall", "tool_calls": delta.get("tool_calls")})
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None:
        out.append({"type": "StreamEnd", "finish_reason": finish_reason})
    return out


def assemble_tool_call_partials(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge partial tool-call chunks via ``ToolCallAssembler``."""
    assembler = ToolCallAssembler()
    order: list[tuple[int, str]] = []
    meta: dict[tuple[int, str], dict[str, Any]] = {}
    started: set[str] = set()

    for chunk in chunks:
        index = int(chunk.get("index", 0))
        call_id = str(chunk.get("id", ""))
        function = chunk.get("function") if isinstance(chunk.get("function"), dict) else {}
        name = str(function.get("name", ""))
        args = str(function.get("arguments", ""))
        key = (index, call_id)

        if key not in meta:
            order.append(key)
            meta[key] = {
                "type": chunk.get("type", "function"),
                "name": name,
            }
        if name and call_id not in started:
            started.add(call_id)
            assembler.on_started(call_id, name, index)
        assembler.on_partial(call_id, args)

    by_id: dict[str, Any] = {}
    for tc in assembler.finalize():
        raw = tc.arguments_raw
        if raw is None and tc.arguments:
            raw = json.dumps(tc.arguments, separators=(",", ":"))
        if raw is None:
            raw = ""
        by_id[tc.id] = raw

    assembled: list[dict[str, Any]] = []
    for index, call_id in order:
        info = meta[(index, call_id)]
        assembled.append(
            {
                "index": index,
                "id": call_id,
                "type": info["type"],
                "function": {
                    "name": info["name"],
                    "arguments": by_id.get(call_id, ""),
                },
            }
        )
    return assembled


async def _decode_sse_chunks(
    raw_chunks: list[str],
    prefix: str,
    done_signal: str,
) -> tuple[int, bool]:
    done_received = False
    for chunk in raw_chunks:
        for line in chunk.splitlines():
            if not line.startswith(prefix):
                continue
            payload = line[len(prefix) :].strip()
            if payload == done_signal:
                done_received = True

    decoder = SSEDecoder(prefix=prefix, done_signal=done_signal)
    body = "".join(raw_chunks)

    async def byte_stream() -> AsyncIterator[bytes]:
        yield body.encode("utf-8")

    frame_count = 0
    async for _frame in decoder.decode(byte_stream()):
        frame_count += 1
    return frame_count, done_received


def decode_sse_chunks_sync(
    raw_chunks: list[str],
    prefix: str = "data: ",
    done_signal: str = "[DONE]",
) -> tuple[int, bool]:
    """Decode raw SSE chunks synchronously using production ``SSEDecoder``."""
    return asyncio.run(_decode_sse_chunks(raw_chunks, prefix, done_signal))
