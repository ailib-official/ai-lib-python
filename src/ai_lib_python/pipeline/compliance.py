"""Conformance helpers for ai-protocol compliance YAML runners.

合规测试辅助：SSE 解码、event_map / tool 累积、协议加载与消息构建。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


@dataclass(frozen=True)
class ProtocolLoadingResult:
    """Outcome of a compliance protocol_loading evaluation."""

    valid: bool
    manifest: dict[str, Any] | None
    errors: list[str]


def manifest_has_required_shape(manifest: dict[str, Any]) -> bool:
    """Minimal required shape for protocol_loading compliance."""
    if not isinstance(manifest.get("id"), str) or not manifest.get("id"):
        return False
    if not isinstance(manifest.get("protocol_version"), str) or not manifest.get(
        "protocol_version"
    ):
        return False
    endpoint = manifest.get("endpoint")
    if not isinstance(endpoint, dict):
        return False
    base_url = endpoint.get("base_url")
    return isinstance(base_url, str) and bool(base_url)


def capability_profile_phase_errors(manifest: dict[str, Any]) -> list[str]:
    """Return staged capability_profile semantic errors for IOS/IOSPC phases."""
    cp = manifest.get("capability_profile")
    if cp is None:
        return []
    if not isinstance(cp, dict):
        return ["capability_profile must be object"]

    errors: list[str] = []
    phase = cp.get("phase")
    if phase == "ios_v1":
        if "process" in cp or "contract" in cp:
            errors.append("must NOT have additional properties")
        if not any(key in cp for key in ("inputs", "outcomes", "systems")):
            errors.append("must match at least one schema in anyOf")
    elif phase == "iospc_v1":
        if not any(key in cp for key in ("inputs", "outcomes", "systems")):
            errors.append("iospc_v1 requires inputs or outcomes or systems")
        if "process" not in cp and "contract" not in cp:
            errors.append("iospc_v1 requires process or contract")
    elif phase is not None:
        errors.append("phase must be ios_v1 or iospc_v1")
    return errors


def compliance_load_manifest_file(path: Path) -> ProtocolLoadingResult:
    """Load and validate a manifest file using production protocol APIs."""
    from ai_lib_python.protocol.loader import ProtocolLoader
    from ai_lib_python.protocol.manifest import ProtocolManifest
    from ai_lib_python.protocol.v2 import ManifestV2

    loader = ProtocolLoader(fallback_to_github=False)
    try:
        manifest = loader.load_file(path)
    except Exception as exc:
        return ProtocolLoadingResult(valid=False, manifest=None, errors=[str(exc)])

    cp_errors = capability_profile_phase_errors(manifest)
    if cp_errors:
        return ProtocolLoadingResult(valid=False, manifest=manifest, errors=cp_errors)

    if not manifest_has_required_shape(manifest):
        return ProtocolLoadingResult(
            valid=False,
            manifest=manifest,
            errors=["missing required manifest shape"],
        )

    version = str(manifest.get("protocol_version", ""))
    try:
        if (
            version.startswith("2")
            or manifest.get("capability_profile") is not None
            or manifest.get("core") is not None
        ):
            ManifestV2.model_validate(manifest)
        else:
            ProtocolManifest.model_validate(manifest)
    except Exception as exc:
        return ProtocolLoadingResult(
            valid=False,
            manifest=manifest,
            errors=[f"runtime_deserialize: {exc}"],
        )

    return ProtocolLoadingResult(valid=True, manifest=manifest, errors=[])


def compliance_normalize_message_body(messages: list[Any]) -> dict[str, Any]:
    """Build normalized request body for message_building compliance cases."""
    normalized_messages = [msg for msg in messages if isinstance(msg, dict)]
    return {"messages": normalized_messages}


def compliance_events_from_openai_frame(frame: dict[str, Any]) -> list[dict[str, Any]]:
    """Map a decoded OpenAI chat frame to compliance event records."""
    choices = frame.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    choice = choices[0] if isinstance(choices[0], dict) else {}
    delta_raw = choice.get("delta")
    delta = delta_raw if isinstance(delta_raw, dict) else {}

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
        function_raw = chunk.get("function")
        function = function_raw if isinstance(function_raw, dict) else {}
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
