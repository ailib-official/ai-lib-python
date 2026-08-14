"""Experimental STT / TTS via manifest L-Exec (ALP-GEN-001/002 / ALR-GEN-002).

OpenAI adapter 形状；其他 adapter 显式配置错误（与 Rust ALR-GEN-002 对齐）。
Preferred host entry for PT-GEN audio. Legacy ``stt`` / ``tts`` builders
prefer these endpoint keys when ``model_capabilities`` declare them.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from ai_lib_python.errors import ValidationError
from ai_lib_python.generative.endpoints import (
    KEY_SPEECH_TO_TEXT,
    KEY_TEXT_TO_SPEECH,
    adapter_name,
    require_generative_endpoint,
)
from ai_lib_python.generative.types import (
    SpeechToTextRequest,
    SpeechToTextResult,
    TextToSpeechRequest,
    TextToSpeechResult,
)
from ai_lib_python.protocol.manifest import ProtocolManifest
from ai_lib_python.transport import HttpTransport
from ai_lib_python.transport.auth import resolve_credential


def _require_openai_adapter(ep: dict[str, Any], capability: str) -> str:
    name = adapter_name(ep)
    if name != "openai":
        raise ValidationError(
            f"{capability} adapter `{name}` not implemented in ALP-GEN-001 (openai only)"
        )
    return name


def _transport_for(manifest: ProtocolManifest, model: str, capability: str) -> HttpTransport:
    resolved = resolve_credential(manifest.id, manifest, None)
    if not resolved.secret:
        tried = list(resolved.required_envs) + list(resolved.conventional_envs)
        raise ValidationError(
            f"API key required for {capability} (provider={manifest.id}; tried {tried})"
        )
    return HttpTransport(manifest, model_id=model, api_key=resolved.secret)


class SpeechToTextClient:
    """Experimental speech-to-text client (capability: ``speech_to_text``)."""

    def __init__(
        self,
        *,
        transport: HttpTransport,
        model: str,
        endpoint_path: str,
        adapter: str,
    ) -> None:
        self._transport = transport
        self._model = model
        self._endpoint_path = endpoint_path
        self._adapter = adapter

    @classmethod
    def from_manifest(cls, manifest: ProtocolManifest, model: str) -> SpeechToTextClient:
        ep = require_generative_endpoint(manifest, model, KEY_SPEECH_TO_TEXT)
        adapter = _require_openai_adapter(ep, "speech_to_text")
        return cls(
            transport=_transport_for(manifest, model, "speech_to_text"),
            model=model,
            endpoint_path=str(ep["path"]),
            adapter=adapter,
        )

    @property
    def endpoint_path(self) -> str:
        return self._endpoint_path

    @property
    def adapter(self) -> str:
        return self._adapter

    async def transcribe(self, request: SpeechToTextRequest) -> SpeechToTextResult:
        if request.model != self._model:
            raise ValidationError(
                f"request model `{request.model}` != client model `{self._model}`"
            )
        path = Path(request.audio_source)
        data = path.read_bytes()
        files = {"file": (path.name or "audio.wav", data, "application/octet-stream")}
        form: dict[str, str] = {"model": self._model}
        if request.language:
            form["language"] = request.language
        if request.prompt:
            form["prompt"] = request.prompt
        response = await self._transport.post(self._endpoint_path, files=files, data=form)
        payload = response.json()
        text = payload.get("text") if isinstance(payload, dict) else ""
        return SpeechToTextResult(model=self._model, text=text if isinstance(text, str) else "")


class TextToSpeechClient:
    """Experimental text-to-speech client (capability: ``text_to_speech``)."""

    def __init__(
        self,
        *,
        transport: HttpTransport,
        model: str,
        endpoint_path: str,
        adapter: str,
    ) -> None:
        self._transport = transport
        self._model = model
        self._endpoint_path = endpoint_path
        self._adapter = adapter

    @classmethod
    def from_manifest(cls, manifest: ProtocolManifest, model: str) -> TextToSpeechClient:
        ep = require_generative_endpoint(manifest, model, KEY_TEXT_TO_SPEECH)
        adapter = _require_openai_adapter(ep, "text_to_speech")
        return cls(
            transport=_transport_for(manifest, model, "text_to_speech"),
            model=model,
            endpoint_path=str(ep["path"]),
            adapter=adapter,
        )

    @property
    def endpoint_path(self) -> str:
        return self._endpoint_path

    @property
    def adapter(self) -> str:
        return self._adapter

    async def synthesize(self, request: TextToSpeechRequest) -> TextToSpeechResult:
        if request.model != self._model:
            raise ValidationError(
                f"request model `{request.model}` != client model `{self._model}`"
            )
        body: dict[str, str] = {"model": self._model, "input": request.input}
        if request.voice:
            body["voice"] = request.voice
        if request.response_format:
            body["response_format"] = request.response_format
        response = await self._transport.post(self._endpoint_path, json=body)
        raw = response.content
        ctype = response.headers.get("content-type")
        return TextToSpeechResult(
            model=self._model,
            audio_base64=base64.b64encode(raw).decode("ascii") if raw else None,
            content_type=ctype,
        )
