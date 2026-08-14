"""STT 客户端：用于语音转文本调用。

STT (Speech-to-Text) client.

HTTP uses shared [`HttpTransport`] — same stack as chat/embeddings ([GOV-007]).

When built ``from_manifest`` and the model declares ``speech_to_text``
(omit≠false), the builder prefers ``endpoints.speech_to_text`` (PT-GEN /
ALP-GEN-002). Prefer ``ai_lib_python.generative.SpeechToTextClient`` for new
hosts; this module remains the legacy entry with PT-GEN path convergence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ai_lib_python.transport import HttpTransport

if TYPE_CHECKING:
    from ai_lib_python.protocol.manifest import ProtocolManifest


def _normalize_endpoint_path(path: str) -> str:
    if path.startswith(("http://", "https://", "/")):
        return path
    return f"/{path}"


@dataclass
class TranscriptionSegment:
    """A single segment of transcription."""

    id: int
    start: float
    end: float
    text: str


@dataclass
class Transcription:
    """Transcription result from STT."""

    text: str
    language: str | None = None
    confidence: float | None = None
    segments: list[TranscriptionSegment] | None = None

    @classmethod
    def from_openai_format(cls, data: dict[str, Any]) -> Transcription:
        text = data.get("text", "")
        language = data.get("language")
        segments_data = data.get("segments", [])
        segments = None
        if segments_data:
            segments = [
                TranscriptionSegment(
                    id=s.get("id", 0),
                    start=s.get("start", 0.0),
                    end=s.get("end", 0.0),
                    text=s.get("text", ""),
                )
                for s in segments_data
            ]
        return cls(text=text, language=language, confidence=None, segments=segments)


@dataclass
class SttOptions:
    """Options for STT transcription."""

    language: str | None = None
    prompt: str | None = None
    temperature: float | None = None
    response_format: str | None = None


class SttClient:
    """Client for speech-to-text transcription (e.g. OpenAI Whisper)."""

    def __init__(
        self,
        *,
        transport: HttpTransport,
        model: str,
        endpoint_path: str = "/v1/audio/transcriptions",
    ) -> None:
        self._transport = transport
        self._model = model
        self._endpoint_path = _normalize_endpoint_path(endpoint_path)

    @classmethod
    def builder(cls) -> SttClientBuilder:
        """Get a builder for creating STT clients."""
        return SttClientBuilder()

    @property
    def endpoint_path(self) -> str:
        """Resolved request path (PT-GEN L-Exec or legacy default)."""
        return self._endpoint_path

    async def transcribe(self, audio: bytes, options: SttOptions | None = None) -> Transcription:
        """Transcribe audio to text."""
        opts = options or SttOptions()
        files = {"file": ("audio.wav", audio, "audio/wav")}
        data: dict[str, str] = {"model": self._model}
        if opts.language:
            data["language"] = opts.language
        if opts.prompt:
            data["prompt"] = opts.prompt
        if opts.temperature is not None:
            data["temperature"] = str(opts.temperature)
        if opts.response_format:
            data["response_format"] = opts.response_format

        response = await self._transport.post(
            self._endpoint_path,
            files=files,
            data=data,
        )
        return Transcription.from_openai_format(response.json())

    @property
    def model(self) -> str:
        """Get the model identifier."""
        return self._model

    async def close(self) -> None:
        await self._transport.close()


class SttClientBuilder:
    """Builder for SttClient."""

    def __init__(self) -> None:
        self._model: str | None = None
        self._api_key: str | None = None
        self._base_url: str | None = None
        self._endpoint_path: str | None = None
        self._timeout: float = 60.0
        self._manifest: ProtocolManifest | None = None

    def model(self, model: str) -> SttClientBuilder:
        self._model = model
        return self

    def api_key(self, api_key: str | None) -> SttClientBuilder:
        self._api_key = api_key
        return self

    def base_url(self, url: str) -> SttClientBuilder:
        self._base_url = url
        return self

    def endpoint_path(self, path: str) -> SttClientBuilder:
        self._endpoint_path = path
        return self

    def timeout(self, timeout: float) -> SttClientBuilder:
        self._timeout = timeout
        return self

    def from_manifest(self, manifest: ProtocolManifest, model_id: str) -> SttClientBuilder:
        from ai_lib_python.transport.auth import resolve_credential

        resolved = resolve_credential(manifest.id, manifest, self._api_key)
        if not resolved.secret:
            raise ValueError(f"API key required for STT (provider={manifest.id})")
        self._api_key = resolved.secret
        self._base_url = self._base_url or manifest.endpoint.base_url
        self._model = model_id
        self._manifest = manifest
        return self

    async def build(self) -> SttClient:
        model = self._model
        if not model:
            raise ValueError("Model must be specified")
        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key required")
        base_url = self._base_url or "https://api.openai.com"
        endpoint_path = self._endpoint_path
        if endpoint_path is None and self._manifest is not None:
            from ai_lib_python.generative.endpoints import (
                KEY_SPEECH_TO_TEXT,
                require_generative_endpoint,
            )

            if self._manifest.supports_generative_for_model(model, KEY_SPEECH_TO_TEXT):
                ep = require_generative_endpoint(self._manifest, model, KEY_SPEECH_TO_TEXT)
                endpoint_path = str(ep["path"])
        if endpoint_path is None:
            endpoint_path = "/v1/audio/transcriptions"

        if self._manifest is not None:
            transport = HttpTransport(
                manifest=self._manifest,
                model_id=model,
                api_key=api_key,
                base_url_override=base_url,
                timeout=self._timeout,
            )
        else:
            transport = HttpTransport.with_explicit_bearer(
                base_url=base_url,
                api_key=api_key,
                model_id=model,
                timeout=self._timeout,
            )
        return SttClient(transport=transport, model=model, endpoint_path=endpoint_path)
