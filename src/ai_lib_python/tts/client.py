"""TTS 客户端：用于文本转语音调用。

TTS (Text-to-Speech) client.

HTTP uses shared [`HttpTransport`] — same stack as chat/embeddings ([GOV-007]).

When built ``from_manifest`` and the model declares ``text_to_speech``
(omit≠false), the builder prefers ``endpoints.text_to_speech`` (PT-GEN /
ALP-GEN-002). Prefer ``ai_lib_python.generative.TextToSpeechClient`` for new
hosts; this module remains the legacy entry with PT-GEN path convergence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ai_lib_python.transport import HttpTransport

if TYPE_CHECKING:
    from ai_lib_python.protocol.manifest import ProtocolManifest


def _normalize_endpoint_path(path: str) -> str:
    if path.startswith(("http://", "https://", "/")):
        return path
    return f"/{path}"


class AudioFormat(str, Enum):
    """Output audio format."""

    Mp3 = "mp3"
    Opus = "opus"
    Aac = "aac"
    Flac = "flac"
    Wav = "wav"
    Pcm = "pcm"

    @classmethod
    def from_str(cls, s: str) -> AudioFormat:
        m = {
            "mp3": cls.Mp3,
            "opus": cls.Opus,
            "aac": cls.Aac,
            "flac": cls.Flac,
            "wav": cls.Wav,
            "pcm": cls.Pcm,
        }
        return m.get(s.lower(), cls.Mp3)


@dataclass
class AudioOutput:
    """Audio output from TTS."""

    data: bytes
    format: AudioFormat = AudioFormat.Mp3


@dataclass
class TtsOptions:
    """Options for TTS synthesis."""

    voice: str | None = None
    speed: float | None = None
    response_format: str | None = None


class TtsClient:
    """Client for text-to-speech synthesis (e.g. OpenAI TTS)."""

    def __init__(
        self,
        *,
        transport: HttpTransport,
        model: str,
        endpoint_path: str = "/v1/audio/speech",
    ) -> None:
        self._transport = transport
        self._model = model
        self._endpoint_path = _normalize_endpoint_path(endpoint_path)

    @classmethod
    def builder(cls) -> TtsClientBuilder:
        """Get a builder for creating TTS clients."""
        return TtsClientBuilder()

    @property
    def endpoint_path(self) -> str:
        """Resolved request path (PT-GEN L-Exec or legacy default)."""
        return self._endpoint_path

    async def synthesize(self, text: str, options: TtsOptions | None = None) -> AudioOutput:
        """Synthesize text to audio."""
        opts = options or TtsOptions()
        body: dict[str, str | float] = {
            "model": self._model,
            "input": text,
        }
        if opts.voice:
            body["voice"] = opts.voice
        if opts.speed is not None:
            body["speed"] = opts.speed
        if opts.response_format:
            body["response_format"] = opts.response_format

        response = await self._transport.post(self._endpoint_path, json=body)
        data = response.content
        fmt = (
            AudioFormat.from_str(opts.response_format) if opts.response_format else AudioFormat.Mp3
        )
        return AudioOutput(data=data, format=fmt)

    @property
    def model(self) -> str:
        return self._model

    async def close(self) -> None:
        await self._transport.close()


class TtsClientBuilder:
    """Builder for TtsClient."""

    def __init__(self) -> None:
        self._model: str | None = None
        self._api_key: str | None = None
        self._base_url: str | None = None
        self._endpoint_path: str | None = None
        self._timeout: float = 60.0
        self._manifest: ProtocolManifest | None = None

    def model(self, model: str) -> TtsClientBuilder:
        self._model = model
        return self

    def api_key(self, api_key: str | None) -> TtsClientBuilder:
        self._api_key = api_key
        return self

    def base_url(self, url: str) -> TtsClientBuilder:
        self._base_url = url
        return self

    def endpoint_path(self, path: str) -> TtsClientBuilder:
        self._endpoint_path = path
        return self

    def timeout(self, timeout: float) -> TtsClientBuilder:
        self._timeout = timeout
        return self

    def from_manifest(self, manifest: ProtocolManifest, model_id: str) -> TtsClientBuilder:
        from ai_lib_python.transport.auth import resolve_credential

        resolved = resolve_credential(manifest.id, manifest, self._api_key)
        if not resolved.secret:
            raise ValueError(f"API key required for TTS (provider={manifest.id})")
        self._api_key = resolved.secret
        self._base_url = self._base_url or manifest.endpoint.base_url
        self._model = model_id
        self._manifest = manifest
        return self

    async def build(self) -> TtsClient:
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
                KEY_TEXT_TO_SPEECH,
                require_generative_endpoint,
            )

            if self._manifest.supports_generative_for_model(model, KEY_TEXT_TO_SPEECH):
                ep = require_generative_endpoint(self._manifest, model, KEY_TEXT_TO_SPEECH)
                endpoint_path = str(ep["path"])
        if endpoint_path is None:
            endpoint_path = "/v1/audio/speech"

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
        return TtsClient(transport=transport, model=model, endpoint_path=endpoint_path)
