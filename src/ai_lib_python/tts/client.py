"""TTS 客户端：用于文本转语音调用。

TTS (Text-to-Speech) client.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

import httpx


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
        m = {"mp3": cls.Mp3, "opus": cls.Opus, "aac": cls.Aac, "flac": cls.Flac, "wav": cls.Wav, "pcm": cls.Pcm}
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
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com",
        endpoint_path: str = "/v1/audio/speech",
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._endpoint_path = endpoint_path if endpoint_path.startswith("/") else f"/{endpoint_path}"
        self._timeout = timeout

    @classmethod
    def builder(cls) -> TtsClientBuilder:
        """Get a builder for creating TTS clients."""
        return TtsClientBuilder()

    async def synthesize(self, text: str, options: TtsOptions | None = None) -> AudioOutput:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize
            options: Optional TTS options

        Returns:
            AudioOutput with raw bytes and format
        """
        opts = options or TtsOptions()
        endpoint = f"{self._base_url}{self._endpoint_path}"
        body: dict[str, str | float] = {
            "model": self.model,
            "input": text,
        }
        if opts.voice:
            body["voice"] = opts.voice
        if opts.speed is not None:
            body["speed"] = opts.speed
        if opts.response_format:
            body["response_format"] = opts.response_format

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                endpoint,
                json=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        response.raise_for_status()
        data = response.content
        fmt = (
            AudioFormat.from_str(opts.response_format)
            if opts.response_format
            else AudioFormat.Mp3
        )
        return AudioOutput(data=data, format=fmt)

    @property
    def model(self) -> str:
        """Get the model identifier."""
        return self._model


class TtsClientBuilder:
    """Builder for TtsClient."""

    def __init__(self) -> None:
        self._model: str | None = None
        self._api_key: str | None = None
        self._base_url: str | None = None
        self._endpoint_path: str | None = None
        self._timeout: float = 60.0

    def model(self, model: str) -> TtsClientBuilder:
        self._model = model
        return self

    def api_key(self, api_key: str | None) -> TtsClientBuilder:
        self._api_key = api_key
        return self

    def base_url(self, url: str | None) -> TtsClientBuilder:
        self._base_url = url
        return self

    def endpoint_path(self, path: str | None) -> TtsClientBuilder:
        self._endpoint_path = path
        return self

    def timeout(self, timeout: float) -> TtsClientBuilder:
        self._timeout = timeout
        return self

    async def build(self) -> TtsClient:
        """Build the TTS client."""
        model = self._model
        if not model:
            raise ValueError("Model must be specified")
        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key required (OPENAI_API_KEY)")
        base_url = self._base_url or "https://api.openai.com"
        endpoint_path = self._endpoint_path or "/v1/audio/speech"
        return TtsClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            endpoint_path=endpoint_path,
            timeout=self._timeout,
        )
