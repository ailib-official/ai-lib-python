"""STT 客户端：用于语音转文本调用。

STT (Speech-to-Text) client.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


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
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com",
        endpoint_path: str = "/v1/audio/transcriptions",
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._endpoint_path = endpoint_path if endpoint_path.startswith("/") else f"/{endpoint_path}"
        self._timeout = timeout

    @classmethod
    def builder(cls) -> SttClientBuilder:
        """Get a builder for creating STT clients."""
        return SttClientBuilder()

    async def transcribe(self, audio: bytes, options: SttOptions | None = None) -> Transcription:
        """Transcribe audio to text.

        Args:
            audio: Raw audio bytes (e.g. WAV, MP3)
            options: Optional transcription options

        Returns:
            Transcription result
        """
        opts = options or SttOptions()
        endpoint = f"{self._base_url}{self._endpoint_path}"
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

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                endpoint,
                files=files,
                data=data,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        response.raise_for_status()
        return Transcription.from_openai_format(response.json())

    @property
    def model(self) -> str:
        """Get the model identifier."""
        return self._model


class SttClientBuilder:
    """Builder for SttClient."""

    def __init__(self) -> None:
        self._model: str | None = None
        self._api_key: str | None = None
        self._base_url: str | None = None
        self._endpoint_path: str | None = None
        self._timeout: float = 60.0

    def model(self, model: str) -> SttClientBuilder:
        self._model = model
        return self

    def api_key(self, api_key: str | None) -> SttClientBuilder:
        self._api_key = api_key
        return self

    def base_url(self, url: str | None) -> SttClientBuilder:
        self._base_url = url
        return self

    def endpoint_path(self, path: str | None) -> SttClientBuilder:
        self._endpoint_path = path
        return self

    def timeout(self, timeout: float) -> SttClientBuilder:
        self._timeout = timeout
        return self

    async def build(self) -> SttClient:
        """Build the STT client."""
        model = self._model
        if not model:
            raise ValueError("Model must be specified")
        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key required (OPENAI_API_KEY)")
        base_url = self._base_url or "https://api.openai.com"
        endpoint_path = self._endpoint_path or "/v1/audio/transcriptions"
        return SttClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            endpoint_path=endpoint_path,
            timeout=self._timeout,
        )
