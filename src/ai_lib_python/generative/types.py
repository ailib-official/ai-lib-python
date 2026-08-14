"""Experimental generative request/response types (ALP-GEN-001 / PT-GEN-001).

图像生成 / STT / TTS 的 Experimental 请求类型（HTTP 见 generative.image / audio）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ImageGenerationRequest:
    """Experimental text-to-image request (capability: ``image_generation``)."""

    model: str
    prompt: str
    size: str | None = None
    n: int | None = None
    response_format: str | None = None

    def with_size(self, size: str) -> ImageGenerationRequest:
        self.size = size
        return self


@dataclass
class GeneratedImage:
    """One generated image (URL and/or base64)."""

    url: str | None = None
    b64_json: str | None = None


@dataclass
class ImageGenerationResult:
    """Experimental image generation result."""

    model: str
    images: list[GeneratedImage] = field(default_factory=list)


@dataclass
class SpeechToTextRequest:
    """Experimental speech-to-text request (capability: ``speech_to_text``)."""

    model: str
    audio_source: str
    language: str | None = None
    prompt: str | None = None


@dataclass
class SpeechToTextResult:
    """Experimental speech-to-text result."""

    model: str
    text: str


@dataclass
class TextToSpeechRequest:
    """Experimental text-to-speech request (capability: ``text_to_speech``)."""

    model: str
    input: str
    voice: str | None = None
    response_format: str | None = None

    def with_voice(self, voice: str) -> TextToSpeechRequest:
        self.voice = voice
        return self


@dataclass
class TextToSpeechResult:
    """Experimental text-to-speech result."""

    model: str
    audio_base64: str | None = None
    content_type: str | None = None
