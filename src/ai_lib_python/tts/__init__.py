"""TTS 模块：封装文本转语音能力。

TTS (Text-to-Speech) module.

Provides synthesis of text to audio via provider APIs (e.g. OpenAI TTS).
For PT-GEN hosts prefer ``ai_lib_python.generative.TextToSpeechClient``;
``from_manifest`` builders converge onto ``endpoints.text_to_speech`` when declared.
"""

from ai_lib_python.tts.client import (
    AudioFormat,
    AudioOutput,
    TtsClient,
    TtsClientBuilder,
    TtsOptions,
)

__all__ = [
    "AudioFormat",
    "AudioOutput",
    "TtsClient",
    "TtsClientBuilder",
    "TtsOptions",
]
