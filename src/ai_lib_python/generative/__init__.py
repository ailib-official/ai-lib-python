"""Experimental generative L-Exec API (ALP-GEN-001).

生成式能力：同键校验 + 最小 Image/STT/TTS 客户端；HTTP 仅走 HttpTransport。
"""

from ai_lib_python.generative.audio import SpeechToTextClient, TextToSpeechClient
from ai_lib_python.generative.endpoints import (
    KEY_IMAGE_GENERATION,
    KEY_SPEECH_TO_TEXT,
    KEY_TEXT_TO_SPEECH,
    require_generative_endpoint,
    resolve_generative_endpoint,
)
from ai_lib_python.generative.image import ImageGenerationClient
from ai_lib_python.generative.types import (
    GeneratedImage,
    ImageGenerationRequest,
    ImageGenerationResult,
    SpeechToTextRequest,
    SpeechToTextResult,
    TextToSpeechRequest,
    TextToSpeechResult,
)

__all__ = [
    "KEY_IMAGE_GENERATION",
    "KEY_SPEECH_TO_TEXT",
    "KEY_TEXT_TO_SPEECH",
    "GeneratedImage",
    "ImageGenerationClient",
    "ImageGenerationRequest",
    "ImageGenerationResult",
    "SpeechToTextClient",
    "SpeechToTextRequest",
    "SpeechToTextResult",
    "TextToSpeechClient",
    "TextToSpeechRequest",
    "TextToSpeechResult",
    "require_generative_endpoint",
    "resolve_generative_endpoint",
]
