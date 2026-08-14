"""AI-Protocol 官方 Python 运行时：提供统一的多厂商 AI 模型交互接口。

ai-lib-python: Official Python Runtime for AI-Protocol.

The canonical Pythonic implementation for unified AI model interaction.
Core principle: All logic is operators, all configuration is protocol.
"""

from __future__ import annotations

from ai_lib_python._features import (
    HAS_AUDIO,
    HAS_KEYRING,
    HAS_TELEMETRY,
    HAS_TOKENIZER,
    HAS_VISION,
    HAS_WATCHDOG,
    require_extra,
)
from ai_lib_python._version import __version__
from ai_lib_python.client import AiClient, AiClientBuilder, CallStats, ChatResponse
from ai_lib_python.errors import AiLibError, ProtocolError, TransportError
from ai_lib_python.generative import (
    KEY_IMAGE_GENERATION,
    KEY_SPEECH_TO_TEXT,
    KEY_TEXT_TO_SPEECH,
    GeneratedImage,
    ImageGenerationClient,
    ImageGenerationRequest,
    ImageGenerationResult,
    SpeechToTextClient,
    SpeechToTextRequest,
    SpeechToTextResult,
    TextToSpeechClient,
    TextToSpeechRequest,
    TextToSpeechResult,
    require_generative_endpoint,
    resolve_generative_endpoint,
)
from ai_lib_python.types.events import StreamingEvent
from ai_lib_python.types.message import (
    ContentBlock,
    Message,
    MessageContent,
    MessageRole,
)
from ai_lib_python.types.tool import ToolCall, ToolDefinition

__all__ = [
    # Client
    "AiClient",
    "AiClientBuilder",
    # Feature flags
    "HAS_AUDIO",
    "HAS_KEYRING",
    "HAS_TELEMETRY",
    "HAS_TOKENIZER",
    "HAS_VISION",
    "HAS_WATCHDOG",
    "require_extra",
    # Errors
    "AiLibError",
    "CallStats",
    "ChatResponse",
    "ContentBlock",
    # Experimental generative (ALP-GEN-001/002) — not a stable AiClient facade
    "GeneratedImage",
    "ImageGenerationClient",
    "ImageGenerationRequest",
    "ImageGenerationResult",
    "KEY_IMAGE_GENERATION",
    "KEY_SPEECH_TO_TEXT",
    "KEY_TEXT_TO_SPEECH",
    "SpeechToTextClient",
    "SpeechToTextRequest",
    "SpeechToTextResult",
    "TextToSpeechClient",
    "TextToSpeechRequest",
    "TextToSpeechResult",
    "require_generative_endpoint",
    "resolve_generative_endpoint",
    # Types - Message
    "Message",
    "MessageContent",
    "MessageRole",
    "ProtocolError",
    # Types - Events
    "StreamingEvent",
    "ToolCall",
    # Types - Tool
    "ToolDefinition",
    "TransportError",
    # Version
    "__version__",
]
