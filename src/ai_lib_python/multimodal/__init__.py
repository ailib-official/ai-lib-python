"""扩展多模态处理模块 — 提供跨厂商的输入/输出模态验证与格式转换。

Extended multimodal processing module for AI-Protocol V2.
Provides content format validation, modality detection, and
provider-specific formatting helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Modality(str, Enum):
    """Supported input/output modality types."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MultimodalCapabilities:
    """Describes the multimodal capabilities of a provider, derived from the manifest."""

    input_modalities: set[Modality] = field(default_factory=lambda: {Modality.TEXT})
    output_modalities: set[Modality] = field(default_factory=lambda: {Modality.TEXT})
    image_formats: list[str] = field(default_factory=list)
    audio_formats: list[str] = field(default_factory=list)
    video_formats: list[str] = field(default_factory=list)
    max_image_size: str | None = None
    max_audio_duration: str | None = None
    supports_omni: bool = False
    supports_realtime_voice: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> MultimodalCapabilities:
        """Build from a V2 multimodal config section (dict form)."""
        caps = cls()
        if not config:
            return caps

        inp = config.get("input", {})
        if inp:
            vision = inp.get("vision", {})
            if vision.get("supported"):
                caps.input_modalities.add(Modality.IMAGE)
                caps.image_formats = vision.get("formats", [])
                caps.max_image_size = vision.get("max_file_size")

            audio = inp.get("audio", {})
            if audio.get("supported"):
                caps.input_modalities.add(Modality.AUDIO)
                caps.audio_formats = audio.get("formats", [])

            video = inp.get("video") or {}
            if video.get("supported"):
                caps.input_modalities.add(Modality.VIDEO)
                caps.video_formats = video.get("formats", [])

        out = config.get("output", {})
        if out:
            audio_out = out.get("audio") or {}
            if audio_out.get("supported"):
                caps.output_modalities.add(Modality.AUDIO)
            image_out = out.get("image") or {}
            if image_out.get("supported"):
                caps.output_modalities.add(Modality.IMAGE)
            video_out = out.get("video") or {}
            if video_out.get("supported"):
                caps.output_modalities.add(Modality.VIDEO)
                if not caps.video_formats:
                    caps.video_formats = video_out.get("formats", [])

        omni = config.get("omni_mode", {})
        if omni:
            caps.supports_omni = omni.get("supported", False)
            caps.supports_realtime_voice = omni.get("real_time_voice_chat", False)

        return caps

    def supports_input(self, modality: Modality) -> bool:
        return modality in self.input_modalities

    def supports_output(self, modality: Modality) -> bool:
        return modality in self.output_modalities

    def validate_image_format(self, fmt: str) -> bool:
        if not self.image_formats:
            return True
        return fmt.lower() in (f.lower() for f in self.image_formats)

    def validate_audio_format(self, fmt: str) -> bool:
        if not self.audio_formats:
            return True
        return fmt.lower() in (f.lower() for f in self.audio_formats)

    def validate_video_format(self, fmt: str) -> bool:
        if not self.video_formats:
            return True
        return fmt.lower() in (f.lower() for f in self.video_formats)


def detect_modalities(content_blocks: list[dict[str, Any]]) -> set[Modality]:
    """Detect the modalities present in a list of content blocks."""
    modalities: set[Modality] = set()
    _type_map = {
        "text": Modality.TEXT,
        "image": Modality.IMAGE,
        "image_url": Modality.IMAGE,
        "audio": Modality.AUDIO,
        "input_audio": Modality.AUDIO,
        "video": Modality.VIDEO,
    }
    for block in content_blocks:
        block_type = block.get("type", "")
        if block_type in _type_map:
            modalities.add(_type_map[block_type])
    if not modalities:
        modalities.add(Modality.TEXT)
    return modalities


def validate_content_modalities(
    blocks: list[dict[str, Any]],
    caps: MultimodalCapabilities,
) -> list[Modality]:
    """Validate content blocks against capabilities.

    Returns a list of unsupported modalities (empty = valid).
    """
    detected = detect_modalities(blocks)
    return [m for m in detected if not caps.supports_input(m)]


__all__ = [
    "Modality",
    "MultimodalCapabilities",
    "detect_modalities",
    "validate_content_modalities",
]
