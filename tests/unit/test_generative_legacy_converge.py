"""ALP-GEN-002: legacy stt/tts prefer PT-GEN when declared.

Legacy builders：manifest 声明 speech_to_text / text_to_speech 时走 L-Exec path；
未声明则保留 OpenAI 默认 path。显式 endpoint_path 优先。
"""

from __future__ import annotations

import pytest

from ai_lib_python.protocol.manifest import ProtocolManifest
from ai_lib_python.stt import SttClient
from ai_lib_python.tts import TtsClient


def _audio_manifest(*, stt: bool = True, tts: bool = True) -> ProtocolManifest:
    models: dict = {}
    endpoints: dict = {}
    caps = ["text"]
    if stt:
        models["whisper-1"] = {"model_capabilities": {"speech_to_text": True}}
        endpoints["speech_to_text"] = {
            "path": "/audio/transcriptions",
            "method": "POST",
            "adapter": "openai",
        }
        caps.append("speech_to_text")
    if tts:
        models["tts-1"] = {"model_capabilities": {"text_to_speech": True}}
        endpoints["text_to_speech"] = {
            "path": "/audio/speech",
            "method": "POST",
            "adapter": "openai",
        }
        caps.append("text_to_speech")
    return ProtocolManifest.model_validate(
        {
            "id": "openai",
            "protocol_version": "2.0",
            "status": "stable",
            "endpoint": {"base_url": "https://api.openai.com/v1"},
            "capabilities": caps,
            "endpoints": endpoints,
            "metadata": {"models": models},
        }
    )


@pytest.mark.asyncio
async def test_stt_from_manifest_prefers_pt_gen_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    m = _audio_manifest(stt=True, tts=False)
    client = await SttClient.builder().from_manifest(m, "whisper-1").build()
    assert client.endpoint_path == "/audio/transcriptions"
    await client.close()


@pytest.mark.asyncio
async def test_stt_without_pt_gen_keeps_legacy_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    m = ProtocolManifest.model_validate(
        {
            "id": "openai",
            "protocol_version": "2.0",
            "status": "stable",
            "endpoint": {"base_url": "https://api.openai.com/v1"},
            "capabilities": ["text", "stt"],
            "metadata": {"models": {"whisper-1": {"context_window": 1}}},
        }
    )
    client = await SttClient.builder().from_manifest(m, "whisper-1").build()
    assert client.endpoint_path == "/v1/audio/transcriptions"
    await client.close()


@pytest.mark.asyncio
async def test_stt_explicit_endpoint_overrides_pt_gen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    m = _audio_manifest(stt=True, tts=False)
    client = (
        await SttClient.builder().from_manifest(m, "whisper-1").endpoint_path("/custom/stt").build()
    )
    assert client.endpoint_path == "/custom/stt"
    await client.close()


@pytest.mark.asyncio
async def test_tts_from_manifest_prefers_pt_gen_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    m = _audio_manifest(stt=False, tts=True)
    client = await TtsClient.builder().from_manifest(m, "tts-1").build()
    assert client.endpoint_path == "/audio/speech"
    await client.close()


def test_package_exports_generative() -> None:
    import ai_lib_python as alp

    assert hasattr(alp, "SpeechToTextClient")
    assert hasattr(alp, "TextToSpeechClient")
    assert hasattr(alp, "ImageGenerationClient")
    assert "SpeechToTextClient" in alp.__all__
