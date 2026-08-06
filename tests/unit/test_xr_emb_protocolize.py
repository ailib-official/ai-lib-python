"""Unit tests for XR-EMB protocolized embeddings / rerank builders."""

from __future__ import annotations

import pytest

from ai_lib_python.embeddings.client import EmbeddingClient
from ai_lib_python.protocol.manifest import ProtocolManifest
from ai_lib_python.rerank.client import RerankerClient, _rerank_path_from_manifest
from ai_lib_python.stt.client import SttClient
from ai_lib_python.transport.http import HttpTransport
from ai_lib_python.tts.client import TtsClient


def _minimal_manifest(**extra: object) -> ProtocolManifest:
    data = {
        "id": "testprov",
        "protocol_version": "1.0",
        "endpoint": {
            "base_url": "https://example.test/v1",
            "auth": {"type": "bearer", "token_env": "TESTPROV_API_KEY"},
        },
        "capabilities": {"streaming": False, "tools": False, "vision": False},
        "status": "stable",
        "category": "ai_provider",
        "official_url": "https://example.test",
        "support_contact": "devnull",
        "parameter_mappings": {},
    }
    data.update(extra)
    return ProtocolManifest.model_validate(data)


@pytest.mark.asyncio
async def test_rerank_build_without_base_url_errors() -> None:
    with pytest.raises(ValueError, match="base_url required"):
        await RerankerClient.builder().model("rerank-v3").api_key("k").build()


@pytest.mark.asyncio
async def test_rerank_build_without_api_key_errors() -> None:
    with pytest.raises(ValueError, match="API key required"):
        await RerankerClient.builder().model("rerank-v3").base_url("https://example.test").build()


@pytest.mark.asyncio
async def test_rerank_from_manifest_uses_base_and_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TESTPROV_API_KEY", "secret-from-env")
    manifest = _minimal_manifest(endpoints={"rerank": {"path": "/v2/rerank", "method": "POST"}})
    client = await RerankerClient.builder().from_manifest(manifest, "rerank-english-v3").build()
    assert client.model == "rerank-english-v3"
    assert client._transport._base_url == "https://example.test/v1"
    assert client._endpoint_path == "/v2/rerank"
    assert type(client._transport).__name__ == "HttpTransport"


def test_rerank_path_fallback() -> None:
    assert _rerank_path_from_manifest(_minimal_manifest()) == "/rerank"


@pytest.mark.asyncio
async def test_stt_builder_uses_http_transport() -> None:
    client = await (
        SttClient.builder().model("whisper-1").api_key("k").base_url("https://api.openai.com").build()
    )
    assert isinstance(client._transport, HttpTransport)
    assert client._transport._base_url == "https://api.openai.com"


@pytest.mark.asyncio
async def test_tts_builder_uses_http_transport() -> None:
    client = await (
        TtsClient.builder().model("tts-1").api_key("k").base_url("https://api.openai.com").build()
    )
    assert isinstance(client._transport, HttpTransport)
    assert client._transport._base_url == "https://api.openai.com"


@pytest.mark.asyncio
async def test_rerank_explicit_uses_http_transport() -> None:
    client = await (
        RerankerClient.builder()
        .model("rerank-v3")
        .api_key("k")
        .base_url("https://example.test")
        .build()
    )
    assert isinstance(client._transport, HttpTransport)
    assert client._transport._base_url == "https://example.test"


@pytest.mark.asyncio
async def test_embedding_build_rejects_bare_model_id() -> None:
    with pytest.raises(ValueError, match="provider/model-id"):
        await EmbeddingClient.builder().model("text-embedding-3-small").api_key("k").build()


@pytest.mark.asyncio
async def test_embedding_from_manifest_sets_transport_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TESTPROV_API_KEY", "secret-from-env")
    manifest = _minimal_manifest(
        endpoints={"embeddings": {"path": "/custom/embeddings", "method": "POST"}}
    )
    client = await EmbeddingClient.builder().from_manifest(manifest, "emb-small").build()
    assert client.model == "emb-small"
    assert client._get_embedding_endpoint() == "/custom/embeddings"
