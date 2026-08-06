"""重排序客户端：用于文档相关性评分。

Rerank client for document relevance scoring.

HTTP uses shared [`HttpTransport`] — same stack as chat/embeddings ([GOV-007]).
Aligned with XR-EMB-PROTOCOLIZE-CONTRACT / ai-lib-rust:
base_url + path + credentials from protocol manifests or explicit overrides —
no silent Cohere host default ([ARCH-001]).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_lib_python.protocol.loader import ProtocolLoader
from ai_lib_python.transport import HttpTransport
from ai_lib_python.transport.auth import resolve_credential

if TYPE_CHECKING:
    from ai_lib_python.protocol.manifest import ProtocolManifest


@dataclass
class RerankResult:
    """A single rerank result."""

    index: int
    relevance_score: float
    document: str | None = None


@dataclass
class RerankOptions:
    """Options for reranking."""

    top_n: int | None = None
    max_tokens_per_doc: int | None = None


def _rerank_path_from_manifest(manifest: ProtocolManifest) -> str:
    """Resolve rerank path from endpoints/services; else `/rerank`."""
    ep = manifest.endpoints.get("rerank") if manifest.endpoints else None
    if isinstance(ep, dict):
        path = ep.get("path")
        if isinstance(path, str) and path.strip():
            return path if path.startswith("/") else f"/{path}"
    svc = None
    if manifest.services:
        svc = manifest.services.get("rerank")
        if hasattr(svc, "path"):
            path = getattr(svc, "path", None)
            if isinstance(path, str) and path.strip():
                return path if path.startswith("/") else f"/{path}"
        if isinstance(svc, dict):
            path = svc.get("path")
            if isinstance(path, str) and path.strip():
                return path if path.startswith("/") else f"/{path}"
    return "/rerank"


class RerankerClient:
    """Client for document reranking (manifest- or explicitly configured)."""

    def __init__(
        self,
        *,
        transport: HttpTransport,
        model: str,
        endpoint_path: str = "/rerank",
    ) -> None:
        self._transport = transport
        self._model = model
        self._endpoint_path = (
            endpoint_path if endpoint_path.startswith("/") else f"/{endpoint_path}"
        )

    @classmethod
    def builder(cls) -> RerankerClientBuilder:
        """Get a builder for creating Reranker clients."""
        return RerankerClientBuilder()

    async def rerank(
        self,
        query: str,
        documents: list[str],
        options: RerankOptions | None = None,
    ) -> list[RerankResult]:
        """Rerank documents by relevance to query."""
        opts = options or RerankOptions()
        body: dict[str, str | int | list[str]] = {
            "model": self._model,
            "query": query,
            "documents": documents,
        }
        if opts.top_n is not None:
            body["top_n"] = opts.top_n
        if opts.max_tokens_per_doc is not None:
            body["max_tokens_per_doc"] = opts.max_tokens_per_doc

        response = await self._transport.post(self._endpoint_path, json=body)
        data = response.json()
        results = data.get("results", [])
        return [
            RerankResult(
                index=r.get("index", 0),
                relevance_score=float(r.get("relevance_score", 0.0)),
                document=r.get("document"),
            )
            for r in results
        ]

    @property
    def model(self) -> str:
        return self._model

    async def close(self) -> None:
        await self._transport.close()


class RerankerClientBuilder:
    """Builder for RerankerClient (XR-EMB contract)."""

    def __init__(self) -> None:
        self._model: str | None = None
        self._api_key: str | None = None
        self._base_url: str | None = None
        self._endpoint_path: str | None = None
        self._timeout: float = 60.0
        self._protocol_path: str | None = None
        self._manifest: ProtocolManifest | None = None

    def model(self, model: str) -> RerankerClientBuilder:
        self._model = model
        return self

    def api_key(self, api_key: str | None) -> RerankerClientBuilder:
        self._api_key = api_key
        return self

    def base_url(self, url: str | None) -> RerankerClientBuilder:
        self._base_url = url
        return self

    def endpoint_path(self, path: str | None) -> RerankerClientBuilder:
        self._endpoint_path = path
        return self

    def timeout(self, timeout: float) -> RerankerClientBuilder:
        self._timeout = timeout
        return self

    def protocol_path(self, path: str) -> RerankerClientBuilder:
        self._protocol_path = path
        return self

    def from_manifest(
        self,
        manifest: ProtocolManifest,
        model_id: str,
    ) -> RerankerClientBuilder:
        """Apply base_url / path / credential from an already-loaded manifest."""
        resolved = resolve_credential(manifest.id, manifest, self._api_key)
        if not resolved.secret:
            tried = list(resolved.required_envs) + list(resolved.conventional_envs)
            raise ValueError(f"API key required for rerank (provider={manifest.id}; tried {tried})")
        self._api_key = resolved.secret
        self._base_url = manifest.endpoint.base_url
        if self._endpoint_path is None:
            self._endpoint_path = _rerank_path_from_manifest(manifest)
        self._model = model_id
        self._manifest = manifest
        return self

    async def from_model(self, model: str) -> RerankerClient:
        """Load `provider/model-id` via ProtocolLoader then build."""
        parts = model.split("/")
        if len(parts) < 2:
            raise ValueError("Model must be provider/model-id form")
        model_id = "/".join(parts[1:])
        loader = (
            ProtocolLoader(base_path=self._protocol_path)
            if self._protocol_path
            else ProtocolLoader()
        )
        manifest = await loader.load_model(model)
        return await self.from_manifest(manifest, model_id).build()

    async def build(self) -> RerankerClient:
        model = self._model
        if not model:
            raise ValueError("Model must be specified")
        api_key = self._api_key
        if not api_key:
            raise ValueError(
                "API key required: use from_manifest/from_model or set api_key explicitly"
            )
        base_url = self._base_url
        if not base_url:
            raise ValueError(
                "base_url required: use from_manifest/from_model or set base_url "
                "explicitly (no vendor default)"
            )
        endpoint_path = self._endpoint_path or "/rerank"

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
        return RerankerClient(
            transport=transport,
            model=model,
            endpoint_path=endpoint_path,
        )
