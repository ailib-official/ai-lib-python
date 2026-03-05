"""重排序客户端：用于文档相关性评分。

Rerank client for document relevance scoring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


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


class RerankerClient:
    """Client for document reranking (e.g. Cohere Rerank)."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.cohere.com/v2",
        endpoint_path: str = "/rerank",
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._endpoint_path = endpoint_path if endpoint_path.startswith("/") else f"/{endpoint_path}"
        self._timeout = timeout

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
        endpoint = f"{self._base_url}{self._endpoint_path}"
        body: dict[str, str | int | list[str]] = {
            "model": self._model,
            "query": query,
            "documents": documents,
        }
        if opts.top_n is not None:
            body["top_n"] = opts.top_n
        if opts.max_tokens_per_doc is not None:
            body["max_tokens_per_doc"] = opts.max_tokens_per_doc

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


class RerankerClientBuilder:
    """Builder for RerankerClient."""

    def __init__(self) -> None:
        self._model: str | None = None
        self._api_key: str | None = None
        self._base_url: str | None = None
        self._endpoint_path: str | None = None
        self._timeout: float = 60.0

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

    async def build(self) -> RerankerClient:
        model = self._model
        if not model:
            raise ValueError("Model must be specified")
        api_key = self._api_key or os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise ValueError("API key required (COHERE_API_KEY)")
        base_url = self._base_url or "https://api.cohere.com/v2"
        endpoint_path = self._endpoint_path or "/rerank"
        return RerankerClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            endpoint_path=endpoint_path,
            timeout=self._timeout,
        )
