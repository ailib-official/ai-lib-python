"""
Embedding client for generating embeddings.

Provides a unified interface for embedding generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_lib_python.embeddings.types import (
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from ai_lib_python.protocol import ProtocolLoader
from ai_lib_python.transport import HttpTransport

if TYPE_CHECKING:
    from ai_lib_python.protocol.manifest import ProtocolManifest


class EmbeddingClient:
    """Client for generating embeddings.

    Supports batch embedding generation with automatic chunking
    and rate limit handling.

    Example:
        >>> client = await EmbeddingClient.create("openai/text-embedding-3-small")
        >>> response = await client.embed("Hello, world!")
        >>> print(response.first.vector[:5])

        >>> # Batch embedding
        >>> texts = ["Hello", "World", "Test"]
        >>> response = await client.embed_batch(texts)
        >>> for emb in response.embeddings:
        ...     print(f"Text {emb.index}: {len(emb.vector)} dimensions")
    """

    def __init__(
        self,
        manifest: ProtocolManifest,
        transport: HttpTransport,
        model_id: str,
    ) -> None:
        """Initialize embedding client.

        Args:
            manifest: Protocol manifest
            transport: HTTP transport
            model_id: Model identifier
        """
        self._manifest = manifest
        self._transport = transport
        self._model_id = model_id

    @classmethod
    async def create(
        cls,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
    ) -> EmbeddingClient:
        """Create an embedding client.

        Args:
            model: Model identifier (e.g., "openai/text-embedding-3-small")
            api_key: API key (optional, uses environment)
            base_url: Base URL override
            dimensions: Output dimensions override

        Returns:
            EmbeddingClient instance
        """
        return await (
            EmbeddingClientBuilder()
            .model(model)
            .api_key(api_key)
            .base_url(base_url)
            .dimensions(dimensions)
            .build()
        )

    @classmethod
    def builder(cls) -> EmbeddingClientBuilder:
        """Get a builder for creating embedding clients.

        Returns:
            EmbeddingClientBuilder instance
        """
        return EmbeddingClientBuilder()

    async def embed(
        self,
        text: str,
        dimensions: int | None = None,
    ) -> EmbeddingResponse:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            dimensions: Output dimensions (if supported)

        Returns:
            EmbeddingResponse with single embedding
        """
        request = EmbeddingRequest(
            input=text,
            model=self._model_id,
            dimensions=dimensions,
        )
        return await self._execute(request)

    async def embed_batch(
        self,
        texts: list[str],
        dimensions: int | None = None,
        batch_size: int = 100,
    ) -> EmbeddingResponse:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            dimensions: Output dimensions (if supported)
            batch_size: Maximum texts per API call

        Returns:
            EmbeddingResponse with all embeddings
        """
        if len(texts) <= batch_size:
            request = EmbeddingRequest(
                input=texts,
                model=self._model_id,
                dimensions=dimensions,
            )
            return await self._execute(request)

        # Process in batches
        all_embeddings: list[Embedding] = []
        total_usage = EmbeddingUsage()

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            request = EmbeddingRequest(
                input=batch,
                model=self._model_id,
                dimensions=dimensions,
            )
            response = await self._execute(request)

            # Adjust indices for combined result
            for emb in response.embeddings:
                all_embeddings.append(
                    Embedding(
                        index=i + emb.index,
                        vector=emb.vector,
                        object_type=emb.object_type,
                    )
                )

            total_usage.prompt_tokens += response.usage.prompt_tokens
            total_usage.total_tokens += response.usage.total_tokens

        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=self._model_id,
            usage=total_usage,
        )

    async def _execute(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Execute an embedding request.

        Args:
            request: Embedding request

        Returns:
            EmbeddingResponse
        """
        endpoint = self._get_embedding_endpoint()
        payload = request.to_dict()

        response = await self._transport.post(endpoint, json=payload)
        data = response.json()

        return EmbeddingResponse.from_openai_format(data)

    def _get_embedding_endpoint(self) -> str:
        """Get the embedding API endpoint.

        Returns:
            Endpoint path
        """
        # Prefer manifest endpoint mapping when present.
        embedding_cfg = self._manifest.endpoints.get("embeddings")
        if isinstance(embedding_cfg, dict):
            path = embedding_cfg.get("path")
            if isinstance(path, str):
                return path
        return "/v1/embeddings"

    @property
    def model(self) -> str:
        """Get the model identifier."""
        return self._model_id

    @property
    def provider(self) -> str:
        """Get the provider identifier."""
        return self._manifest.id

    async def close(self) -> None:
        """Close the client."""
        await self._transport.close()

    async def __aenter__(self) -> EmbeddingClient:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self, exc_type: Any, exc_val: Any, exc_tb: Any
    ) -> None:
        """Async context manager exit."""
        await self.close()


class EmbeddingClientBuilder:
    """Builder for creating EmbeddingClient instances.

    Example:
        >>> client = await (
        ...     EmbeddingClient.builder()
        ...     .model("openai/text-embedding-3-small")
        ...     .dimensions(512)
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        """Initialize builder."""
        self._model: str | None = None
        self._api_key: str | None = None
        self._base_url: str | None = None
        self._dimensions: int | None = None
        self._timeout: float | None = None

    def model(self, model: str) -> EmbeddingClientBuilder:
        """Set the model.

        Args:
            model: Model identifier (e.g., "openai/text-embedding-3-small")

        Returns:
            Self for chaining
        """
        self._model = model
        return self

    def api_key(self, api_key: str | None) -> EmbeddingClientBuilder:
        """Set the API key.

        Args:
            api_key: API key

        Returns:
            Self for chaining
        """
        self._api_key = api_key
        return self

    def base_url(self, base_url: str | None) -> EmbeddingClientBuilder:
        """Set the base URL.

        Args:
            base_url: Base URL override

        Returns:
            Self for chaining
        """
        self._base_url = base_url
        return self

    def dimensions(self, dimensions: int | None) -> EmbeddingClientBuilder:
        """Set the output dimensions.

        Args:
            dimensions: Output dimensions

        Returns:
            Self for chaining
        """
        self._dimensions = dimensions
        return self

    def timeout(self, timeout: float) -> EmbeddingClientBuilder:
        """Set the request timeout.

        Args:
            timeout: Timeout in seconds

        Returns:
            Self for chaining
        """
        self._timeout = timeout
        return self

    async def build(self) -> EmbeddingClient:
        """Build the embedding client.

        Returns:
            EmbeddingClient instance

        Raises:
            ValueError: If model is not set
        """
        if not self._model:
            raise ValueError("Model must be set before building")

        # Parse model identifier
        parts = self._model.split("/")
        provider_id = parts[0] if len(parts) >= 2 else "openai"
        model_id = parts[-1]

        # Load protocol manifest
        loader = ProtocolLoader()
        manifest = await loader.load_provider(provider_id)

        # Create transport
        transport = HttpTransport(
            manifest=manifest,
            model_id=model_id,
            api_key=self._api_key,
            base_url_override=self._base_url,
            timeout=self._timeout,
        )

        return EmbeddingClient(
            manifest=manifest,
            transport=transport,
            model_id=model_id,
        )
