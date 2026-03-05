"""
Protocol loader for loading manifests from various sources.

Supports:
- Local filesystem paths
- Environment variable configuration
- GitHub raw URLs (fallback)
- Caching for performance
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
import yaml

from ai_lib_python.errors import ProtocolError
from ai_lib_python.protocol.manifest import ProtocolManifest

# Default paths to search for protocol files
_DEFAULT_SEARCH_PATHS = [
    "ai-protocol",
    "../ai-protocol",
    "../../ai-protocol",
]

# GitHub raw URL base for fallback loading
_GITHUB_RAW_BASE = "https://raw.githubusercontent.com/hiddenpath/ai-protocol/main"


class ProtocolLoader:
    """Loads protocol manifests from various sources.

    The loader searches for protocol files in the following order:
    1. Explicit base_path if provided
    2. AI_PROTOCOL_DIR or AI_PROTOCOL_PATH environment variable
    3. Common relative paths (ai-protocol/, ../ai-protocol/, etc.)
    4. GitHub raw URLs (if fallback_to_github is enabled)

    Example:
        >>> loader = ProtocolLoader()
        >>> manifest = await loader.load_provider("openai")
        >>> print(manifest.endpoint.base_url)

        >>> # Load specific model
        >>> manifest = await loader.load_model("anthropic/claude-3-5-sonnet")
    """

    def __init__(
        self,
        base_path: str | Path | None = None,
        fallback_to_github: bool = True,
        hot_reload: bool = False,
        cache_enabled: bool = True,
    ) -> None:
        """Initialize the protocol loader.

        Args:
            base_path: Explicit path to protocol directory
            fallback_to_github: Whether to fallback to GitHub if local not found
            hot_reload: Enable hot reload (cache invalidation on file change)
            cache_enabled: Enable caching of loaded manifests
        """
        self._base_path = Path(base_path) if base_path else None
        self._fallback_to_github = fallback_to_github
        self._hot_reload = hot_reload
        self._cache_enabled = cache_enabled
        self._cache: dict[str, ProtocolManifest] = {}
        self._resolved_base: Path | None = None

    def _resolve_base_path(self) -> Path | None:
        """Resolve the protocol base path.

        Returns:
            Resolved Path or None if not found locally
        """
        if self._resolved_base is not None:
            return self._resolved_base

        # 1. Explicit base path
        if self._base_path and self._base_path.exists():
            self._resolved_base = self._base_path
            return self._resolved_base

        # 2. Environment variable
        env_path = os.getenv("AI_PROTOCOL_DIR") or os.getenv("AI_PROTOCOL_PATH")
        if env_path:
            path = Path(env_path)
            if path.exists():
                self._resolved_base = path
                return self._resolved_base

        # 3. Search common paths
        cwd = Path.cwd()
        for rel_path in _DEFAULT_SEARCH_PATHS:
            candidate = cwd / rel_path
            if candidate.exists():
                self._resolved_base = candidate
                return self._resolved_base

        return None

    def _get_provider_path(self, provider_id: str) -> Path | None:
        """Get the path to a provider manifest file."""
        base = self._resolve_base_path()
        if not base:
            return None

        # Try dist/v1/providers first (pre-built JSON)
        json_path = base / "dist" / "v1" / "providers" / f"{provider_id}.json"
        if json_path.exists():
            return json_path

        # Try v1/providers (YAML source)
        yaml_path = base / "v1" / "providers" / f"{provider_id}.yaml"
        if yaml_path.exists():
            return yaml_path

        return None

    def _get_model_path(self, model_id: str) -> Path | None:
        """Get the path to a model manifest file."""
        base = self._resolve_base_path()
        if not base:
            return None

        # Model ID might be "provider/model" or just filename
        parts = model_id.split("/")
        # Extract model file name: "anthropic/claude-3-5-sonnet" -> "claude"
        model_file = parts[1].split("-")[0] if len(parts) == 2 else model_id

        # Try dist/v1/models first
        json_path = base / "dist" / "v1" / "models" / f"{model_file}.json"
        if json_path.exists():
            return json_path

        yaml_path = base / "v1" / "models" / f"{model_file}.yaml"
        if yaml_path.exists():
            return yaml_path

        return None

    async def _load_from_github(self, path: str) -> dict[str, Any]:
        """Load a file from GitHub raw URL.

        Args:
            path: Relative path within the repository

        Returns:
            Parsed JSON/YAML content
        """
        url = urljoin(_GITHUB_RAW_BASE + "/", path)

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            if response.status_code != 200:
                raise ProtocolError(
                    f"Failed to load from GitHub: {url} (status {response.status_code})",
                    protocol_path=url,
                )

            content = response.text
            parsed: Any = (
                json.loads(content) if path.endswith(".json") else yaml.safe_load(content)
            )
            if isinstance(parsed, dict):
                return parsed
            raise ProtocolError(
                f"Protocol file did not parse to object: {url}",
                protocol_path=url,
            )

    def _load_file(self, path: Path) -> dict[str, Any]:
        """Load and parse a local file.

        Args:
            path: Path to the file

        Returns:
            Parsed content
        """
        content = path.read_text(encoding="utf-8")
        parsed: Any = (
            json.loads(content) if path.suffix == ".json" else yaml.safe_load(content)
        )
        if isinstance(parsed, dict):
            return parsed
        raise ProtocolError(
            f"Protocol file did not parse to object: {path}",
            protocol_path=str(path),
        )

    async def load_provider(self, provider_id: str) -> ProtocolManifest:
        """Load a provider manifest.

        Args:
            provider_id: Provider identifier (e.g., "openai", "anthropic")

        Returns:
            ProtocolManifest for the provider

        Raises:
            ProtocolError: If manifest not found or invalid
        """
        cache_key = f"provider:{provider_id}"

        # Check cache
        if self._cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        # Try local path first
        local_path = self._get_provider_path(provider_id)
        if local_path:
            try:
                data = self._load_file(local_path)
                manifest = ProtocolManifest.model_validate(data)
                if self._cache_enabled:
                    self._cache[cache_key] = manifest
                return manifest
            except Exception as e:
                raise ProtocolError(
                    f"Failed to parse provider manifest: {e}",
                    protocol_path=str(local_path),
                ) from e

        # Fallback to GitHub
        if self._fallback_to_github:
            try:
                data = await self._load_from_github(
                    f"dist/v1/providers/{provider_id}.json"
                )
                manifest = ProtocolManifest.model_validate(data)
                if self._cache_enabled:
                    self._cache[cache_key] = manifest
                return manifest
            except Exception as e:
                raise ProtocolError(
                    f"Provider '{provider_id}' not found locally or on GitHub: {e}",
                    protocol_path=provider_id,
                ) from e

        raise ProtocolError(
            f"Provider '{provider_id}' not found",
            protocol_path=provider_id,
        )

    async def load_model(self, model_id: str) -> ProtocolManifest:
        """Load a model manifest (or its provider manifest).

        For most cases, this loads the provider manifest that contains
        the model configuration.

        Args:
            model_id: Model identifier (e.g., "anthropic/claude-3-5-sonnet")

        Returns:
            ProtocolManifest for the model's provider

        Raises:
            ProtocolError: If manifest not found or invalid
        """
        # Extract provider from model_id
        parts = model_id.split("/")
        provider_id = parts[0] if len(parts) >= 2 else model_id

        return await self.load_provider(provider_id)

    async def load_spec(self, version: str = "v1") -> dict[str, Any]:
        """Load the protocol specification file.

        Args:
            version: Protocol version (default: "v1")

        Returns:
            Parsed spec content
        """
        base = self._resolve_base_path()

        # Try local
        if base:
            json_path = base / "dist" / version / "spec.json"
            if json_path.exists():
                return self._load_file(json_path)

            yaml_path = base / version / "spec.yaml"
            if yaml_path.exists():
                return self._load_file(yaml_path)

        # Fallback to GitHub
        if self._fallback_to_github:
            return await self._load_from_github(f"dist/{version}/spec.json")

        raise ProtocolError(f"Spec '{version}' not found")

    def clear_cache(self) -> None:
        """Clear the manifest cache."""
        self._cache.clear()

    def invalidate(self, key: str) -> None:
        """Invalidate a specific cache entry.

        Args:
            key: Cache key to invalidate (e.g., "provider:openai")
        """
        self._cache.pop(key, None)

    def register_provider(self, data: dict[str, Any]) -> ProtocolManifest:
        """Register a custom provider manifest at runtime.

        Args:
            data: Provider manifest data

        Returns:
            Validated ProtocolManifest
        """
        manifest = ProtocolManifest.model_validate(data)
        cache_key = f"provider:{manifest.id}"
        self._cache[cache_key] = manifest
        return manifest

    @property
    def base_path(self) -> Path | None:
        """Get the resolved base path."""
        return self._resolve_base_path()
