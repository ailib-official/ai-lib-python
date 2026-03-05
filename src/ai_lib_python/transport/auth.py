"""
API key resolution utilities.

Resolves API keys from multiple sources:
1. Explicit value
2. Environment variables
3. System keyring (optional)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_lib_python.protocol.manifest import ProtocolManifest


def resolve_api_key(
    provider_id: str,
    manifest: ProtocolManifest | None = None,
    explicit_key: str | None = None,
) -> str | None:
    """Resolve API key for a provider.

    Resolution order:
    1. Explicit key if provided
    2. Environment variable from manifest (auth.token_env)
    3. Standard environment variable ({PROVIDER_ID}_API_KEY)
    4. System keyring (if available)

    Args:
        provider_id: Provider identifier (e.g., "openai", "anthropic")
        manifest: Optional provider manifest with auth configuration
        explicit_key: Explicitly provided API key

    Returns:
        Resolved API key or None if not found
    """
    # 1. Explicit key
    if explicit_key:
        return explicit_key

    # 2. Environment variable from manifest
    if manifest and manifest.auth and manifest.auth.token_env:
        key = os.getenv(manifest.auth.token_env)
        if key:
            return key

    # 3. Standard environment variable
    env_var = f"{provider_id.upper()}_API_KEY"
    key = os.getenv(env_var)
    if key:
        return key

    # Also try with underscores replaced by hyphens and vice versa
    alt_env_var = f"{provider_id.upper().replace('-', '_')}_API_KEY"
    if alt_env_var != env_var:
        key = os.getenv(alt_env_var)
        if key:
            return key

    # 4. Try keyring (optional)
    key = _try_keyring(provider_id)
    if key:
        return key

    return None


def _try_keyring(provider_id: str) -> str | None:
    """Try to get API key from system keyring.

    Args:
        provider_id: Provider identifier

    Returns:
        API key from keyring or None
    """
    try:
        import keyring

        # Try with service name "ai-protocol"
        key = keyring.get_password("ai-protocol", provider_id)
        if isinstance(key, str) and key:
            return key

        # Try with service name "ai-lib"
        key = keyring.get_password("ai-lib", provider_id)
        if isinstance(key, str) and key:
            return key

    except ImportError:
        # keyring not installed
        pass
    except Exception:
        # Keyring error (common in containers, WSL, etc.)
        pass

    return None


def get_auth_header(
    provider_id: str,
    manifest: ProtocolManifest | None = None,
    api_key: str | None = None,
) -> dict[str, str]:
    """Get authentication header for a provider.

    Args:
        provider_id: Provider identifier
        manifest: Optional provider manifest
        api_key: Optional explicit API key

    Returns:
        Dictionary with authentication header(s)
    """
    key = resolve_api_key(provider_id, manifest, api_key)
    if not key:
        return {}

    # Determine auth type from manifest
    auth_type = "bearer"
    header_name = "Authorization"

    if manifest and manifest.auth:
        auth_type = manifest.auth.type.lower()
        if manifest.auth.header_name:
            header_name = manifest.auth.header_name

    # Build header based on auth type
    if auth_type == "bearer":
        return {header_name: f"Bearer {key}"}
    elif auth_type == "api_key":
        # Some APIs use X-API-Key style
        if header_name == "Authorization":
            header_name = "X-API-Key"
        return {header_name: key}
    else:
        # Default to bearer
        return {header_name: f"Bearer {key}"}
