"""Credential resolution and auth attachment utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_lib_python.protocol.manifest import AuthConfig, ProtocolManifest


class CredentialSourceKind(str, Enum):
    """Where a credential was resolved from."""

    EXPLICIT = "explicit"
    MANIFEST_ENV = "manifest_env"
    CONVENTIONAL_ENV = "conventional_env"
    KEYRING = "keyring"
    NONE = "none"


@dataclass(frozen=True)
class ResolvedCredential:
    """Resolved secret plus redacted diagnostic metadata."""

    secret: str | None
    source_kind: CredentialSourceKind
    source_name: str | None = None
    required_envs: list[str] = field(default_factory=list)
    conventional_envs: list[str] = field(default_factory=list)

    @classmethod
    def missing(cls, required_envs: list[str], conventional_envs: list[str]) -> ResolvedCredential:
        """Return a missing credential with actionable diagnostic env names."""
        return cls(
            secret=None,
            source_kind=CredentialSourceKind.NONE,
            source_name=None,
            required_envs=required_envs,
            conventional_envs=conventional_envs,
        )

    def __repr__(self) -> str:
        redacted = "<redacted>" if self.secret else None
        return (
            "ResolvedCredential("
            f"secret={redacted!r}, "
            f"source_kind={self.source_kind!r}, "
            f"source_name={self.source_name!r}, "
            f"required_envs={self.required_envs!r}, "
            f"conventional_envs={self.conventional_envs!r})"
        )


def provider_id(manifest: ProtocolManifest | None, fallback: str) -> str:
    """Return protocol provider id, falling back to caller-provided provider id."""
    if manifest is None:
        return fallback
    raw = getattr(manifest, "provider_id", None) or manifest.id
    return str(raw)


def primary_auth(manifest: ProtocolManifest | None) -> AuthConfig | None:
    """Return active auth config: V2 endpoint.auth wins, V1 top-level auth falls back."""
    if manifest is None:
        return None
    endpoint_auth = getattr(manifest.endpoint, "auth", None)
    return endpoint_auth or manifest.auth


def shadowed_auth(manifest: ProtocolManifest | None) -> AuthConfig | None:
    """Return divergent top-level auth shadowed by endpoint.auth, if present."""
    if manifest is None:
        return None
    endpoint_auth = getattr(manifest.endpoint, "auth", None)
    top_auth = manifest.auth
    if endpoint_auth is None or top_auth is None:
        return None
    same = (
        endpoint_auth.type == top_auth.type
        and endpoint_auth.token_env == top_auth.token_env
        and endpoint_auth.key_env == top_auth.key_env
    )
    return None if same else top_auth


def required_envs(manifest: ProtocolManifest | None) -> list[str]:
    """Return env vars from the active auth block only."""
    auth = primary_auth(manifest)
    if auth is None:
        return []
    env = auth.token_env or auth.key_env
    if isinstance(env, str) and env.strip():
        return [env.strip()]
    return []


def conventional_envs(provider: str) -> list[str]:
    """Return canonical `${PROVIDER_ID_UPPER_WITH_UNDERSCORES}_API_KEY` fallback."""
    normalized = provider.upper().replace("-", "_")
    return [f"{normalized}_API_KEY"]


def _env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def resolve_credential(
    provider: str,
    manifest: ProtocolManifest | None = None,
    explicit_credential: str | None = None,
    *,
    allow_keyring: bool = True,
) -> ResolvedCredential:
    """Resolve a provider credential using the PT-074 unified chain."""
    pid = provider_id(manifest, provider)
    req_envs = required_envs(manifest)
    conv_envs = conventional_envs(pid)

    if explicit_credential and explicit_credential.strip():
        return ResolvedCredential(
            secret=explicit_credential.strip(),
            source_kind=CredentialSourceKind.EXPLICIT,
            source_name="explicit",
            required_envs=req_envs,
            conventional_envs=conv_envs,
        )

    for name in req_envs:
        if value := _env_value(name):
            return ResolvedCredential(
                secret=value,
                source_kind=CredentialSourceKind.MANIFEST_ENV,
                source_name=name,
                required_envs=req_envs,
                conventional_envs=conv_envs,
            )

    for name in conv_envs:
        if value := _env_value(name):
            return ResolvedCredential(
                secret=value,
                source_kind=CredentialSourceKind.CONVENTIONAL_ENV,
                source_name=name,
                required_envs=req_envs,
                conventional_envs=conv_envs,
            )

    if allow_keyring and (value := _try_keyring(pid)):
        return ResolvedCredential(
            secret=value,
            source_kind=CredentialSourceKind.KEYRING,
            source_name=f"ai-protocol/{pid}",
            required_envs=req_envs,
            conventional_envs=conv_envs,
        )

    return ResolvedCredential.missing(req_envs, conv_envs)


def resolve_api_key(
    provider_id: str,
    manifest: ProtocolManifest | None = None,
    explicit_key: str | None = None,
) -> str | None:
    """Resolve API key for compatibility callers."""
    return resolve_credential(provider_id, manifest, explicit_key).secret


def _try_keyring(provider_id: str) -> str | None:
    """Try to get API key from system keyring.

    Args:
        provider_id: Provider identifier

    Returns:
        API key from keyring or None
    """
    try:
        import keyring  # type: ignore[import-not-found, unused-ignore]

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
    resolved = resolve_credential(provider_id, manifest, api_key)
    headers, _ = build_auth_metadata(manifest, resolved)
    return headers


def build_auth_metadata(
    manifest: ProtocolManifest | None,
    credential: ResolvedCredential,
    *,
    redacted: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build auth headers/query params from active auth metadata."""
    secret = "<redacted>" if redacted and credential.secret else credential.secret
    if not secret:
        return {}, {}

    auth = primary_auth(manifest)
    if auth is None:
        return {"Authorization": f"Bearer {secret}"}, {}

    auth_type = auth.type.lower()
    if auth_type == "query_param":
        return {}, {auth.param_name or "api_key": secret}
    if auth_type in {"api_key", "custom_header", "header"}:
        return {auth.header_name or "X-API-Key": secret}, {}
    # Bearer and unknown types use reversible Bearer fallback.
    header = auth.header_name or "Authorization"
    prefix = auth.prefix if auth.prefix is not None else "Bearer"
    prefix = prefix.strip()
    value = secret if not prefix else f"{prefix} {secret}"
    return {header: value}, {}


def diagnostic_text(
    credential: ResolvedCredential, manifest: ProtocolManifest | None = None
) -> str:
    """Return secret-safe diagnostic text for missing or shadowed credentials."""
    parts: list[str] = []
    if credential.required_envs:
        parts.append(f"required_envs={credential.required_envs!r}")
    if credential.conventional_envs:
        parts.append(f"conventional_envs={credential.conventional_envs!r}")
    if shadowed := shadowed_auth(manifest):
        env = shadowed.token_env or shadowed.key_env
        if env:
            parts.append(f"shadowed_env={env}")
    return " ".join(parts)
