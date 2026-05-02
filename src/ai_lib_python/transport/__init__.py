"""
Transport layer - HTTP client for API communication.

Provides httpx-based transport with:
- Async streaming support
- Proxy configuration
- Timeout management
- API key resolution
- Connection pooling
"""

from ai_lib_python.transport.auth import (
    CredentialSourceKind,
    ResolvedCredential,
    build_auth_metadata,
    resolve_api_key,
    resolve_credential,
)
from ai_lib_python.transport.http import HttpTransport
from ai_lib_python.transport.pool import (
    ConnectionPool,
    PoolConfig,
    PooledTransport,
    PoolStats,
    close_global_pool,
    get_connection_pool,
    set_connection_pool,
)

__all__ = [
    "ConnectionPool",
    "CredentialSourceKind",
    "HttpTransport",
    "PoolConfig",
    "PoolStats",
    "PooledTransport",
    "ResolvedCredential",
    "build_auth_metadata",
    "close_global_pool",
    "get_connection_pool",
    "resolve_api_key",
    "resolve_credential",
    "set_connection_pool",
]
