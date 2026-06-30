"""Package version — single source via installed distribution metadata."""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("ai-lib-python")
    except PackageNotFoundError:
        __version__ = "0.8.5"
except ImportError:  # pragma: no cover - Python < 3.8 guard
    __version__ = "0.8.5"
