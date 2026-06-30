"""Version metadata consistency tests."""

from __future__ import annotations

import re
from pathlib import Path

from ai_lib_python import __version__


def test_version_matches_pyproject() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None
    assert __version__ == match.group(1)
