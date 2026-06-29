"""
Pytest fixtures for AI-Protocol compliance tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# E-only dirs per ai-protocol/tests/compliance/ep-boundary/E_ONLY_CASES.md
_E_ONLY_CASE_PREFIXES = (
    "cases/01-protocol-loading/",
    "cases/02-error-classification/",
    "cases/03-message-building/",
    "cases/04-streaming/",
    "cases/05-request-building/",
    "cases/07-advanced-capabilities/",
    "cases/08-generative-capabilities/",
    "cases/09-credential-resolution/",
)


def compliance_subset() -> str | None:
    """Return COMPLIANCE_SUBSET env value (e.g. e_only) or None for full matrix."""
    raw = os.environ.get("COMPLIANCE_SUBSET", "").strip()
    return raw or None


def case_matches_subset(case: dict, subset: str | None) -> bool:
    """Filter cases per PT-067 E-only subset; None means full compliance matrix."""
    if subset is None or subset == "full":
        return True
    if subset != "e_only":
        return True
    source = str(case.get("_source_file", "")).replace("\\", "/")
    return any(prefix in source for prefix in _E_ONLY_CASE_PREFIXES)


# Default compliance directory: ai-protocol/tests/compliance (sibling repo)
_DEFAULT_COMPLIANCE = (
    Path(__file__).resolve().parents[2] / ".." / ".." / "ai-protocol" / "tests" / "compliance"
)
COMPLIANCE_DIR = Path(
    os.environ.get(
        "COMPLIANCE_DIR",
        str(_DEFAULT_COMPLIANCE.resolve()),
    )
)


@pytest.fixture(scope="session")
def compliance_dir() -> Path:
    """Session fixture for the compliance test cases directory."""
    return COMPLIANCE_DIR


@pytest.fixture(scope="session")
def mock_http_url() -> str | None:
    """Optional mock HTTP URL for compliance tests that need to hit ai-protocol-mock."""
    return os.environ.get("MOCK_HTTP_URL")
