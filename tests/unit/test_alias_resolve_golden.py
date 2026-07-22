"""MULTI-ALIAS-XLANG-001 — consume ai-protocol alias-resolve golden vectors."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _protocol_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("AI_PROTOCOL_ROOT") or os.environ.get("AI_PROTOCOL_DIR")
    if env:
        roots.append(Path(env))
    here = Path(__file__).resolve()
    roots.extend(
        [
            here.parents[2] / "ai-protocol",
            here.parents[3] / "ai-protocol",
            Path("/home/alex/ai-protocol"),
        ]
    )
    return roots


def _find(*rel: str) -> Path | None:
    for root in _protocol_roots():
        p = root.joinpath(*rel)
        if p.is_file():
            return p
    return None


def _canonical_from_identity(value: object, key: str) -> str | None:
    if not isinstance(value, dict):
        return None
    families = value.get("families")
    if isinstance(families, list):
        for family in families:
            if not isinstance(family, dict):
                continue
            canonical = family.get("canonical_id")
            if not isinstance(canonical, str):
                continue
            if key == canonical:
                return canonical
            aliases = family.get("aliases") or []
            if isinstance(aliases, list) and key in aliases:
                return canonical
        return None
    canonical = value.get("canonical_id")
    if not isinstance(canonical, str):
        return None
    if key == canonical:
        return canonical
    aliases = value.get("aliases") or []
    if isinstance(aliases, list) and key in aliases:
        return canonical
    return None


def test_alias_resolve_golden_matches_identity_map() -> None:
    golden_path = _find("v2", "alias-resolve.golden.json")
    map_path = _find("v2", "provider-identity.fixture.json")
    if golden_path is None or map_path is None:
        pytest.skip("ai-protocol MULTI-ALIAS golden not present in checkout")

    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    identity = json.loads(map_path.read_text(encoding="utf-8"))
    vectors = golden.get("vectors") or []
    assert vectors, "golden vectors must be non-empty"
    for row in vectors:
        inp = row["input"]
        expected = row.get("canonical")
        got = _canonical_from_identity(identity, inp)
        assert got == expected, f"input={inp!r}: map={got!r} golden={expected!r}"
