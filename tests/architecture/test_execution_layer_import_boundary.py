"""PT-069: execution-layer packages must not import contact/policy modules (PT-067 matrix).

Loads `ai-protocol/tests/compliance/ep-boundary/module-matrix.yaml` and scans
`src/ai_lib_python/<execution_layer>/` for explicit imports of `ai_lib_python.<contact>`.
Keep in sync with `check_ep_boundary.py --python-root` in ai-protocol.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _module_matrix_path() -> Path:
    # Match tests/compliance/conftest.py: two levels up from package root, then ai-protocol/
    root = _repo_root()
    return (
        (root / ".." / ".." / "ai-protocol" / "tests" / "compliance" / "ep-boundary" / "module-matrix.yaml")
        .resolve()
    )


def _compile_contact_import_patterns(contact: list[str]) -> list[tuple[re.Pattern[str], str]]:
    """Build regexes for `from ai_lib_python.X` / `import ai_lib_python.X` where X is contact."""
    if not contact:
        return []
    alt = "|".join(re.escape(name) for name in sorted(set(contact)))
    return [
        (re.compile(rf"from\s+ai_lib_python\.({alt})\b"), "from"),
        (re.compile(rf"import\s+ai_lib_python\.({alt})\b"), "import"),
    ]


def _scan_tree(
    tree_root: Path,
    patterns: list[tuple[re.Pattern[str], str]],
) -> list[tuple[Path, int, str]]:
    violations: list[tuple[Path, int, str]] = []
    for path in sorted(tree_root.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.split("#", 1)[0]
            for cre, _kind in patterns:
                if cre.search(stripped):
                    violations.append((path, lineno, line.strip()))
                    break
    return violations


def test_execution_layer_packages_do_not_import_contact_modules() -> None:
    matrix_path = _module_matrix_path()
    if not matrix_path.is_file():
        pytest.skip(f"module-matrix.yaml not found (need ai-protocol sibling): {matrix_path}")

    data: dict[str, Any] = yaml.safe_load(matrix_path.read_text(encoding="utf-8")) or {}
    py_cfg = data.get("python") or {}
    execution_names: list[str] = list(py_cfg.get("execution_layer") or [])
    contact_names: list[str] = list(py_cfg.get("contact") or [])
    assert execution_names, "python.execution_layer empty in module-matrix.yaml"
    assert contact_names, "python.contact empty in module-matrix.yaml"

    pkg_root = _repo_root() / "src" / "ai_lib_python"
    patterns = _compile_contact_import_patterns(contact_names)
    all_violations: list[tuple[Path, int, str]] = []

    for name in execution_names:
        sub = pkg_root / name
        if not sub.is_dir():
            continue
        all_violations.extend(_scan_tree(sub, patterns))

    if all_violations:
        msg_lines = ["Execution-layer packages imported contact-layer modules:"]
        for path, lineno, line in all_violations:
            rel = path.relative_to(_repo_root())
            msg_lines.append(f"  {rel}:{lineno}: {line}")
        pytest.fail("\n".join(msg_lines))
