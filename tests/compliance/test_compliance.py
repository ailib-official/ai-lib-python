"""
AI-Protocol compliance test runner.

Discovers and runs declarative YAML test cases from the ai-protocol
compliance suite, ensuring cross-runtime behavioral consistency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import yaml

from tests.compliance.conftest import COMPLIANCE_DIR

if TYPE_CHECKING:
    from pathlib import Path


def discover_test_cases(compliance_dir: Path) -> list[dict[str, Any]]:
    """Discover all YAML test cases from the compliance directory."""
    cases: list[dict[str, Any]] = []
    cases_dir = compliance_dir / "cases"
    if not cases_dir.exists():
        return cases

    for yaml_file in sorted(cases_dir.rglob("*.yaml")):
        with yaml_file.open(encoding="utf-8") as f:
            content = f.read()
        # Multi-document YAML
        for doc in yaml.safe_load_all(content):
            if doc and isinstance(doc, dict) and "id" in doc:
                doc = dict(doc)
                doc["_source_file"] = str(yaml_file)
                cases.append(doc)
    return cases


def get_test_cases() -> list[dict[str, Any]]:
    """Get all test cases, parametrized for pytest."""
    if not COMPLIANCE_DIR.exists():
        return []
    return discover_test_cases(COMPLIANCE_DIR)


def _load_provider_manifest(compliance_dir: Path, manifest_path: str) -> dict[str, Any] | None:
    """Load provider manifest from setup.manifest_path (relative to compliance dir)."""
    path = compliance_dir / manifest_path
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_provider_classification(case: dict[str, Any], compliance_dir: Path) -> dict[str, Any] | None:
    """Extract error_classification from provider manifest if setup.manifest_path is set."""
    setup = case.get("setup")
    if not setup or not isinstance(setup, dict):
        return None
    manifest_path = setup.get("manifest_path")
    if not manifest_path:
        return None
    manifest = _load_provider_manifest(compliance_dir, manifest_path)
    if not manifest:
        return None
    return manifest.get("error_classification")


# Parametrize tests by case
@pytest.mark.parametrize(
    "case",
    get_test_cases(),
    ids=lambda c: f"{c.get('id', 'unknown')}: {c.get('name', 'unnamed')}",
)
def test_compliance(case: dict[str, Any]) -> None:
    """Run a single compliance test case."""
    input_data = case["input"]
    expected = case["expected"]
    test_type = input_data["type"]

    if test_type == "error_classification":
        run_error_classification(input_data, expected, case)
    else:
        pytest.skip(f"Test type '{test_type}' not yet implemented")


def run_error_classification(
    input_data: dict[str, Any],
    expected: dict[str, Any],
    case: dict[str, Any],
) -> None:
    """Run error_classification test."""
    from ai_lib_python.errors.classification import classify_http_error

    http_status = input_data["http_status"]
    response_body = input_data.get("response_body")

    # Load provider classification from manifest if specified
    provider_classification = _get_provider_classification(case, COMPLIANCE_DIR)

    # Classify using the standard mechanism
    error_class = classify_http_error(
        http_status,
        response_body,
        provider_classification=provider_classification,
    )
    standard_code = error_class.standard_code

    # Assert expected values
    case_id = case.get("id", "unknown")
    case_name = case.get("name", "unnamed")

    if "error_code" in expected:
        assert standard_code.code == expected["error_code"], (
            f"[{case_id}] {case_name}: error_code expected {expected['error_code']}, "
            f"got {standard_code.code}"
        )

    if "error_name" in expected:
        assert standard_code.name == expected["error_name"], (
            f"[{case_id}] {case_name}: error_name expected {expected['error_name']}, "
            f"got {standard_code.name}"
        )

    if "retryable" in expected:
        assert standard_code.retryable == expected["retryable"], (
            f"[{case_id}] {case_name}: retryable expected {expected['retryable']}, "
            f"got {standard_code.retryable}"
        )

    if "fallbackable" in expected:
        assert standard_code.fallbackable == expected["fallbackable"], (
            f"[{case_id}] {case_name}: fallbackable expected {expected['fallbackable']}, "
            f"got {standard_code.fallbackable}"
        )
