"""COMPLIANCE_SUBSET filtering (PT-073a / E_ONLY_CASES.md)."""

from tests.compliance.conftest import case_matches_subset


def test_e_only_includes_protocol_loading():
    case = {"_source_file": "/x/cases/01-protocol-loading/foo.yaml"}
    assert case_matches_subset(case, "e_only") is True


def test_e_only_excludes_resilience():
    case = {"_source_file": "/x/cases/06-resilience/retry-policy.yaml"}
    assert case_matches_subset(case, "e_only") is False


def test_full_includes_resilience():
    case = {"_source_file": "/x/cases/06-resilience/retry-policy.yaml"}
    assert case_matches_subset(case, None) is True
    assert case_matches_subset(case, "full") is True
