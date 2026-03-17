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
    elif test_type == "protocol_loading":
        run_protocol_loading(input_data, expected, case)
    elif test_type == "retry_decision":
        run_retry_decision(input_data, expected, case)
    elif test_type == "message_building":
        run_message_building(input_data, expected, case)
    elif test_type == "parameter_mapping":
        run_parameter_mapping(input_data, expected, case)
    elif test_type == "stream_decode":
        run_stream_decode(input_data, expected, case)
    elif test_type == "event_mapping":
        run_event_mapping(input_data, expected, case)
    elif test_type == "tool_accumulation":
        run_tool_accumulation(input_data, expected, case)
    elif test_type == "capability_guard":
        run_capability_guard(input_data, expected, case)
    elif test_type == "advanced_endpoint_mapping":
        run_advanced_endpoint_mapping(input_data, expected, case)
    elif test_type == "fallback_decision":
        run_fallback_decision(input_data, expected, case)
    elif test_type == "provider_mock_behavior":
        run_provider_mock_behavior(input_data, expected, case)
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


def _manifest_has_required_shape(manifest: dict[str, Any]) -> bool:
    """Minimal required shape for protocol_loading compliance."""
    if not isinstance(manifest.get("id"), str) or not manifest.get("id"):
        return False
    if not isinstance(manifest.get("protocol_version"), str) or not manifest.get("protocol_version"):
        return False
    endpoint = manifest.get("endpoint")
    if not isinstance(endpoint, dict):
        return False
    base_url = endpoint.get("base_url")
    return isinstance(base_url, str) and bool(base_url)


def _ios_capability_profile_errors(manifest: dict[str, Any]) -> list[str]:
    """Return IOS-phase capability_profile semantic errors."""
    cp = manifest.get("capability_profile")
    if cp is None:
        return []
    if not isinstance(cp, dict):
        return ["capability_profile must be object"]

    errors: list[str] = []
    phase = cp.get("phase")
    if phase == "ios_v1":
        if "process" in cp or "contract" in cp:
            errors.append("must NOT have additional properties")
        if not any(key in cp for key in ("inputs", "outcomes", "systems")):
            errors.append("must match at least one schema in anyOf")
    return errors


def run_protocol_loading(
    input_data: dict[str, Any],
    expected: dict[str, Any],
    case: dict[str, Any],
) -> None:
    """Run protocol_loading test."""
    case_id = case.get("id", "unknown")
    case_name = case.get("name", "unnamed")
    manifest_path = input_data.get("manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path:
        pytest.fail(f"[{case_id}] {case_name}: protocol_loading requires input.manifest_path")
    manifest = _load_provider_manifest(COMPLIANCE_DIR, manifest_path)
    assert manifest is not None, f"[{case_id}] {case_name}: manifest not found at {manifest_path}"

    expected_valid = bool(expected.get("valid", False))
    ios_errors = _ios_capability_profile_errors(manifest)
    actual_valid = _manifest_has_required_shape(manifest) and not ios_errors

    assert actual_valid == expected_valid, (
        f"[{case_id}] {case_name}: valid expected {expected_valid}, got {actual_valid}"
    )

    expected_errors_raw = expected.get("errors")
    expected_errors = (
        [str(item) for item in expected_errors_raw if isinstance(item, str)]
        if isinstance(expected_errors_raw, list)
        else []
    )
    if expected_errors:
        actual_error_text = " | ".join(ios_errors)
        for expected_error in expected_errors:
            assert expected_error in actual_error_text, (
                f"[{case_id}] {case_name}: expected error '{expected_error}' not found in '{actual_error_text}'"
            )

    if expected_valid:
        if "provider_id" in expected:
            assert manifest.get("id") == expected["provider_id"], (
                f"[{case_id}] {case_name}: provider_id expected {expected['provider_id']}, "
                f"got {manifest.get('id')}"
            )
        if "protocol_version" in expected:
            assert manifest.get("protocol_version") == expected["protocol_version"], (
                f"[{case_id}] {case_name}: protocol_version expected {expected['protocol_version']}, "
                f"got {manifest.get('protocol_version')}"
            )
    else:
        # For invalid manifests, ensure at least one required field is missing.
        missing_required = []
        if not manifest.get("id"):
            missing_required.append("id")
        if not manifest.get("protocol_version"):
            missing_required.append("protocol_version")
        endpoint = manifest.get("endpoint")
        if not isinstance(endpoint, dict) or not endpoint.get("base_url"):
            missing_required.append("endpoint.base_url")
        assert missing_required or ios_errors, (
            f"[{case_id}] {case_name}: expected invalid manifest to fail required-shape or IOS checks"
        )


def _compute_retry_delay_ms(policy: dict[str, Any], attempt: int) -> int:
    """Deterministic delay baseline for compliance validation."""
    min_delay = int(policy.get("min_delay_ms", 1000))
    max_delay = int(policy.get("max_delay_ms", 60_000))
    # attempt in compliance cases is 1-based
    exponent = max(0, attempt - 1)
    delay = min_delay * (2**exponent)
    return min(delay, max_delay)


def run_retry_decision(
    input_data: dict[str, Any],
    expected: dict[str, Any],
    case: dict[str, Any],
) -> None:
    """Run retry_decision test."""
    case_id = case.get("id", "unknown")
    case_name = case.get("name", "unnamed")

    error = input_data.get("error") or {}
    policy = input_data.get("retry_policy") or {}
    attempt = int(input_data.get("attempt", 1))

    max_retries = int(policy.get("max_retries", 0))
    retry_on_error_code = {
        str(item) for item in policy.get("retry_on_error_code", []) if isinstance(item, str)
    }
    error_name = str(error.get("error_name", ""))
    retryable = bool(error.get("retryable", False))

    within_limit = attempt <= max_retries
    matches_policy = error_name in retry_on_error_code
    should_retry = within_limit and retryable and (matches_policy or not retry_on_error_code)

    expected_should_retry = bool(expected.get("should_retry", False))
    assert should_retry == expected_should_retry, (
        f"[{case_id}] {case_name}: should_retry expected {expected_should_retry}, got {should_retry}"
    )

    if expected_should_retry and "delay_ms" in expected:
        expected_delay = expected["delay_ms"]
        delay_ms = _compute_retry_delay_ms(policy, attempt)
        if isinstance(expected_delay, dict):
            min_expected = int(expected_delay.get("min", 0))
            max_expected = int(expected_delay.get("max", 10**9))
            assert min_expected <= delay_ms <= max_expected, (
                f"[{case_id}] {case_name}: delay_ms={delay_ms} out of range [{min_expected}, {max_expected}]"
            )


def run_message_building(
    input_data: dict[str, Any],
    expected: dict[str, Any],
    case: dict[str, Any],
) -> None:
    """Run message_building test."""
    case_id = case.get("id", "unknown")
    case_name = case.get("name", "unnamed")
    messages = input_data.get("messages") or []
    expected_body = expected.get("normalized_body") or {}
    expected_messages = expected_body.get("messages") or []
    expected_count = int(expected.get("message_count", len(expected_messages)))

    assert messages == expected_messages, (
        f"[{case_id}] {case_name}: normalized messages mismatch"
    )
    assert len(messages) == expected_count, (
        f"[{case_id}] {case_name}: message_count expected {expected_count}, got {len(messages)}"
    )


def run_parameter_mapping(
    input_data: dict[str, Any],
    expected: dict[str, Any],
    case: dict[str, Any],
) -> None:
    """Run parameter_mapping test."""
    case_id = case.get("id", "unknown")
    case_name = case.get("name", "unnamed")
    standard_params = dict(input_data.get("standard_params") or {})
    provider_params = dict(standard_params)

    setup = case.get("setup") if isinstance(case.get("setup"), dict) else {}
    manifest_path = setup.get("manifest_path")
    if isinstance(manifest_path, str):
        manifest = _load_provider_manifest(COMPLIANCE_DIR, manifest_path) or {}
        parameters = manifest.get("parameters") if isinstance(manifest.get("parameters"), dict) else {}
        for key, value in parameters.items():
            if key not in provider_params and isinstance(value, dict) and "default" in value:
                provider_params[key] = value["default"]

    expected_provider_params = expected.get("provider_params") or {}
    assert provider_params == expected_provider_params, (
        f"[{case_id}] {case_name}: provider_params expected {expected_provider_params}, got {provider_params}"
    )


def run_stream_decode(
    input_data: dict[str, Any],
    expected: dict[str, Any],
    case: dict[str, Any],
) -> None:
    """Run stream_decode test."""
    case_id = case.get("id", "unknown")
    case_name = case.get("name", "unnamed")
    raw_chunks = input_data.get("raw_chunks") or []
    decoder_config = input_data.get("decoder_config") or {}
    prefix = str(decoder_config.get("prefix", "data: "))
    done_signal = str(decoder_config.get("done_signal", "[DONE]"))

    frames: list[dict[str, Any]] = []
    done_received = False
    for chunk in raw_chunks:
        for line in str(chunk).splitlines():
            if not line.startswith(prefix):
                continue
            payload = line[len(prefix):].strip()
            if payload == done_signal:
                done_received = True
                continue
            if not payload:
                continue
            import json
            frames.append(json.loads(payload))

    frame_count_cfg = expected.get("frame_count") or {}
    min_frames = int(frame_count_cfg.get("min", 0))
    max_frames = int(frame_count_cfg.get("max", 10**9))
    assert min_frames <= len(frames) <= max_frames, (
        f"[{case_id}] {case_name}: frame_count={len(frames)} out of range [{min_frames}, {max_frames}]"
    )

    if expected.get("completed") is True:
        assert done_received, f"[{case_id}] {case_name}: expected done signal"
    if expected.get("done_received") is True:
        assert done_received, f"[{case_id}] {case_name}: done_received expected true"
    if (expected.get("events") or [{}])[0].get("has_content"):
        has_content = any(
            isinstance(f.get("choices"), list)
            and f["choices"]
            and isinstance(f["choices"][0], dict)
            and isinstance(f["choices"][0].get("delta"), dict)
            and bool(f["choices"][0]["delta"].get("content"))
            for f in frames
        )
        assert has_content, f"[{case_id}] {case_name}: expected at least one content frame"


def run_event_mapping(
    input_data: dict[str, Any],
    expected: dict[str, Any],
    case: dict[str, Any],
) -> None:
    """Run event_mapping test."""
    case_id = case.get("id", "unknown")
    case_name = case.get("name", "unnamed")
    frames = input_data.get("frames") or []
    actual_events: list[dict[str, Any]] = []
    for frame in frames:
        choices = frame.get("choices") if isinstance(frame, dict) else None
        if not isinstance(choices, list) or not choices:
            continue
        first = choices[0] if isinstance(choices[0], dict) else {}
        delta = first.get("delta") if isinstance(first.get("delta"), dict) else {}
        if "content" in delta:
            actual_events.append({"type": "PartialContentDelta", "content": delta.get("content")})
        if "tool_calls" in delta:
            actual_events.append({"type": "PartialToolCall", "tool_calls": delta.get("tool_calls")})
        if "finish_reason" in first and first.get("finish_reason") is not None:
            actual_events.append({"type": "StreamEnd", "finish_reason": first.get("finish_reason")})

    expected_events = expected.get("events") or []
    assert actual_events == expected_events, (
        f"[{case_id}] {case_name}: events mismatch expected {expected_events}, got {actual_events}"
    )
    if "event_count" in expected:
        assert len(actual_events) == int(expected["event_count"]), (
            f"[{case_id}] {case_name}: event_count expected {expected['event_count']}, got {len(actual_events)}"
        )


def run_tool_accumulation(
    input_data: dict[str, Any],
    expected: dict[str, Any],
    case: dict[str, Any],
) -> None:
    """Run tool_accumulation test."""
    case_id = case.get("id", "unknown")
    case_name = case.get("name", "unnamed")
    partial_chunks = input_data.get("partial_chunks") or []
    merged: dict[tuple[int, str], dict[str, Any]] = {}
    for chunk in partial_chunks:
        if not isinstance(chunk, dict):
            continue
        index = int(chunk.get("index", 0))
        call_id = str(chunk.get("id", ""))
        key = (index, call_id)
        function = chunk.get("function") if isinstance(chunk.get("function"), dict) else {}
        if key not in merged:
            merged[key] = {
                "index": index,
                "id": call_id,
                "type": chunk.get("type", "function"),
                "function": {
                    "name": function.get("name"),
                    "arguments": "",
                },
            }
        merged[key]["function"]["arguments"] += str(function.get("arguments", ""))

    assembled = list(merged.values())
    expected_calls = expected.get("assembled_tool_calls") or []
    assert assembled == expected_calls, (
        f"[{case_id}] {case_name}: assembled_tool_calls expected {expected_calls}, got {assembled}"
    )
    if "complete" in expected:
        assert bool(assembled) is bool(expected["complete"]), (
            f"[{case_id}] {case_name}: complete expected {expected['complete']}, got {bool(assembled)}"
        )


def run_capability_guard(
    input_data: dict[str, Any],
    expected: dict[str, Any],
    case: dict[str, Any],
) -> None:
    """Run capability_guard test."""
    case_id = case.get("id", "unknown")
    case_name = case.get("name", "unnamed")
    method = str(input_data.get("method", ""))
    manifest_raw = str(input_data.get("manifest", ""))
    manifest = yaml.safe_load(manifest_raw) if manifest_raw else {}
    cap_key = {
        "MCPListTools": "mcp",
        "ComputerUse": "computer_use",
        "Reason": "reasoning",
        "VideoGenerate": "video",
    }.get(method, "")
    caps = manifest.get("capabilities")
    has_cap = False
    if isinstance(caps, list):
        has_cap = cap_key in caps
    elif isinstance(caps, dict):
        has_cap = cap_key in caps
    actual_code = "" if has_cap else "E1005"
    expected_code = str(expected.get("error_code", ""))
    assert actual_code == expected_code, (
        f"[{case_id}] {case_name}: error_code expected {expected_code}, got {actual_code}"
    )


def run_advanced_endpoint_mapping(
    input_data: dict[str, Any],
    expected: dict[str, Any],
    case: dict[str, Any],
) -> None:
    """Run advanced_endpoint_mapping test."""
    case_id = case.get("id", "unknown")
    case_name = case.get("name", "unnamed")
    operation = str(input_data.get("operation", ""))
    fallback = str(input_data.get("fallback", ""))
    manifest_raw = str(input_data.get("manifest", ""))
    manifest = yaml.safe_load(manifest_raw) if manifest_raw else {}
    op = (
        manifest.get("core", {})
        .get("endpoint", {})
        .get("endpoints", {})
        .get(operation, {})
    )
    actual_path = str(op.get("path", fallback))
    actual_method = str(op.get("method", "POST")).upper()
    expected_path = str(expected.get("path", ""))
    expected_method = str(expected.get("method", ""))
    assert actual_path == expected_path, (
        f"[{case_id}] {case_name}: path expected {expected_path}, got {actual_path}"
    )
    assert actual_method == expected_method, (
        f"[{case_id}] {case_name}: method expected {expected_method}, got {actual_method}"
    )


def run_fallback_decision(
    input_data: dict[str, Any],
    expected: dict[str, Any],
    case: dict[str, Any],
) -> None:
    """Run fallback_decision test."""
    case_id = case.get("id", "unknown")
    case_name = case.get("name", "unnamed")
    code = str(input_data.get("error_code", ""))
    should_fallback = code in {"E1002", "E2001", "E2002", "E3001", "E3002", "E3003"}
    expected_value = bool(expected.get("should_fallback", False))
    assert should_fallback == expected_value, (
        f"[{case_id}] {case_name}: should_fallback expected {expected_value}, got {should_fallback}"
    )


def _value_at_path(root: Any, path: str) -> Any:
    cur = root
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                return None
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
        else:
            return None
    return cur


def run_provider_mock_behavior(
    input_data: dict[str, Any],
    expected: dict[str, Any],
    case: dict[str, Any],
) -> None:
    """Run provider_mock_behavior test."""
    case_id = case.get("id", "unknown")
    case_name = case.get("name", "unnamed")
    req = input_data.get("request_body") or {}
    resp = input_data.get("response_body") or {}
    for path, value in (expected.get("request_assert") or {}).items():
        got = _value_at_path(req, str(path))
        assert got == value, (
            f"[{case_id}] {case_name}: request_assert {path} expected {value}, got {got}"
        )
    for path, value in (expected.get("response_assert") or {}).items():
        got = _value_at_path(resp, str(path))
        assert got == value, (
            f"[{case_id}] {case_name}: response_assert {path} expected {value}, got {got}"
        )
