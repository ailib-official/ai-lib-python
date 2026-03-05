"""Computer Use 抽象层单元测试。"""

from __future__ import annotations

import pytest

from ai_lib_python.computer_use import (
    ActionType,
    ComputerAction,
    ImplementationStyle,
    SafetyPolicy,
    SafetyViolation,
    SandboxMode,
    extract_provider_config,
)


class TestSafetyPolicy:
    def test_default(self) -> None:
        policy = SafetyPolicy()
        assert policy.confirmation_required
        assert policy.sandbox_mode == SandboxMode.RECOMMENDED
        assert policy.action_logging

    def test_max_actions(self) -> None:
        policy = SafetyPolicy(max_actions_per_turn=5)
        action = ComputerAction.screenshot()
        policy.validate_action(action, actions_this_turn=4)  # ok
        with pytest.raises(SafetyViolation, match="Max actions"):
            policy.validate_action(action, actions_this_turn=5)

    def test_domain_allowlist(self) -> None:
        policy = SafetyPolicy(domain_allowlist={"example.com", "safe.org"})
        ok_action = ComputerAction.browser_navigate("https://example.com/page")
        policy.validate_action(ok_action)

        bad_action = ComputerAction.browser_navigate("https://evil.com/phish")
        with pytest.raises(SafetyViolation, match="not in the allowlist"):
            policy.validate_action(bad_action)

    def test_sensitive_path(self) -> None:
        policy = SafetyPolicy()
        bad = ComputerAction.file_read("/home/user/.ssh/id_rsa")
        with pytest.raises(SafetyViolation, match="sensitive"):
            policy.validate_action(bad)

        safe = ComputerAction.file_read("/tmp/output.txt")
        policy.validate_action(safe)  # no error

    def test_from_config(self) -> None:
        config = {
            "confirmation_required": False,
            "sandbox_mode": "required",
            "max_actions_per_turn": 10,
        }
        policy = SafetyPolicy.from_config(config)
        assert not policy.confirmation_required
        assert policy.sandbox_mode == SandboxMode.REQUIRED
        assert policy.max_actions_per_turn == 10


class TestComputerAction:
    def test_screenshot_factory(self) -> None:
        action = ComputerAction.screenshot("jpeg")
        assert action.action_type == ActionType.SCREENSHOT
        assert action.params["format"] == "jpeg"

    def test_mouse_click_factory(self) -> None:
        action = ComputerAction.mouse_click(100, 200)
        assert action.action_type == ActionType.MOUSE_CLICK
        assert action.params["x"] == 100
        assert action.params["y"] == 200

    def test_keyboard_type_factory(self) -> None:
        action = ComputerAction.keyboard_type("hello world")
        assert action.params["text"] == "hello world"


class TestExtractProviderConfig:
    def test_anthropic(self) -> None:
        config = {
            "supported": True,
            "implementation": "screen_based",
            "provider_mapping": {
                "tool_type": "computer_20251124",
                "beta_header": "computer-use-2025-11-24",
            },
        }
        prov = extract_provider_config(config)
        assert prov is not None
        assert prov.tool_type == "computer_20251124"
        assert prov.implementation == ImplementationStyle.SCREEN_BASED

    def test_google_tool_based(self) -> None:
        config = {
            "supported": True,
            "implementation": "tool_based",
            "provider_mapping": {"tool_type": "computer_use"},
        }
        prov = extract_provider_config(config)
        assert prov is not None
        assert prov.implementation == ImplementationStyle.TOOL_BASED

    def test_unsupported(self) -> None:
        assert extract_provider_config({"supported": False}) is None
        assert extract_provider_config(None) is None
