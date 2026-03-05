"""Computer Use 抽象层 — 提供跨厂商的 GUI 自动化操作标准化和安全控制。

Computer Use abstraction layer for AI-Protocol. Provides:
- Normalized action types across providers (screen_based, tool_based)
- Safety policy enforcement (confirmation, sandbox, logging, domain allowlist)
- Provider-specific configuration extraction
- Action validation before execution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlparse

# ─── Normalized Action Types ────────────────────────────────────────────────


class ActionType(str, Enum):
    """Normalized computer use action types."""

    SCREENSHOT = "screenshot"
    MOUSE_CLICK = "mouse_click"
    MOUSE_DOUBLE_CLICK = "mouse_double_click"
    MOUSE_DRAG = "mouse_drag"
    SCROLL = "scroll"
    MOUSE_MOVE = "mouse_move"
    KEYBOARD_TYPE = "keyboard_type"
    KEYBOARD_SHORTCUT = "keyboard_shortcut"
    BROWSER_NAVIGATE = "browser_navigate"
    BROWSER_CLICK_ELEMENT = "browser_click_element"
    BROWSER_FILL_FORM = "browser_fill_form"
    ZOOM_REGION = "zoom_region"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"


class MouseButton(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


@dataclass
class ComputerAction:
    """A normalized computer use action — provider-agnostic."""

    action_type: ActionType
    params: dict[str, Any] = field(default_factory=dict)

    # -- convenience factories --

    @classmethod
    def screenshot(cls, fmt: str = "png") -> ComputerAction:
        return cls(ActionType.SCREENSHOT, {"format": fmt})

    @classmethod
    def mouse_click(
        cls, x: float, y: float, button: MouseButton = MouseButton.LEFT
    ) -> ComputerAction:
        return cls(ActionType.MOUSE_CLICK, {"x": x, "y": y, "button": button.value})

    @classmethod
    def keyboard_type(cls, text: str) -> ComputerAction:
        return cls(ActionType.KEYBOARD_TYPE, {"text": text})

    @classmethod
    def keyboard_shortcut(cls, keys: list[str]) -> ComputerAction:
        return cls(ActionType.KEYBOARD_SHORTCUT, {"keys": keys})

    @classmethod
    def browser_navigate(cls, url: str) -> ComputerAction:
        return cls(ActionType.BROWSER_NAVIGATE, {"url": url})

    @classmethod
    def file_read(cls, path: str) -> ComputerAction:
        return cls(ActionType.FILE_READ, {"path": path})

    @classmethod
    def file_write(cls, path: str, content: str) -> ComputerAction:
        return cls(ActionType.FILE_WRITE, {"path": path, "content": content})


class ImplementationStyle(str, Enum):
    """Provider implementation approach."""

    SCREEN_BASED = "screen_based"
    TOOL_BASED = "tool_based"
    HYBRID = "hybrid"


class SandboxMode(str, Enum):
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


# ─── Safety Policy ──────────────────────────────────────────────────────────


class SafetyViolation(Exception):
    """Raised when a computer use action violates the safety policy."""


@dataclass
class SafetyPolicy:
    """Safety policy for computer use actions.

    Loaded from the manifest's ``computer_use.safety`` configuration.
    All validations are enforced *before* the action is dispatched.
    """

    confirmation_required: bool = True
    sandbox_mode: SandboxMode = SandboxMode.RECOMMENDED
    action_logging: bool = True
    domain_allowlist: set[str] = field(default_factory=set)
    sensitive_data_protection: bool = True
    max_actions_per_turn: int = 0
    action_timeout_ms: int = 30_000

    @classmethod
    def from_config(cls, safety_dict: dict[str, Any] | None) -> SafetyPolicy:
        """Build a safety policy from a manifest's ``computer_use.safety`` dict."""
        if not safety_dict:
            return cls()
        return cls(
            confirmation_required=safety_dict.get("confirmation_required", True),
            sandbox_mode=SandboxMode(safety_dict.get("sandbox_mode", "recommended")),
            action_logging=safety_dict.get("action_logging", True),
            domain_allowlist=set(safety_dict.get("domain_allowlist_entries", [])),
            sensitive_data_protection=safety_dict.get("sensitive_data_protection", True),
            max_actions_per_turn=safety_dict.get("max_actions_per_turn", 0),
            action_timeout_ms=safety_dict.get("action_timeout_ms", 30_000),
        )

    def validate_action(
        self,
        action: ComputerAction,
        actions_this_turn: int = 0,
    ) -> None:
        """Validate an action against this policy. Raises :class:`SafetyViolation`."""
        if self.max_actions_per_turn > 0 and actions_this_turn >= self.max_actions_per_turn:
            raise SafetyViolation(
                f"Max actions per turn exceeded: limit={self.max_actions_per_turn}, "
                f"attempted={actions_this_turn + 1}"
            )

        if action.action_type == ActionType.BROWSER_NAVIGATE and self.domain_allowlist:
            url = action.params.get("url", "")
            domain = _extract_domain(url)
            if domain not in self.domain_allowlist:
                raise SafetyViolation(
                    f"Domain '{domain}' is not in the allowlist: {sorted(self.domain_allowlist)}"
                )

        if self.sensitive_data_protection and action.action_type in (
            ActionType.FILE_READ,
            ActionType.FILE_WRITE,
        ):
            path = action.params.get("path", "")
            if _is_sensitive_path(path):
                raise SafetyViolation(f"Access to sensitive path '{path}' is blocked")


# ─── Provider Configuration ─────────────────────────────────────────────────


@dataclass
class CuProviderConfig:
    """Provider-specific computer use configuration."""

    tool_type: str = "computer_use"
    beta_header: str | None = None
    implementation: ImplementationStyle = ImplementationStyle.SCREEN_BASED
    model_requirement: str | None = None


def extract_provider_config(cu_config: dict[str, Any] | None) -> CuProviderConfig | None:
    """Extract provider-specific CU configuration from a manifest section."""
    if not cu_config or not cu_config.get("supported"):
        return None

    impl_str = cu_config.get("implementation", "screen_based")
    implementation = ImplementationStyle(impl_str)

    mapping = cu_config.get("provider_mapping", {})
    return CuProviderConfig(
        tool_type=mapping.get("tool_type", "computer_use"),
        beta_header=mapping.get("beta_header"),
        implementation=implementation,
        model_requirement=mapping.get("model_requirement"),
    )


# ─── Helpers ────────────────────────────────────────────────────────────────

_SENSITIVE_PATTERNS = (
    ".ssh", ".gnupg", ".aws", "credentials", "secrets",
    ".env", "password", "token", ".kube/config",
)


def _extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.hostname or ""
    except Exception:
        return url.split("//")[-1].split("/")[0].split(":")[0]


def _is_sensitive_path(path: str) -> bool:
    lower = path.lower()
    return any(p in lower for p in _SENSITIVE_PATTERNS)


__all__ = [
    "ActionType",
    "ComputerAction",
    "CuProviderConfig",
    "ImplementationStyle",
    "MouseButton",
    "SafetyPolicy",
    "SafetyViolation",
    "SandboxMode",
    "extract_provider_config",
]
