"""能力注册表 — 根据 Manifest 声明动态加载和管理运行时模块。

Capability registry that dynamically checks which capabilities are available
based on V2 Manifest declarations and installed pip extras.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ai_lib_python.protocol.v2.capabilities import CapabilitiesV2, Capability


class CapabilityStatus(str, Enum):
    """Status of a capability in the registry."""

    ACTIVE_REQUIRED = "active_required"
    ACTIVE_OPTIONAL = "active_optional"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class CapabilityGap:
    """Describes why a capability is unavailable."""

    capability: Capability
    required: bool
    extra_name: str | None

    def __str__(self) -> str:
        if self.extra_name:
            return (
                f"Capability {self.capability.value!r} is required but not available. "
                f"Install with: pip install ai-lib-python[{self.extra_name}]"
            )
        return f"Capability {self.capability.value!r} is required but not available"


class CapabilityRegistry:
    """Runtime capability registry — tracks declared vs. available capabilities.

    Usage::

        caps = CapabilitiesV2(required=[Capability.TEXT, Capability.STREAMING],
                              optional=[Capability.VISION])
        registry = CapabilityRegistry.from_capabilities(caps)
        registry.validate_requirements()   # raises if required caps are missing
        registry.is_active(Capability.VISION)  # True if installed
    """

    def __init__(
        self,
        required: set[Capability],
        optional: set[Capability],
        available: set[Capability],
    ) -> None:
        self._required = frozenset(required)
        self._optional = frozenset(optional)
        self._available = frozenset(available)

    @classmethod
    def from_capabilities(cls, caps: CapabilitiesV2) -> CapabilityRegistry:
        """Build a registry from a V2 capabilities declaration."""
        required = set(caps.required_capabilities())
        all_caps = set(caps.all_capabilities())
        optional = all_caps - required
        available = cls._detect_available()
        return cls(required=required, optional=optional, available=available)

    # ---- queries -------------------------------------------------------

    def validate_requirements(self) -> None:
        """Raise :class:`RuntimeError` if any required capability is missing."""
        gaps = self.gaps()
        if gaps:
            lines = [str(g) for g in gaps]
            raise RuntimeError(
                "Missing required capabilities:\n" + "\n".join(f"  - {line}" for line in lines)
            )

    def gaps(self) -> list[CapabilityGap]:
        """Return the list of required-but-unavailable capabilities."""
        return [
            CapabilityGap(
                capability=cap,
                required=True,
                extra_name=cap.extra_name,
            )
            for cap in self._required
            if cap not in self._available
        ]

    def active_capabilities(self) -> frozenset[Capability]:
        """Return capabilities that are both declared and available."""
        return (self._required | self._optional) & self._available

    def is_active(self, cap: Capability) -> bool:
        """Check if *cap* is usable (declared and installed)."""
        return cap in (self._required | self._optional) and cap in self._available

    def status_report(self) -> dict[Capability, CapabilityStatus]:
        """Get a human-readable status map for all declared capabilities."""
        report: dict[Capability, CapabilityStatus] = {}
        for cap in self._required | self._optional:
            if cap in self._available:
                status = (
                    CapabilityStatus.ACTIVE_REQUIRED
                    if cap in self._required
                    else CapabilityStatus.ACTIVE_OPTIONAL
                )
            else:
                status = CapabilityStatus.UNAVAILABLE
            report[cap] = status
        return report

    # ---- capability detection ------------------------------------------

    @staticmethod
    def _detect_available() -> set[Capability]:
        """Detect which capabilities are available in the current environment."""
        available: set[Capability] = set()

        # Core — always available
        available.update({
            Capability.TEXT,
            Capability.STREAMING,
            Capability.TOOLS,
            Capability.PARALLEL_TOOLS,
            Capability.AGENTIC,
            Capability.STRUCTURED_OUTPUT,
        })

        # Vision (requires pillow)
        if _importable("PIL"):
            available.add(Capability.VISION)

        # Audio/Video (requires soundfile)
        if _importable("soundfile"):
            available.add(Capability.AUDIO)
            available.add(Capability.VIDEO)

        # Embeddings — always available (thin wrapper)
        available.add(Capability.EMBEDDINGS)

        # Batch — always available
        available.add(Capability.BATCH)

        # MCP — check for mcp package
        if _importable("mcp"):
            available.add(Capability.MCP_CLIENT)
            available.add(Capability.MCP_SERVER)

        # Reasoning — always available (protocol-driven, no extra dep)
        available.add(Capability.REASONING)

        return available


def _importable(module_name: str) -> bool:
    """Check if a Python module is importable without side effects."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


__all__ = [
    "CapabilityGap",
    "CapabilityRegistry",
    "CapabilityStatus",
]
