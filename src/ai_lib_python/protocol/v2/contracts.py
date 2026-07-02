"""Embedded ProviderContract YAML (synced from ai-protocol v2/contracts).

嵌入的 ProviderContract 真源；合规测试与 manifest encoder 共用。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ai_lib_python.protocol.v2.provider_contract import ProviderContract

_EMBEDDED_DIR = Path(__file__).resolve().parent / "embedded"


def _load_yaml(name: str) -> ProviderContract:
    path = _EMBEDDED_DIR / name
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid contract YAML: {name}")
    return ProviderContract.model_validate(data)


def anthropic_messages_contract() -> ProviderContract:
    return _load_yaml("anthropic-messages.contract.yaml")


def gemini_generate_contract() -> ProviderContract:
    return _load_yaml("gemini-generate.contract.yaml")


def contract_for_api_style(api_style: str) -> ProviderContract:
    if api_style == "anthropic_messages":
        return anthropic_messages_contract()
    if api_style == "gemini_generate":
        return gemini_generate_contract()
    raise ValueError(f"no embedded ProviderContract for api_style {api_style}")
