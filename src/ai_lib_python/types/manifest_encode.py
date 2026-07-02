"""Manifest-driven ContentBlock encoding (PT-079 / ALP-DOC-001).

按 ProviderContract `content_block_mapping` 将统一 ContentBlock 编码为厂商 wire JSON。
"""

from __future__ import annotations

from typing import Any

from ai_lib_python.protocol.v2.contracts import (
    anthropic_messages_contract,
    contract_for_api_style,
    gemini_generate_contract,
)
from ai_lib_python.protocol.v2.provider_contract import DocumentBlockMapping, ProviderContract


def _document_mapping(contract: ProviderContract) -> DocumentBlockMapping:
    mapping = contract.request_mapping.content_block_mapping
    if mapping is None or mapping.document is None:
        raise ValueError(
            f"ProviderContract {contract.provider_id} missing content_block_mapping.document"
        )
    return mapping.document


def encode_blocks_anthropic(
    contract: ProviderContract,
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if contract.api_style != "anthropic_messages":
        raise ValueError(f"expected anthropic_messages contract, got {contract.api_style}")
    doc_mapping = _document_mapping(contract)
    return [_encode_anthropic_block(block, doc_mapping) for block in blocks]


def encode_blocks_gemini(
    contract: ProviderContract,
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if contract.api_style != "gemini_generate":
        raise ValueError(f"expected gemini_generate contract, got {contract.api_style}")
    doc_mapping = _document_mapping(contract)
    return [_encode_gemini_block(block, doc_mapping) for block in blocks]


def encode_blocks_for_anthropic_contract(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return encode_blocks_anthropic(anthropic_messages_contract(), blocks)


def encode_blocks_for_gemini_contract(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return encode_blocks_gemini(gemini_generate_contract(), blocks)


def encode_blocks_for_api_style(
    api_style: str, blocks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    contract = contract_for_api_style(api_style)
    if api_style == "anthropic_messages":
        return encode_blocks_anthropic(contract, blocks)
    if api_style == "gemini_generate":
        return encode_blocks_gemini(contract, blocks)
    raise ValueError(f"unsupported api_style: {api_style}")


def _encode_anthropic_block(
    block: dict[str, Any], doc_mapping: DocumentBlockMapping
) -> dict[str, Any]:
    block_type = str(block.get("block_type", "text"))
    if block_type == "text":
        return {"type": "text", "text": str(block.get("text", ""))}
    if block_type != "document":
        raise ValueError(f"unsupported block_type: {block_type}")
    return _encode_anthropic_document(block, doc_mapping)


def _encode_anthropic_document(
    block: dict[str, Any],
    mapping: DocumentBlockMapping,
) -> dict[str, Any]:
    if mapping.format != "anthropic_document":
        raise ValueError(f"unsupported document format for Anthropic: {mapping.format}")
    source_type = str(block.get("source_type", "base64"))
    if source_type == "ref" and mapping.rejects_ref_before_encode():
        raise ValueError(
            "document ref must be resolved to base64 or url before sending to Anthropic"
        )
    if source_type != "base64":
        raise ValueError(f"unsupported document source_type: {source_type}")
    type_field = mapping.type_field or "document"
    return {
        "type": type_field,
        "source": {
            "type": "base64",
            "media_type": str(block.get("mime_type", mapping.default_mime())),
            "data": str(block.get("data", "")),
        },
    }


def _encode_gemini_block(
    block: dict[str, Any], doc_mapping: DocumentBlockMapping
) -> dict[str, Any]:
    block_type = str(block.get("block_type", "text"))
    if block_type == "text":
        return {"text": str(block.get("text", ""))}
    if block_type != "document":
        raise ValueError(f"unsupported block_type: {block_type}")
    return _encode_gemini_document(block, doc_mapping)


def _encode_gemini_document(block: dict[str, Any], mapping: DocumentBlockMapping) -> dict[str, Any]:
    if mapping.format != "gemini_inline_data":
        raise ValueError(f"unsupported document format for Gemini: {mapping.format}")
    source_type = str(block.get("source_type", "base64"))
    if source_type == "ref" and mapping.rejects_ref_before_encode():
        raise ValueError(
            "Gemini document blocks require base64 inline data; resolve ref before send"
        )
    if source_type != "base64":
        raise ValueError(f"unsupported document source_type: {source_type}")
    return {
        "inlineData": {
            "mimeType": str(block.get("mime_type", mapping.default_mime())),
            "data": str(block.get("data", "")),
        }
    }
