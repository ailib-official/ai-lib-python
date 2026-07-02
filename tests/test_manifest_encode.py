"""Unit tests for manifest-driven content block encoder (ALP-DOC-001)."""

from __future__ import annotations

import pytest

from ai_lib_python.types.manifest_encode import (
    encode_blocks_for_anthropic_contract,
    encode_blocks_for_gemini_contract,
)

_PDF_B64 = "JVBERi0xLjQK"


def test_anthropic_document_base64() -> None:
    encoded = encode_blocks_for_anthropic_contract(
        [
            {
                "block_type": "document",
                "source_type": "base64",
                "data": _PDF_B64,
                "mime_type": "application/pdf",
            }
        ]
    )
    assert encoded == [
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": _PDF_B64,
            },
        }
    ]


def test_gemini_document_inline() -> None:
    encoded = encode_blocks_for_gemini_contract(
        [
            {"block_type": "text", "text": "Summarize"},
            {
                "block_type": "document",
                "source_type": "base64",
                "data": _PDF_B64,
                "mime_type": "application/pdf",
            },
        ]
    )
    assert encoded == [
        {"text": "Summarize"},
        {"inlineData": {"mimeType": "application/pdf", "data": _PDF_B64}},
    ]


def test_anthropic_document_ref_rejected() -> None:
    with pytest.raises(ValueError, match="ref must be resolved"):
        encode_blocks_for_anthropic_contract(
            [
                {
                    "block_type": "document",
                    "source_type": "ref",
                    "data": "upload://abc",
                    "mime_type": "application/pdf",
                }
            ]
        )
