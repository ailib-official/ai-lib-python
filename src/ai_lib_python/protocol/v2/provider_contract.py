"""ProviderContract types for manifest-driven content block encoding (PT-079 / ALP-DOC-001).

与 ai-protocol `schemas/v2/provider-contract.json` 编码相关字段对齐的子集。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentBlockMapping(BaseModel):
    """Document block mapping per PT-079-R1 schema."""

    format: str
    type_field: str | None = None
    default_mime_type: str | None = None
    ref_resolution: str | None = None

    def default_mime(self) -> str:
        return self.default_mime_type or "application/pdf"

    def rejects_ref_before_encode(self) -> bool:
        return (self.ref_resolution or "error_before_encode") == "error_before_encode"


class ContentBlockMapping(BaseModel):
    text: dict[str, str] | None = None
    image: dict[str, str] | None = None
    document: DocumentBlockMapping | None = None


class RequestMappingContract(BaseModel):
    message_format: str
    role_mapping: dict[str, str] = Field(default_factory=dict)
    content_block_mapping: ContentBlockMapping | None = None


class ProviderContract(BaseModel):
    contract_version: str
    provider_id: str
    api_style: str
    request_mapping: RequestMappingContract
