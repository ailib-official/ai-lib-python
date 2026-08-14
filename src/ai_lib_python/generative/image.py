"""Experimental image generation via manifest L-Exec (ALP-GEN-001 / ALR-GEN-002).

按 ``endpoints.image_generation`` 走统一 HttpTransport；adapter 来自 manifest。
"""

from __future__ import annotations

from typing import Any

from ai_lib_python.errors import ValidationError
from ai_lib_python.generative.endpoints import (
    KEY_IMAGE_GENERATION,
    adapter_name,
    require_generative_endpoint,
)
from ai_lib_python.generative.types import (
    GeneratedImage,
    ImageGenerationRequest,
    ImageGenerationResult,
)
from ai_lib_python.protocol.manifest import ProtocolManifest
from ai_lib_python.transport import HttpTransport
from ai_lib_python.transport.auth import resolve_credential


def openai_image_body(req: ImageGenerationRequest) -> dict[str, Any]:
    """OpenAI Images request shape."""
    body: dict[str, Any] = {"model": req.model, "prompt": req.prompt}
    if req.size:
        body["size"] = req.size
    if req.n is not None:
        body["n"] = req.n
    if req.response_format:
        body["response_format"] = req.response_format
    return body


def dashscope_image_body(req: ImageGenerationRequest) -> dict[str, Any]:
    """DashScope native multimodal-generation shape (PT-GEN-003)."""
    return {
        "model": req.model,
        "input": {
            "messages": [{"role": "user", "content": [{"text": req.prompt}]}],
        },
    }


def parse_openai_image(model: str, payload: dict[str, Any]) -> ImageGenerationResult:
    images: list[GeneratedImage] = []
    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            b64 = item.get("b64_json")
            images.append(
                GeneratedImage(
                    url=url if isinstance(url, str) else None,
                    b64_json=b64 if isinstance(b64, str) else None,
                )
            )
    return ImageGenerationResult(model=model, images=images)


def parse_dashscope_image(model: str, payload: dict[str, Any]) -> ImageGenerationResult:
    images: list[GeneratedImage] = []
    try:
        url = payload["output"]["choices"][0]["message"]["content"][0]["image"]
        if isinstance(url, str):
            images.append(GeneratedImage(url=url))
    except (KeyError, IndexError, TypeError):
        try:
            url = payload["output"]["results"][0]["url"]
            if isinstance(url, str):
                images.append(GeneratedImage(url=url))
        except (KeyError, IndexError, TypeError):
            pass
    return ImageGenerationResult(model=model, images=images)


class ImageGenerationClient:
    """Experimental image generation client (capability: ``image_generation``)."""

    def __init__(
        self,
        *,
        transport: HttpTransport,
        model: str,
        endpoint_path: str,
        adapter: str,
    ) -> None:
        self._transport = transport
        self._model = model
        self._endpoint_path = endpoint_path
        self._adapter = adapter

    @classmethod
    def from_manifest(cls, manifest: ProtocolManifest, model: str) -> ImageGenerationClient:
        ep = require_generative_endpoint(manifest, model, KEY_IMAGE_GENERATION)
        resolved = resolve_credential(manifest.id, manifest, None)
        if not resolved.secret:
            tried = list(resolved.required_envs) + list(resolved.conventional_envs)
            raise ValidationError(
                f"API key required for image_generation (provider={manifest.id}; tried {tried})"
            )
        transport = HttpTransport(
            manifest,
            model_id=model,
            api_key=resolved.secret,
        )
        return cls(
            transport=transport,
            model=model,
            endpoint_path=str(ep["path"]),
            adapter=adapter_name(ep),
        )

    @property
    def endpoint_path(self) -> str:
        return self._endpoint_path

    @property
    def adapter(self) -> str:
        return self._adapter

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        if request.model != self._model:
            raise ValidationError(
                f"request model `{request.model}` != client model `{self._model}`"
            )
        body = (
            dashscope_image_body(request)
            if self._adapter == "dashscope"
            else openai_image_body(request)
        )
        response = await self._transport.post(self._endpoint_path, json=body)
        payload = response.json()
        if self._adapter == "dashscope":
            return parse_dashscope_image(self._model, payload)
        return parse_openai_image(self._model, payload)
