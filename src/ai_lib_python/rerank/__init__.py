"""重排序模块：封装文档重排序能力。

Rerank module.

Provides document reranking by relevance via provider APIs (e.g. Cohere Rerank).
"""

from ai_lib_python.rerank.client import (
    RerankerClient,
    RerankerClientBuilder,
    RerankOptions,
    RerankResult,
)

__all__ = [
    "RerankOptions",
    "RerankResult",
    "RerankerClient",
    "RerankerClientBuilder",
]
