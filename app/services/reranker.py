"""Re-ranker поверх retrieval-кандидатов (блок 5.4).

BAAI/bge-reranker-v2-m3 через sentence-transformers.CrossEncoder — локальный
cross-encoder, без внешнего API/ключа (согласуется с self-hosted embeddings,
ADR-003 в docs/architecture.md). Ленивая загрузка модели — по аналогии с
app/services/embeddings.py::_get_model.
"""

from __future__ import annotations

import logging

from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        logger.info("reranker_model_loading", extra={"model": _MODEL_NAME})
        _model = CrossEncoder(_MODEL_NAME)
    return _model


class Reranker:
    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model_name = model_name

    def rerank(
        self, query: str, candidates: list[NodeWithScore], top_n: int
    ) -> list[NodeWithScore]:
        if not candidates:
            return []

        model = _get_model()
        pairs = [(query, node.node.get_content()) for node in candidates]
        scores = model.predict(pairs)

        reranked = [
            NodeWithScore(node=node.node, score=float(score))
            for node, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda n: n.score, reverse=True)
        return reranked[:top_n]
