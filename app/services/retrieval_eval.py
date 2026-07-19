"""Retrieval-метрики для сравнения chunking-стратегий (блок 5.4).

Golden dataset — tests/eval/retrieval_dataset.json, схема:
{"question": str, "relevant_doc_ids": [str, ...]}, где id — имя файла из
data/rag-block-03/. Источник doc_id у retrieved-ноды — metadata["file_name"],
которую заполняет SimpleDirectoryReader (тот же ключ, что уже использует
app/services/rag.py::RAGService.answer для sources).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol

from llama_index.core.schema import NodeWithScore


class Retriever(Protocol):
    def retrieve(self, question: str) -> list[NodeWithScore]: ...


def load_dataset(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _retrieved_doc_ids(nodes: list[NodeWithScore]) -> list[str]:
    ids: list[str] = []
    for node in nodes:
        file_name = node.node.metadata.get("file_name")
        if file_name is not None and file_name not in ids:
            ids.append(file_name)
    return ids


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """1.0, если хотя бы один релевантный doc_id попал в top-k, иначе 0.0."""
    top_k = retrieved_ids[:k]
    return 1.0 if any(doc_id in top_k for doc_id in relevant_ids) else 0.0


def mrr_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 10) -> float:
    """1 / позиция первого релевантного doc_id в top-k (0, если не найден)."""
    top_k = retrieved_ids[:k]
    for position, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_ids:
            return 1.0 / position
    return 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 10) -> float:
    """Доля релевантных doc_id, попавших в top-k."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for doc_id in relevant_ids if doc_id in top_k)
    return hits / len(relevant_ids)


def evaluate_retrieval(retriever: Retriever, dataset: list[dict]) -> dict:
    """Гоняет retriever по всему golden dataset, возвращает усреднённые
    Hit Rate@5 / MRR@10 / Recall@10 + среднюю латентность retrieve в мс.
    """
    hit_rates: list[float] = []
    mrrs: list[float] = []
    recalls: list[float] = []
    latencies_ms: list[float] = []

    for item in dataset:
        question = item["question"]
        relevant_ids = item["relevant_doc_ids"]

        start = time.perf_counter()
        nodes = retriever.retrieve(question)
        latencies_ms.append((time.perf_counter() - start) * 1000)

        retrieved_ids = _retrieved_doc_ids(nodes)
        hit_rates.append(hit_rate_at_k(retrieved_ids, relevant_ids, k=5))
        mrrs.append(mrr_at_k(retrieved_ids, relevant_ids, k=10))
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, k=10))

    n = len(dataset) or 1
    return {
        "hit_rate@5": sum(hit_rates) / n,
        "mrr@10": sum(mrrs) / n,
        "recall@10": sum(recalls) / n,
        "avg_latency_ms": sum(latencies_ms) / n,
    }
