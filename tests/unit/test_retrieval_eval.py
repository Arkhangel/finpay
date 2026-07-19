"""Unit-тесты для app/services/retrieval_eval.py: чистые функции метрик на
синтетических id (без Qdrant/эмбеддингов) + evaluate_retrieval на фейковом
retriever'е.
"""
from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from app.services.retrieval_eval import (
    evaluate_retrieval,
    hit_rate_at_k,
    load_dataset,
    mrr_at_k,
    recall_at_k,
)


def test_hit_rate_at_k_hit_within_k():
    assert hit_rate_at_k(["a", "b", "c"], ["c"], k=5) == 1.0


def test_hit_rate_at_k_miss_outside_k():
    assert hit_rate_at_k(["a", "b", "c", "d", "e", "f"], ["f"], k=5) == 0.0


def test_hit_rate_at_k_no_relevant_retrieved():
    assert hit_rate_at_k(["a", "b"], ["z"], k=5) == 0.0


def test_mrr_at_k_first_position():
    assert mrr_at_k(["a", "b", "c"], ["a"], k=10) == 1.0


def test_mrr_at_k_third_position():
    assert mrr_at_k(["a", "b", "c"], ["c"], k=10) == 1.0 / 3


def test_mrr_at_k_not_found_returns_zero():
    assert mrr_at_k(["a", "b"], ["z"], k=10) == 0.0


def test_mrr_at_k_respects_k_cutoff():
    assert mrr_at_k(["a", "b", "c"], ["c"], k=2) == 0.0


def test_recall_at_k_partial_match():
    assert recall_at_k(["a", "b", "c"], ["a", "z"], k=10) == 0.5


def test_recall_at_k_full_match():
    assert recall_at_k(["a", "b"], ["a", "b"], k=10) == 1.0


def test_recall_at_k_empty_relevant_returns_zero():
    assert recall_at_k(["a", "b"], [], k=10) == 0.0


def _node(file_name: str, score: float) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text="x", metadata={"file_name": file_name}), score=score)


class _FakeRetriever:
    def __init__(self, answers: dict[str, list[NodeWithScore]]) -> None:
        self._answers = answers

    def retrieve(self, question: str) -> list[NodeWithScore]:
        return self._answers[question]


def test_evaluate_retrieval_averages_across_dataset():
    dataset = [
        {"question": "q1", "relevant_doc_ids": ["a.md"]},
        {"question": "q2", "relevant_doc_ids": ["z.md"]},
    ]
    retriever = _FakeRetriever(
        {
            "q1": [_node("a.md", 0.9), _node("b.md", 0.5)],
            "q2": [_node("b.md", 0.9), _node("c.md", 0.5)],
        }
    )

    metrics = evaluate_retrieval(retriever, dataset)

    assert metrics["hit_rate@5"] == 0.5
    assert metrics["mrr@10"] == 0.5
    assert metrics["recall@10"] == 0.5
    assert metrics["avg_latency_ms"] >= 0.0


def test_load_dataset_reads_schema(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(
        '[{"question": "q?", "relevant_doc_ids": ["a.md", "b.md"]}]', encoding="utf-8"
    )

    dataset = load_dataset(path)

    assert dataset == [{"question": "q?", "relevant_doc_ids": ["a.md", "b.md"]}]
