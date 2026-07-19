"""Эксперимент chunking + retrieval eval — блок 5.4, задачи 3, 5, 6, 7.

Индексирует корпус `data/rag-block-03/` тремя стратегиями chunking
(`app/services/chunking.py`) в три отдельные коллекции Qdrant
(`docs_fixed`/`docs_recursive`/`docs_semantic`), считает Hit Rate@5 / MRR@10 /
Recall@10 / скорость retrieval по golden dataset
(`tests/eval/retrieval_dataset.json`), сравнивает лучшую по Hit Rate@5
стратегию до/после re-ranker (`app/services/reranker.py`) и подбирает
`(chunk_size, overlap, top_k)` на лучшей стратегии (grid search).

Печатает всё в консоль — реальные числа отсюда вручную переносятся в
`docs/chunking_experiment.md`, без подгонки под ожидаемый результат (тот же
принцип, что в `scripts/qdrant_experiments.py` + `docs/vector_store.md`).

    ENVIRONMENT=local uv run scripts/chunking_experiment.py

Требует поднятый Qdrant (`docker compose up -d qdrant`). Для песочниц без
доступного docker-демона можно переключиться на embedded Qdrant без сервера:

    CHUNKING_EXPERIMENT_QDRANT_PATH=/tmp/qdrant_local uv run scripts/chunking_experiment.py

— тот же qdrant-client API (collections/points/query), отличается только
backend хранения; сравнение chunking-стратегий это не меняет.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core import Settings as LlamaSettings
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.services import chunking
from app.services.reranker import Reranker
from app.services.retrieval_eval import evaluate_retrieval, load_dataset
from app.settings import settings

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "tests/eval/retrieval_dataset.json"

COLLECTIONS = {
    "fixed_size": "docs_fixed",
    "recursive": "docs_recursive",
    "semantic": "docs_semantic",
}
# Hit Rate@5/MRR@10/Recall@10 режут по top-10 кандидатов — retriever должен
# отдавать минимум top-10, иначе метрики молча вырождаются в @top_k продакшена
# (settings.rag.similarity_top_k=3). Сравнение стратегий и re-ranker считаются
# на этой глубине; top_k из настроек — отдельный параметр задачи 7.
EVAL_TOP_K = 10
RERANK_CANDIDATE_K = 20  # top-N кандидатов до re-ranker, из которых он выбирает top_n


def _qdrant_client() -> QdrantClient:
    local_path = os.getenv("CHUNKING_EXPERIMENT_QDRANT_PATH")
    if local_path:
        return QdrantClient(path=local_path)
    return QdrantClient(url=settings.qdrant.url, api_key=settings.qdrant.api_key or None)


class RerankingRetriever:
    """Оборачивает базовый retriever: берёт RERANK_CANDIDATE_K кандидатов,
    пересортировывает Reranker'ом, обрезает до top_k."""

    def __init__(self, base_retriever, reranker: Reranker, top_k: int) -> None:
        self._base = base_retriever
        self._reranker = reranker
        self._top_k = top_k

    def retrieve(self, question: str):
        candidates = self._base.retrieve(question)
        return self._reranker.rerank(question, candidates, top_n=self._top_k)


def build_nodes(strategy: str, documents, embed_model, *, chunk_size: int = 512, chunk_overlap: int = 64):
    if strategy == "semantic":
        return chunking.semantic(documents, embed_model=embed_model)
    fn = chunking.STRATEGIES[strategy]
    return fn(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def chunk_stats(nodes, n_documents: int) -> dict:
    lengths = [len(n.get_content()) for n in nodes]
    return {
        "total_chunks": len(nodes),
        "avg_chunks_per_doc": len(nodes) / n_documents if n_documents else 0.0,
        "avg_chunk_len": sum(lengths) / len(lengths) if lengths else 0.0,
    }


def build_index(client: QdrantClient, collection_name: str, nodes, embed_model) -> VectorStoreIndex:
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        client.delete_collection(collection_name)

    vector_store = QdrantVectorStore(collection_name=collection_name, client=client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)


def index_nodes(client: QdrantClient, collection_name: str, nodes, embed_model, top_k: int):
    index = build_index(client, collection_name, nodes, embed_model)
    return index.as_retriever(similarity_top_k=top_k, embed_model=embed_model)


def print_row(label: str, metrics: dict, avg_chunk_len: float | None = None) -> None:
    chunk_len = f"{avg_chunk_len:.0f}" if avg_chunk_len is not None else "-"
    print(
        f"{label:<28} | hit_rate@5={metrics['hit_rate@5']:.3f} | "
        f"mrr@10={metrics['mrr@10']:.3f} | recall@10={metrics['recall@10']:.3f} | "
        f"avg_chunk_len={chunk_len} | latency={metrics['avg_latency_ms']:.1f}ms"
    )


def main() -> None:
    dataset = load_dataset(DATASET_PATH)
    documents = SimpleDirectoryReader(input_dir=settings.rag.corpus_dir, recursive=True).load_data()
    print(f"Корпус: {len(documents)} документов, golden dataset: {len(dataset)} вопросов\n")

    embed_model = chunking.build_embed_model(settings)
    LlamaSettings.embed_model = embed_model  # SemanticSplitterNodeParser и VectorStoreIndex должны совпадать

    client = _qdrant_client()

    print("--- Задачи 3-5: три стратегии chunking ---")
    results: dict[str, dict] = {}
    stats: dict[str, dict] = {}
    for strategy, collection in COLLECTIONS.items():
        nodes = build_nodes(strategy, documents, embed_model)
        stats[strategy] = chunk_stats(nodes, len(documents))
        retriever = index_nodes(client, collection, nodes, embed_model, EVAL_TOP_K)
        metrics = evaluate_retrieval(retriever, dataset)
        results[strategy] = metrics
        print(
            f"{strategy:<12} chunks: total={stats[strategy]['total_chunks']:>3} "
            f"avg/doc={stats[strategy]['avg_chunks_per_doc']:.1f} "
            f"avg_len={stats[strategy]['avg_chunk_len']:.0f}"
        )
        print_row(f"  {strategy}", metrics, stats[strategy]["avg_chunk_len"])
    print()

    # На малом корпусе Hit Rate@5/Recall@10 легко насыщаются до 1.0 у всех
    # стратегий сразу (см. docs/chunking_experiment.md) — тай-брейк по MRR@10,
    # который чувствителен к порядку внутри top-K, а не только к факту попадания.
    best_strategy = max(
        results, key=lambda s: (results[s]["hit_rate@5"], results[s]["mrr@10"], results[s]["recall@10"])
    )
    print(f"Лучшая стратегия (Hit Rate@5, тай-брейк по MRR@10): {best_strategy}\n")

    print("--- Задача 6: re-ranker на лучшей стратегии ---")
    reranker = Reranker()
    best_nodes = build_nodes(best_strategy, documents, embed_model)
    best_index = build_index(client, COLLECTIONS[best_strategy], best_nodes, embed_model)

    # before: тот же EVAL_TOP_K, что и в сравнении стратегий выше — иначе
    # сравнение до/после re-ranker считалось бы на разной глубине.
    before_retriever = best_index.as_retriever(similarity_top_k=EVAL_TOP_K, embed_model=embed_model)
    before_metrics = evaluate_retrieval(before_retriever, dataset)

    # after: берём RERANK_CANDIDATE_K кандидатов из того же индекса, re-ranker обрезает до EVAL_TOP_K
    candidate_retriever = best_index.as_retriever(similarity_top_k=RERANK_CANDIDATE_K, embed_model=embed_model)
    reranking_retriever = RerankingRetriever(candidate_retriever, reranker, EVAL_TOP_K)
    after_metrics = evaluate_retrieval(reranking_retriever, dataset)

    print_row(f"{best_strategy} (before rerank)", before_metrics, stats[best_strategy]["avg_chunk_len"])
    print_row(f"{best_strategy} (after rerank)", after_metrics, stats[best_strategy]["avg_chunk_len"])
    print()

    print("--- Задача 7: grid search (chunk_size, overlap, top-K=10/20) ---")
    if best_strategy == "semantic":
        print("Лучшая стратегия — semantic: у неё нет chunk_size/overlap, поэтому")
        print("сетка идёт по (breakpoint_percentile_threshold, top_k) вместо этого.")
        grid = [(90, 10), (95, 10), (90, 20), (95, 20)]
        for threshold, k in grid:
            from llama_index.core.node_parser import SemanticSplitterNodeParser

            splitter = SemanticSplitterNodeParser(
                buffer_size=1, breakpoint_percentile_threshold=threshold, embed_model=embed_model
            )
            nodes = splitter.get_nodes_from_documents(documents)
            retriever = index_nodes(client, "docs_grid_search", nodes, embed_model, k)
            metrics = evaluate_retrieval(retriever, dataset)
            print_row(f"threshold={threshold} top_k={k}", metrics, chunk_stats(nodes, len(documents))["avg_chunk_len"])
        client.delete_collection("docs_grid_search")
    else:
        grid = [
            (chunk_size, overlap, k)
            for chunk_size in (256, 512)
            for overlap in (32, 64)
            for k in (10, 20)
        ]
        for chunk_size, overlap, k in grid:
            nodes = build_nodes(best_strategy, documents, embed_model, chunk_size=chunk_size, chunk_overlap=overlap)
            retriever = index_nodes(client, "docs_grid_search", nodes, embed_model, k)
            metrics = evaluate_retrieval(retriever, dataset)
            print_row(
                f"size={chunk_size} overlap={overlap} top_k={k}",
                metrics,
                chunk_stats(nodes, len(documents))["avg_chunk_len"],
            )
        client.delete_collection("docs_grid_search")

    client.close()


if __name__ == "__main__":
    main()
