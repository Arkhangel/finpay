"""CLI-смоук для self-hosted эмбеддингов (intfloat/multilingual-e5-base) — блок 5.1.

Два прогона:
1) Кеш — второй вызов embed_texts по тем же текстам не гоняет модель заново,
   latency заметно короче (диск-кеш экономит CPU-инференс).
2) E5-префиксы — без query:/passage: score между релевантной парой заметно
   ниже, чем с правильными префиксами (asymmetric retrieval).

    ENVIRONMENT=local uv run scripts/embeddings_smoke.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.embeddings import embed_documents, embed_query, embed_texts

BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "tests" / "eval" / "mini_benchmark.json"


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _texts_from_benchmark() -> list[str]:
    pairs = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    texts: list[str] = []
    for pair in pairs:
        texts.extend([pair["query"], pair["relevant"], pair["irrelevant"]])
    return texts


def cache_demo() -> None:
    texts = _texts_from_benchmark()

    start = time.perf_counter()
    vectors = embed_texts(texts)
    cold = time.perf_counter() - start

    start = time.perf_counter()
    embed_texts(texts)
    warm = time.perf_counter() - start

    print("--- Кеш ---")
    print(f"Текстов: {len(texts)}, размерность: {len(vectors[0])}")
    print(f"Холодный прогон: {cold:.3f} с")
    print(f"Тёплый прогон (кеш): {warm:.3f} с")


def e5_prefix_demo() -> None:
    """Сравнивает разделение relevant/irrelevant (gap) с префиксами и без —
    по всей mini_benchmark, а не по одной паре (единичный пример шумит)."""
    pairs = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))

    gaps_prefixed = []
    gaps_raw = []
    for pair in pairs:
        q_pfx = embed_query(pair["query"])
        rel_pfx = embed_documents([pair["relevant"]])[0]
        irr_pfx = embed_documents([pair["irrelevant"]])[0]
        gaps_prefixed.append(_cosine(q_pfx, rel_pfx) - _cosine(q_pfx, irr_pfx))

        q_raw = embed_texts([pair["query"]])[0]
        rel_raw = embed_texts([pair["relevant"]])[0]
        irr_raw = embed_texts([pair["irrelevant"]])[0]
        gaps_raw.append(_cosine(q_raw, rel_raw) - _cosine(q_raw, irr_raw))

    avg_prefixed = sum(gaps_prefixed) / len(gaps_prefixed)
    avg_raw = sum(gaps_raw) / len(gaps_raw)

    print("\n--- E5 query:/passage: префиксы ---")
    print(f"Пар в benchmark: {len(pairs)}")
    print(f"Средний gap (relevant - irrelevant) с префиксами: {avg_prefixed:.4f}")
    print(f"Средний gap (relevant - irrelevant) без префиксов: {avg_raw:.4f}")
    print(
        "Эффект на этом корпусе небольшой (короткие декларативные факты, лёгкие "
        "негативы) — заметнее он на длинных документах и трудных негативах."
    )


def main() -> None:
    cache_demo()
    e5_prefix_demo()


if __name__ == "__main__":
    main()