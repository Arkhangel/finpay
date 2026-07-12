"""Эксперименты для docs/vector_store.md — блок 5.2, задачи 5 и 6.

Предполагает, что `scripts/load_to_qdrant.py` уже загрузил коллекцию
`settings.qdrant.collection` (documents). Печатает всё в консоль — реальные
числа отсюда переносятся в docs/vector_store.md, без подгонки под ожидаемый
результат.

Задача 5 (cosine vs dot): создаёт две временные коллекции на тех же векторах,
что уже лежат в `documents`, сравнивает top-5 ранжирование на 5 запросах,
удаляет обе коллекции в конце.

Задача 6 (фильтры): 3 примера на основной коллекции — match по category,
range по created_at (демонстрируется на паре active/archived тарифа из
load_to_qdrant.py), составной must+must_not.

    ENVIRONMENT=local uv run scripts/qdrant_experiments.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.services.embeddings import embed_query
from app.settings import settings

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "tests/eval/mini_benchmark.json"
COSINE_COLLECTION = "documents_cosine"
DOT_COLLECTION = "documents_dot"


def _client() -> AsyncQdrantClient:
    return AsyncQdrantClient(url=settings.qdrant.url, api_key=settings.qdrant.api_key or None)


async def _scroll_all_points(client: AsyncQdrantClient) -> list[PointStruct]:
    points: list[PointStruct] = []
    offset = None
    while True:
        batch, offset = await client.scroll(
            collection_name=settings.qdrant.collection,
            with_payload=True,
            with_vectors=True,
            limit=256,
            offset=offset,
        )
        points.extend(
            PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in batch
        )
        if offset is None:
            break
    return points


def _sample_queries(n: int = 5) -> list[str]:
    pairs = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    return [p["query"] for p in pairs[:n]]


async def cosine_vs_dot(client: AsyncQdrantClient) -> None:
    points = await _scroll_all_points(client)
    dim = settings.embeddings.dim

    for name, distance in [(COSINE_COLLECTION, Distance.COSINE), (DOT_COLLECTION, Distance.DOT)]:
        existing = {c.name for c in (await client.get_collections()).collections}
        if name in existing:
            await client.delete_collection(name)
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=distance),
        )
        await client.upsert(collection_name=name, points=points, wait=True)

    print("\n--- Задача 5: cosine vs dot ---")
    print(f"{'query':<55} | {'top-5 cosine == top-5 dot?':<28}")
    for query in _sample_queries(5):
        vector = embed_query(query)
        cosine_res = await client.query_points(
            collection_name=COSINE_COLLECTION, query=vector, limit=5, with_payload=False
        )
        dot_res = await client.query_points(
            collection_name=DOT_COLLECTION, query=vector, limit=5, with_payload=False
        )
        cosine_ids = [p.id for p in cosine_res.points]
        dot_ids = [p.id for p in dot_res.points]
        match = cosine_ids == dot_ids
        print(f"{query[:53]:<55} | {'ДА' if match else 'НЕТ — ' + str(dot_ids)}")
        print(f"  cosine top-5: {cosine_ids}")
        print(f"  dot    top-5: {dot_ids}")

    await client.delete_collection(COSINE_COLLECTION)
    await client.delete_collection(DOT_COLLECTION)
    print(f"\nВременные коллекции {COSINE_COLLECTION}/{DOT_COLLECTION} удалены.")


def _print_results(points, label: str) -> None:
    print(f"\n{label}")
    for p in points:
        payload = p.payload or {}
        print(f"  score={p.score:.4f} source={payload.get('source')} status={payload.get('status')}")


async def filter_examples(client: AsyncQdrantClient) -> None:
    print("\n--- Задача 6: фильтрация по metadata ---")

    query = "Какая стандартная комиссия за транзакцию?"
    vector = embed_query(query)

    # 1) Match по строке — category="tariffs"
    match_filter = Filter(must=[FieldCondition(key="category", match=MatchValue(value="tariffs"))])
    res = await client.query_points(
        collection_name=settings.qdrant.collection,
        query=vector,
        query_filter=match_filter,
        limit=3,
        with_payload=True,
    )
    print(f"\n1) Match: category=\"tariffs\", запрос: {query!r}")
    _print_results(res.points, "top-3:")

    # 2) Range по дате — внутри category="tariffs" (иначе эффект не виден: на
    # общем запросе golden_dataset-факты и так обгоняют архивный тариф по
    # чистой релевантности, а вот внутри тарифов архивный документ ранжируется
    # первым — см. пример 1 — и без date-фильтра туда и попадает).
    print(f"\n2) Range: category=\"tariffs\" без date-фильтра vs с ним, запрос: {query!r}")
    res_nofilter = await client.query_points(
        collection_name=settings.qdrant.collection,
        query=vector,
        query_filter=match_filter,
        limit=3,
        with_payload=True,
    )
    _print_results(res_nofilter.points, "top-3 (category=tariffs, без date-фильтра):")

    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    range_filter = Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="tariffs")),
            FieldCondition(key="created_at", range=DatetimeRange(gte=cutoff)),
        ]
    )
    res_filtered = await client.query_points(
        collection_name=settings.qdrant.collection,
        query=vector,
        query_filter=range_filter,
        limit=3,
        with_payload=True,
    )
    print(f"created_at >= {cutoff} (последние 30 дней), + category=\"tariffs\":")
    _print_results(res_filtered.points, "top-3 (category=tariffs, с date-фильтром):")

    # 3) Композитный must + must_not — только тарифы, исключая архивные
    composite_filter = Filter(
        must=[FieldCondition(key="category", match=MatchValue(value="tariffs"))],
        must_not=[FieldCondition(key="status", match=MatchValue(value="archived"))],
    )
    res_composite = await client.query_points(
        collection_name=settings.qdrant.collection,
        query=vector,
        query_filter=composite_filter,
        limit=3,
        with_payload=True,
    )
    print(f"\n3) must + must_not: category=\"tariffs\" AND NOT status=\"archived\", запрос: {query!r}")
    _print_results(res_composite.points, "top-3:")


async def main() -> None:
    client = _client()
    await cosine_vs_dot(client)
    await filter_examples(client)
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
