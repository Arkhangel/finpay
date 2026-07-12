"""Живой smoke-тест VectorStore через testcontainers.qdrant — блок 5.2.

Пропускается, если Docker недоступен (тот же паттерн, что уже используется для
Postgres-testcontainer в tests/chat/conftest.py) — так тест безопасно
пропускается в песочницах без Docker-демона и реально гоняется там, где Docker
есть.
"""
from __future__ import annotations

import pytest

from qdrant_client.models import PointStruct

from app.services.vector_store import VectorStore

DIM = 4


def _docker_available() -> bool:
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@pytest.fixture
def qdrant_container():
    if not _docker_available():
        pytest.skip("Docker not available for Qdrant testcontainer")

    from testcontainers.qdrant import QdrantContainer

    with QdrantContainer(image="qdrant/qdrant:v1.14.0") as container:
        yield container


async def test_ensure_collection_upsert_search_roundtrip(qdrant_container):
    client = qdrant_container.get_async_client()
    store = VectorStore(client, collection="documents_live_test", dim=DIM)

    await store.ensure_collection()

    points = [
        PointStruct(id=1, vector=[1.0, 0.0, 0.0, 0.0], payload={"source": "a"}),
        PointStruct(id=2, vector=[0.0, 1.0, 0.0, 0.0], payload={"source": "b"}),
    ]
    await store.upsert(points, batch_size=1)

    results = await store.search([1.0, 0.0, 0.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].payload["source"] == "a"

    await client.close()


async def test_ensure_collection_rejects_dim_mismatch(qdrant_container):
    client = qdrant_container.get_async_client()

    store_a = VectorStore(client, collection="documents_dim_test", dim=DIM)
    await store_a.ensure_collection()

    store_b = VectorStore(client, collection="documents_dim_test", dim=DIM + 1)
    with pytest.raises(ValueError, match=str(DIM + 1)):
        await store_b.ensure_collection()

    await client.close()
