"""Unit-тесты для app/services/vector_store.py: моки AsyncQdrantClient, без реального Qdrant."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.vector_store import VectorStore


def _collections_response(names: list[str]) -> MagicMock:
    resp = MagicMock()
    resp.collections = [MagicMock(name=n) for n in names]
    for m, n in zip(resp.collections, names):
        m.name = n
    return resp


def _collection_info(dim: int) -> MagicMock:
    info = MagicMock()
    info.config.params.vectors.size = dim
    return info


async def test_ensure_collection_creates_when_missing():
    client = AsyncMock()
    client.get_collections.return_value = _collections_response([])

    store = VectorStore(client, collection="documents", dim=768)
    await store.ensure_collection()

    client.create_collection.assert_called_once()
    call_kwargs = client.create_collection.call_args[1]
    assert call_kwargs["collection_name"] == "documents"
    assert call_kwargs["vectors_config"].size == 768
    assert client.create_payload_index.call_count == 5


async def test_ensure_collection_noop_when_exists_with_matching_dim():
    client = AsyncMock()
    client.get_collections.return_value = _collections_response(["documents"])
    client.get_collection.return_value = _collection_info(768)

    store = VectorStore(client, collection="documents", dim=768)
    await store.ensure_collection()

    client.create_collection.assert_not_called()
    client.create_payload_index.assert_not_called()


async def test_ensure_collection_raises_on_dim_mismatch():
    client = AsyncMock()
    client.get_collections.return_value = _collections_response(["documents"])
    client.get_collection.return_value = _collection_info(1536)

    store = VectorStore(client, collection="documents", dim=768)

    with pytest.raises(ValueError, match="768"):
        await store.ensure_collection()


async def test_upsert_batches_with_wait_only_on_last_batch():
    client = AsyncMock()
    points = list(range(5))

    store = VectorStore(client, collection="documents", dim=768)
    await store.upsert(points, batch_size=2)

    assert client.upsert.call_count == 3
    calls = client.upsert.call_args_list
    assert calls[0][1]["points"] == [0, 1]
    assert calls[0][1]["wait"] is False
    assert calls[1][1]["points"] == [2, 3]
    assert calls[1][1]["wait"] is False
    assert calls[2][1]["points"] == [4]
    assert calls[2][1]["wait"] is True


async def test_upsert_default_batch_size_is_256():
    client = AsyncMock()
    points = list(range(10))

    store = VectorStore(client, collection="documents", dim=768)
    await store.upsert(points)

    client.upsert.assert_called_once()
    assert client.upsert.call_args[1]["wait"] is True


async def test_search_calls_query_points_and_returns_points():
    client = AsyncMock()
    scored_points = [MagicMock(), MagicMock()]
    result = MagicMock()
    result.points = scored_points
    client.query_points.return_value = result

    store = VectorStore(client, collection="documents", dim=768)
    returned = await store.search([0.1, 0.2], top_k=3, query_filter=None)

    assert returned == scored_points
    call_kwargs = client.query_points.call_args[1]
    assert call_kwargs["collection_name"] == "documents"
    assert call_kwargs["query"] == [0.1, 0.2]
    assert call_kwargs["limit"] == 3
    assert call_kwargs["with_payload"] is True
