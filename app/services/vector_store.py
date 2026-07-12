"""VectorStore — тонкая асинхронная обёртка над Qdrant (блок 5.2).

Остальной код не знает про qdrant-client напрямую: на блоке 5.3 эта обёртка
подменяется на LlamaIndex QdrantVectorStore без изменения интерфейса
(ensure_collection/upsert/search).
"""

from __future__ import annotations

import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

logger = logging.getLogger(__name__)

# Корпус — сотни, не миллионы точек: дефолтные HNSW-параметры Qdrant дают
# recall/latency с большим запасом, тюнинг на этом масштабе не оправдан.
_HNSW_M = 16
_HNSW_EF_CONSTRUCT = 100

# Поля, по которым в проекте реально фильтруют (см. docs/vector_store.md).
_PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "source": PayloadSchemaType.KEYWORD,
    "created_at": PayloadSchemaType.DATETIME,
    "category": PayloadSchemaType.KEYWORD,
    "access_level": PayloadSchemaType.KEYWORD,
    "status": PayloadSchemaType.KEYWORD,
}


class VectorStore:
    def __init__(self, client: AsyncQdrantClient, collection: str, dim: int) -> None:
        self._client = client
        self._collection = collection
        self._dim = dim

    async def ensure_collection(self) -> None:
        existing = {c.name for c in (await self._client.get_collections()).collections}

        if self._collection not in existing:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dim,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(m=_HNSW_M, ef_construct=_HNSW_EF_CONSTRUCT),
                ),
            )
            for field_name, schema in _PAYLOAD_INDEXES.items():
                await self._client.create_payload_index(self._collection, field_name, schema)
            logger.info(
                "qdrant_collection_created",
                extra={"collection": self._collection, "dim": self._dim},
            )
            return

        info = await self._client.get_collection(self._collection)
        existing_dim = info.config.params.vectors.size
        if existing_dim != self._dim:
            raise ValueError(
                f"Коллекция {self._collection!r} уже существует с размерностью {existing_dim}, "
                f"а сконфигурирована модель эмбеддингов с размерностью {self._dim}. "
                "Смените settings.qdrant.collection или пересоздайте коллекцию."
            )

    async def upsert(self, points: list[PointStruct], batch_size: int = 256) -> None:
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self._client.upsert(
                collection_name=self._collection,
                points=batch,
                wait=(i + batch_size >= len(points)),
            )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        query_filter: Filter | None = None,
    ) -> list[ScoredPoint]:
        result = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return result.points
