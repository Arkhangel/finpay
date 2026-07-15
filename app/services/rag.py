"""RAGService — RAG pipeline на LlamaIndex (блок 5.3).

SimpleDirectoryReader → SentenceSplitter → QdrantVectorStore → VectorStoreIndex
→ QueryEngine. Параллельная bare-metal реализация того же контракта —
app/services/rag_baremetal.py, сравнение — docs/rag.md.

Запуск отдельно: python -m app.services.rag
"""

from __future__ import annotations

import logging

from llama_index.core import Settings as LlamaSettings
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient

from app.settings import Settings as AppSettings
from app.settings import settings as app_settings

logger = logging.getLogger(__name__)

_FALLBACK_ANSWER = "В базе знаний не нашлось ответа на этот вопрос."


class RAGService:
    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or app_settings
        self._client = QdrantClient(
            url=self._settings.qdrant.url, api_key=self._settings.qdrant.api_key or None
        )
        self._aclient = AsyncQdrantClient(
            url=self._settings.qdrant.url, api_key=self._settings.qdrant.api_key or None
        )
        self._index: VectorStoreIndex | None = None

    def build(self) -> None:
        """Индексирует корпус (первый запуск) либо подключается к готовой коллекции."""
        rag = self._settings.rag
        emb = self._settings.embeddings

        # e5-модель различает запрос/документ через текстовые префиксы — та же
        # конвенция, что и в app/services/embeddings.py::embed_query/embed_documents.
        # Без этого ретрив разойдётся с bare-metal-версией и коллекциями Б5.1/Б5.2.
        LlamaSettings.embed_model = HuggingFaceEmbedding(
            model_name=emb.model,
            device=emb.device,
            query_instruction="query: ",
            text_instruction="passage: ",
            normalize=True,
        )
        # OpenAILike, а не OpenAI: настроенный host (Groq и т.п.) — не сам OpenAI,
        # его модели не входят в захардкоженный список llama_index.llms.openai.
        LlamaSettings.llm = OpenAILike(
            model=self._settings.openai.model,
            api_key=self._settings.openai.api_key,
            api_base=self._settings.openai.host or None,
            is_chat_model=True,
            context_window=8192,
        )
        LlamaSettings.node_parser = SentenceSplitter(
            chunk_size=rag.chunk_size, chunk_overlap=rag.chunk_overlap
        )

        vector_store = QdrantVectorStore(
            collection_name=rag.collection, client=self._client, aclient=self._aclient
        )

        existing = {c.name for c in self._client.get_collections().collections}
        if rag.collection in existing:
            self._index = VectorStoreIndex.from_vector_store(vector_store)
            logger.info("rag_index_attached", extra={"collection": rag.collection})
            return

        documents = SimpleDirectoryReader(input_dir=rag.corpus_dir, recursive=True).load_data()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self._index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        logger.info(
            "rag_index_built", extra={"collection": rag.collection, "documents": len(documents)}
        )

    async def answer(self, question: str) -> dict:
        if self._index is None:
            raise RuntimeError("RAGService.build() must be called before answer()")

        rag = self._settings.rag
        query_engine = self._index.as_query_engine(similarity_top_k=rag.similarity_top_k)
        response = await query_engine.aquery(question)

        source_nodes = response.source_nodes or []
        top_score = source_nodes[0].score if source_nodes and source_nodes[0].score is not None else 0.0

        sources = [
            {
                "text": n.text[:300],
                "source": n.metadata.get("file_name"),
                "score": round(n.score, 3) if n.score is not None else None,
            }
            for n in source_nodes
        ]

        # Честный fallback: не доверяем LLM самостоятельно распознать "не знаю" —
        # если top-1 retrieval ниже порога, ответ переопределяется явно.
        answer_text = _FALLBACK_ANSWER if top_score < rag.score_threshold else str(response)

        return {
            "answer": answer_text,
            "top_score": round(top_score, 3),
            "sources": sources,
        }

    async def aclose(self) -> None:
        await self._aclient.close()
        self._client.close()


if __name__ == "__main__":
    import asyncio

    async def _main() -> None:
        service = RAGService()
        service.build()
        try:
            result = await service.answer("Какая стандартная комиссия за транзакцию?")
            print(result)
        finally:
            await service.aclose()

    asyncio.run(_main())
