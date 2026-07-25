"""RAGService — query pipeline корпоративного RAG-ассистента (блок 5.5).

Индексация — отдельная забота scripts/ingest.py (IngestionPipeline +
DocstoreStrategy.UPSERTS в коллекцию settings.rag.kb_collection); этот
сервис только подключается к готовой коллекции и отвечает на вопросы:
retrieval (top_k) → опциональный re-ranking (top_n) → код-гард по
top_score ДО вызова LLM → генерация с нумерованными цитатами [1], [2].

Параллельная bare-metal реализация контракта Б5.3 (без re-ranking/цитат/
score-guard-до-LLM — исторический baseline) — app/services/rag_baremetal.py,
сравнение — docs/rag.md.

Запуск отдельно: python -m app.services.rag
"""

from __future__ import annotations

import logging

from llama_index.core import Settings as LlamaSettings
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient

from app.services.reranker import Reranker
from app.settings import Settings as AppSettings
from app.settings import settings as app_settings

logger = logging.getLogger(__name__)

# Формулировка зафиксирована в задании (Б5.5, задача 7) и в docs/rag.md —
# используется и как код-гард-ответ, и как инструкция в системном промпте,
# чтобы LLM формулировала отказ так же, если код-гард почему-то не сработал.
REFUSAL_ANSWER = "По базе не нашёл, могу эскалировать."

# Публичный (не только для этого модуля) промпт с инструкцией цитировать —
# переиспользуется в app/chat/service.py, чтобы multi-turn чат (М4) отвечал
# по тем же правилам, что и одношаговый /rag/query.
CITATION_SYSTEM_PROMPT = (
    "Ты — ассистент технической поддержки FinPay. Отвечай только на основе "
    "пронумерованного контекста ниже, не используй знания вне его. "
    "Ссылайся на источники прямо в тексте ответа в формате [1], [2] и т.д., "
    "у каждого факта должна быть хотя бы одна ссылка. Если в контексте нет "
    f'ответа на вопрос — ответь ровно: "{REFUSAL_ANSWER}". '
    "Не выдумывай цифры, сроки и факты, которых нет в контексте."
)


def build_citation_context(sources: list[dict], nodes: list[NodeWithScore]) -> str:
    """Нумерованный контекст `[1] (файл): текст` — формат, на который ссылается
    CITATION_SYSTEM_PROMPT. Общий для RAGService.answer() и ChatService."""
    return "\n\n".join(
        f"[{s['id']}] ({s['file_name']}): {node.node.get_content()}"
        for s, node in zip(sources, nodes, strict=True)
    )


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
        self._llm = None
        self._reranker = Reranker() if self._settings.rag.reranker_enabled else None

    def build(self) -> None:
        """Подключается к коллекции, наполненной scripts/ingest.py.

        Индексация здесь не выполняется — это отдельный процесс
        (scripts/ingest.py), который можно гонять инкрементально и по
        расписанию, не блокируя старт сервиса.
        """
        rag = self._settings.rag
        emb = self._settings.embeddings

        # e5-модель различает запрос/документ через текстовые префиксы — та же
        # конвенция, что и в scripts/ingest.py и app/services/embeddings.py.
        LlamaSettings.embed_model = HuggingFaceEmbedding(
            model_name=emb.model,
            device=emb.device,
            query_instruction="query: ",
            text_instruction="passage: ",
            normalize=True,
        )
        # OpenAILike, а не OpenAI: настроенный host (Groq и т.п.) — не сам
        # OpenAI, его модели не входят в захардкоженный список
        # llama_index.llms.openai. Хранится и как self._llm (для answer()),
        # и в LlamaSettings (внутренние механизмы LlamaIndex вроде
        # VectorStoreIndex по-прежнему читают LLM оттуда).
        self._llm = OpenAILike(
            model=self._settings.openai.model,
            api_key=self._settings.openai.api_key,
            api_base=self._settings.openai.host or None,
            is_chat_model=True,
            context_window=8192,
        )
        LlamaSettings.llm = self._llm

        vector_store = QdrantVectorStore(
            collection_name=rag.kb_collection, client=self._client, aclient=self._aclient
        )

        existing = {c.name for c in self._client.get_collections().collections}
        if rag.kb_collection not in existing:
            raise RuntimeError(
                f"Коллекция {rag.kb_collection!r} не найдена — сначала прогоните "
                "`python scripts/ingest.py data/`."
            )

        self._index = VectorStoreIndex.from_vector_store(vector_store)
        logger.info("rag_index_attached", extra={"collection": rag.kb_collection})

    async def retrieve(self, question: str) -> dict:
        """Retrieval → опциональный re-ranking → score-guard-флаг, без генерации.

        Переиспользуется и в answer() (контракт /rag/query, Б5.3), и в
        app/chat/service.py::ChatService (multi-turn, Б5.5) — там генерация
        идёт через свой LLM-клиент и стриминг, а не через self._llm.
        """
        if self._index is None:
            raise RuntimeError("RAGService.build() must be called before retrieve()")

        rag = self._settings.rag

        retriever = self._index.as_retriever(similarity_top_k=rag.similarity_top_k)
        nodes: list[NodeWithScore] = await retriever.aretrieve(question)

        if self._reranker is not None and nodes:
            nodes = self._reranker.rerank(question, nodes, top_n=rag.rerank_top_n)

        top_score = nodes[0].score if nodes and nodes[0].score is not None else 0.0
        confident = top_score >= rag.score_threshold

        sources = [
            {
                "id": i + 1,
                "file_name": node.node.metadata.get("source"),
                "page": node.node.metadata.get("page"),
                "score": round(node.score, 3) if node.score is not None else None,
                "snippet": node.node.get_content()[:300],
            }
            for i, node in enumerate(nodes)
        ]

        if not confident:
            logger.info(
                "rag_score_guard_triggered",
                extra={"top_score": round(top_score, 3), "threshold": rag.score_threshold},
            )

        return {
            "nodes": nodes,
            "sources": sources,
            "top_score": round(top_score, 3),
            "confident": confident,
        }

    async def _generate(self, question: str, retrieval: dict) -> str:
        # Честный fallback ДО вызова LLM: если top-1 retrieval ниже порога,
        # генерация вообще не запускается — не тратим вызов модели на
        # заведомо нерелевантный контекст и не рискуем, что она всё равно
        # что-то придумает поверх мусорных чанков.
        if not retrieval["confident"]:
            return REFUSAL_ANSWER

        context = build_citation_context(retrieval["sources"], retrieval["nodes"])
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=CITATION_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=f"Контекст:\n{context}\n\nВопрос: {question}"),
        ]
        response = await self._llm.achat(messages)
        return (response.message.content or "").strip()

    async def answer(self, question: str) -> dict:
        retrieval = await self.retrieve(question)
        answer_text = await self._generate(question, retrieval)
        return {
            "answer": answer_text,
            "top_score": retrieval["top_score"],
            "confident": retrieval["confident"],
            "sources": retrieval["sources"],
        }

    async def evaluate_inputs(self, question: str) -> dict:
        """question -> {user_input, response, retrieved_contexts} — плоский
        формат для RAGAS-метрик (блок 5.6, см. app/eval/metrics.py).

        retrieved_contexts — полный текст нод (node.get_content()), а не
        обрезанный до 300 символов snippet из retrieve()["sources"] (тот
        усечён для API-ответа /rag/query; Faithfulness и т.п. метрикам нужен
        полный контекст, иначе обоснованный по факту ответ будет считаться
        неподтверждённым)."""
        retrieval = await self.retrieve(question)
        response = await self._generate(question, retrieval)
        retrieved_contexts = [node.node.get_content() for node in retrieval["nodes"]]
        return {
            "user_input": question,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
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
