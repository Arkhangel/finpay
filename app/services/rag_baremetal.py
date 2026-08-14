"""RAGBaremetalService — та же RAG-логика, что app/services/rag.py, но без
LlamaIndex (блок 5.3): чтение файлов → наивный чанкинг → эмбеддинги →
upsert в свою коллекцию с плоским payload → query_points → ручная сборка
промпта → openai.chat.completions.create. Сравнение — docs/rag.md.

Запуск отдельно: python -m app.services.rag_baremetal
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from app.services.embeddings import embed_documents, embed_query
from app.services.vector_store import VectorStore
from app.settings import Settings as AppSettings
from app.settings import settings as app_settings

logger = logging.getLogger(__name__)

_FALLBACK_ANSWER = "В базе знаний не нашлось ответа на этот вопрос."
_ID_NAMESPACE = uuid.UUID("6f9c3b1a-0f3a-4a5e-9d1a-8f6b6e6a2b1c")

_SYSTEM_PROMPT = (
    "Ты ассистент поддержки FinPay. Отвечай только на основе приведённого "
    "контекста. Если в контексте нет ответа на вопрос — прямо скажи, что не "
    "нашёл информацию, не выдумывай факты."
)


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        from docx import Document as DocxDocument

        doc = DocxDocument(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"Неподдерживаемый формат файла: {path}")


def _naive_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Наивное окно по символам — без учёта границ предложений/токенов."""
    text = text.strip()
    if not text:
        return []

    step = chunk_size - chunk_overlap
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


class RAGBaremetalService:
    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or app_settings
        self._qdrant = AsyncQdrantClient(
            url=self._settings.qdrant.url, api_key=self._settings.qdrant.api_key or None
        )
        self._store = VectorStore(
            self._qdrant, self._settings.rag.collection_baremetal, self._settings.embeddings.dim
        )
        self._openai = AsyncOpenAI(
            api_key=self._settings.openai.api_key,
            base_url=self._settings.openai.host or None,
        )

    async def build(self) -> None:
        """Индексирует корпус, если коллекция ещё не существует."""
        existing = {c.name for c in (await self._qdrant.get_collections()).collections}
        if self._settings.rag.collection_baremetal in existing:
            logger.info(
                "rag_baremetal_index_attached",
                extra={"collection": self._settings.rag.collection_baremetal},
            )
            return

        await self._store.ensure_collection()

        rag = self._settings.rag
        corpus_dir = Path(rag.corpus_dir)
        chunks: list[str] = []
        file_names: list[str] = []
        for file_path in sorted(corpus_dir.iterdir()):
            if not file_path.is_file():
                continue
            text = _read_file(file_path)
            for chunk in _naive_chunks(text, rag.chunk_size, rag.chunk_overlap):
                chunks.append(chunk)
                file_names.append(file_path.name)

        vectors = embed_documents(chunks)
        points = [
            PointStruct(
                id=str(uuid.uuid5(_ID_NAMESPACE, f"{file_name}:{i}")),
                vector=vector,
                payload={"text": chunk, "file_name": file_name},
            )
            for i, (chunk, file_name, vector) in enumerate(zip(chunks, file_names, vectors))
        ]
        await self._store.upsert(points)
        logger.info(
            "rag_baremetal_index_built",
            extra={"collection": rag.collection_baremetal, "chunks": len(points)},
        )

    async def answer(self, question: str) -> dict:
        rag = self._settings.rag
        query_vector = embed_query(question)
        results = await self._store.search(query_vector, top_k=rag.similarity_top_k)

        top_score = results[0].score if results else 0.0
        sources = [
            {
                "text": r.payload["text"][:300],
                "source": r.payload["file_name"],
                "score": round(r.score, 3),
            }
            for r in results
        ]

        # Здесь top_score всегда сырой cosine similarity — re-ranker'а нет
        # вообще, поэтому нужен score_threshold_no_rerank, а не
        # rerank-калиброванный score_threshold (иначе guard — no-op, см.
        # RagSettings.score_threshold_no_rerank и global-аудит).
        if top_score < rag.score_threshold_no_rerank:
            return {"answer": _FALLBACK_ANSWER, "top_score": round(top_score, 3), "sources": sources}

        context = "\n\n---\n\n".join(f"[{r.payload['file_name']}]\n{r.payload['text']}" for r in results)
        messages = [
            {"role": "system", "content": f"{_SYSTEM_PROMPT}\n\nКонтекст:\n{context}"},
            {"role": "user", "content": question},
        ]
        response = await self._openai.chat.completions.create(
            model=self._settings.openai.model,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )
        answer_text = (response.choices[0].message.content or "").strip()

        return {"answer": answer_text, "top_score": round(top_score, 3), "sources": sources}

    async def aclose(self) -> None:
        await self._qdrant.close()
        await self._openai.close()


if __name__ == "__main__":
    import asyncio

    async def _main() -> None:
        service = RAGBaremetalService()
        await service.build()
        try:
            result = await service.answer("Какая стандартная комиссия за транзакцию?")
            print(result)
        finally:
            await service.aclose()

    asyncio.run(_main())
