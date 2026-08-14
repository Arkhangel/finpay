"""Загрузка базы знаний FinPay в Qdrant — блок 5.2.

Идемпотентно: id детерминированы (uuid5 от `source`), повторный запуск делает
upsert по тем же id вместо дублирования точек. Источники — реальные материалы
проекта, разрезанные на чанки (не файлы целиком), плюс одна намеренно
сконструированная демо-пара active/archived для фильтр-примеров в
docs/vector_store.md (у реальной базы фактов сейчас нет истории версий, не на
чем иначе показать "без фильтра — старое, с фильтром — только свежее").

    ENVIRONMENT=local uv run scripts/load_to_qdrant.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct
from tqdm import tqdm

from app.services.embeddings import embed_documents
from app.services.vector_store import VectorStore
from app.settings import settings

ROOT = Path(__file__).resolve().parents[1]

# Фиксированный namespace проекта для uuid5(NAMESPACE, source) — детерминированные id.
NAMESPACE = uuid.UUID("f6a6b1d0-5b2e-4e8a-9f1a-1f2c3d4e5f60")

_HEADING_RE = re.compile(r"^(#{2,3})\s+(.*)$")

SERVICE_FACTS_CATEGORIES = {
    "Подключение и аккаунт": "onboarding",
    "Способы оплаты": "payment_methods",
    "Тарифы и комиссии": "tariffs",
    "Лимиты": "limits",
    "Статусы транзакций": "transaction_status",
    "Возвраты (рефанды)": "refunds",
    "Вебхуки": "webhooks",
    "Коды ошибок API": "errors",
    "Документация и поддержка": "support_contacts",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _days_ago_iso(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def heading_chunks(text: str) -> list[tuple[str, str]]:
    """Режет markdown по ##/### заголовкам, пропуская ```-кодовые блоки — иначе
    строки-комментарии вида `# → ...` из примеров curl/bash ловились бы как
    заголовки."""
    chunks: list[tuple[str, str]] = []
    heading: str | None = None
    body: list[str] = []
    in_fence = False

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            if heading is not None:
                body.append(line)
            continue
        if not in_fence:
            m = _HEADING_RE.match(line)
            if m:
                if heading is not None:
                    chunks.append((heading, "\n".join(body).strip()))
                heading = m.group(2).strip()
                body = []
                continue
        if heading is not None:
            body.append(line)

    if heading is not None:
        chunks.append((heading, "\n".join(body).strip()))

    return chunks


def load_service_facts() -> list[dict]:
    text = (ROOT / "app/prompts/service_facts.j2").read_text(encoding="utf-8")
    docs = []
    for heading, body in heading_chunks(text):
        category = SERVICE_FACTS_CATEGORIES.get(heading, "support_contacts")
        bullets = [line[2:].strip() for line in body.splitlines() if line.startswith("- ")]
        for i, bullet in enumerate(bullets):
            docs.append(
                {
                    "source": f"service_facts.j2#{heading}#{i}",
                    "text": bullet,
                    "category": category,
                    "access_level": "public",
                    "status": "active",
                    "created_at": _now_iso(),
                }
            )
    return docs


def load_golden_dataset() -> list[dict]:
    # Б5.6 переложил golden dataset в tests/eval/golden_dataset.json с другой
    # схемой (плоский список user_input/reference/reference_contexts/source,
    # без items/question/expected_answer/category) — старый eval/golden_dataset.json
    # удалён вместе с ранним, вытесненным G-Eval harness'ом.
    data = json.loads((ROOT / "tests/eval/golden_dataset.json").read_text(encoding="utf-8"))
    docs = []
    for i, item in enumerate(data):
        docs.append(
            {
                "source": f"golden_dataset#{i}",
                "text": f"{item['user_input']}\n{item['reference']}",
                "category": item.get("source", "golden_dataset"),
                "access_level": "public",
                "status": "active",
                "created_at": _now_iso(),
            }
        )
    return docs


def load_tool_docs() -> list[dict]:
    docs = []
    for path in sorted((ROOT / "app/prompts/tools").glob("*.md")):
        docs.append(
            {
                "source": f"tools/{path.name}",
                "text": path.read_text(encoding="utf-8").strip(),
                "category": "tools",
                "access_level": "public",
                "status": "active",
                "created_at": _now_iso(),
            }
        )
    return docs


def load_markdown_sections(rel_path: str, category: str) -> list[dict]:
    """docs/chat.md, docs/architecture.md, README.md — внутренняя инженерная
    документация, не для показа клиенту (access_level=internal)."""
    text = (ROOT / rel_path).read_text(encoding="utf-8")
    docs = []
    for heading, body in heading_chunks(text):
        if not body.strip():
            continue
        docs.append(
            {
                "source": f"{rel_path}#{heading}",
                "text": f"{heading}\n{body}",
                "category": category,
                "access_level": "internal",
                "status": "active",
                "created_at": _now_iso(),
            }
        )
    return docs


def demo_archived_tariff() -> dict:
    """Намеренно сконструированный документ для демо range/must_not фильтров
    в docs/vector_store.md — см. docstring модуля."""
    return {
        "source": "service_facts.j2#archived_tariff_2025",
        "text": (
            "Устаревший тариф (архив, действовал до 2025 года): комиссия 2.5% "
            "от суммы транзакции вне зависимости от оборота мерчанта."
        ),
        "category": "tariffs",
        "access_level": "public",
        "status": "archived",
        "created_at": _days_ago_iso(90),
    }


def build_documents() -> list[dict]:
    docs: list[dict] = []
    docs += load_service_facts()
    docs += load_golden_dataset()
    docs += load_tool_docs()
    docs += load_markdown_sections("docs/chat.md", "chat_module")
    docs += load_markdown_sections("docs/architecture.md", "architecture")
    docs += load_markdown_sections("README.md", "project_docs")
    docs.append(demo_archived_tariff())
    return docs


async def main() -> None:
    documents = build_documents()
    print(f"Документов к загрузке: {len(documents)}")

    texts = [doc["text"] for doc in documents]
    vectors: list[list[float]] = []
    embed_batch = settings.embeddings.batch_size
    for i in tqdm(range(0, len(texts), embed_batch), desc="embedding"):
        vectors.extend(embed_documents(texts[i : i + embed_batch]))

    points = [
        PointStruct(
            id=str(uuid.uuid5(NAMESPACE, doc["source"])),
            vector=vector,
            payload=doc,
        )
        for doc, vector in zip(documents, vectors)
    ]

    client = AsyncQdrantClient(url=settings.qdrant.url, api_key=settings.qdrant.api_key or None)
    store = VectorStore(client, settings.qdrant.collection, settings.embeddings.dim)

    await store.ensure_collection()
    await store.upsert(points, batch_size=256)

    info = await client.get_collection(settings.qdrant.collection)
    print(f"points_count = {info.points_count}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
