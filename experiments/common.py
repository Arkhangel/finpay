"""Общее для experiments/multi_agent_langgraph.py и single_agent_baseline.py
(блок 6.5): один и тот же tool, одни и те же 5 вопросов, одна и та же логика
подсчёта токенов/вызовов модели — иначе сравнение нечестное (см. ТЗ).

search_knowledge_base импортирован напрямую из app.services.agent_graph
(Б6.3) — та же реализация-обёртка над RAGService из М5, не копия.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.tools import tool  # noqa: E402

from app.services.rag import RAGService  # noqa: E402
from app.settings import settings as app_settings  # noqa: E402

MODEL_NAME = app_settings.openai.model  # Groq openai/gpt-oss-120b — тот же, что везде в проекте

_rag: RAGService | None = None


def _get_rag() -> RAGService:
    global _rag
    if _rag is None:
        _rag = RAGService()
        _rag.build()
    return _rag


# Собственная обёртка над RAGService (не agent_graph.py::search_knowledge_base
# из Б6.3) — тот возвращает только текст top-1 сниппета, без id/file_name,
# то есть буквально нечего честно подставить в [1]/[2]. Здесь — та же
# RAGService (реальный RAG из М5), но с пронумерованными источниками в
# возврате, чтобы цитирование в финальном ответе опиралось на реальные
# file_name, а не на выдуманные номера. Импортируется ОДНОЙ функцией в оба
# скрипта эксперимента — иначе сравнение нечестное (разные tools).
@tool
async def search_knowledge_base(query: str) -> str:
    """Ищет в базе знаний FinPay (тарифы, правила, возвраты) фрагменты,
    релевантные запросу, и возвращает их пронумерованными — используй номер
    каждого фрагмента как ссылку [N] при цитировании в финальном ответе.
    Вызывай перед ответом на любой фактический вопрос о FinPay. Возвращает
    строку 'В базе знаний ничего не найдено', если релевантных документов
    нет — в этом случае не выдумывай факты.
    """
    result = await _get_rag().retrieve(query)
    sources = result["sources"]
    if not sources:
        return "В базе знаний ничего не найдено."
    return "\n".join(f"[{s['id']}] ({s['file_name']}): {s['snippet']}" for s in sources[:3])

QUESTIONS = [
    {
        "id": "01_simple_fact",
        "category": "corpus",
        "question": "Какая стандартная комиссия за транзакцию в FinPay?",
    },
    {
        "id": "02_simple_fact",
        "category": "corpus",
        "question": "В течение какого срока можно оформить возврат средств после оплаты?",
    },
    {
        "id": "03_simple_fact",
        "category": "corpus",
        "question": "На сколько увеличивается комиссия для карт иностранной эмиссии по сравнению со стандартной ставкой?",
    },
    {
        "id": "04_multistep",
        "category": "multistep",
        "question": (
            "Сколько составит итоговая комиссия за транзакцию картой иностранной "
            "эмиссии, и в какие сроки клиенту вернутся деньги, если он оформит "
            "возврат? Учти оба пункта в одном ответе."
        ),
    },
    {
        "id": "05_out_of_corpus",
        "category": "out_of_corpus",
        "question": "Как приготовить борщ?",
    },
]


def ai_messages(messages: list) -> list:
    return [m for m in messages if type(m).__name__ == "AIMessage"]


def usage_from_messages(messages: list) -> dict:
    """Суммирует usage_metadata по всем AIMessage — total_tokens/llm_calls."""
    msgs = ai_messages(messages)
    with_usage = [m for m in msgs if getattr(m, "usage_metadata", None)]
    total = sum(m.usage_metadata["total_tokens"] for m in with_usage)
    return {"total_tokens": total, "llm_calls": len(with_usage)}
