"""RAGAS-метрики для оценки RAG-пайплайна (блок 5.6).

Judge — намеренно ДРУГАЯ модель, чем продакшен (settings.openai.model), чтобы
не путать роли "отвечает" и "оценивает" (см. app/settings/eval.py,
docs/rag_evaluation.md). Тот же Groq-провайдер — бесплатно, без
Anthropic/OpenAI ключей.

Embeddings для AnswerRelevancy — self-hosted (HuggingFaceEmbedding, e5-base),
как и везде в проекте, вместо OpenAI-эмбеддингов по умолчанию в RAGAS.
"""

from __future__ import annotations

import re

from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from app.settings import settings as app_settings

_CITATION_RE = re.compile(r"\[\d+\]")


class _SymmetricE5Embeddings(HuggingFaceEmbeddings):
    """intfloat/multilingual-e5-* модели различают запрос/документ через
    текстовые префиксы, но это асимметричная (retrieval) конвенция. Для
    симметричных задач — сравнение вопрос-с-вопросом, как в AnswerRelevancy —
    e5 card рекомендует префикс "query: " на ОБОИХ текстах, не query/passage.
    Базовый ragas.embeddings.HuggingFaceEmbeddings эмбеддит текст как есть,
    без каких-либо префиксов."""

    def embed_text(self, text: str, **kwargs):
        return super().embed_text(f"query: {text}", **kwargs)

    async def aembed_text(self, text: str, **kwargs):
        return await super().aembed_text(f"query: {text}", **kwargs)

    def embed_texts(self, texts: list[str], **kwargs):
        return super().embed_texts([f"query: {t}" for t in texts], **kwargs)

    async def aembed_texts(self, texts: list[str], **kwargs):
        return await super().aembed_texts([f"query: {t}" for t in texts], **kwargs)


def build_judge() -> InstructorBaseRagasLLM:
    """Judge LLM для RAGAS-метрик.

    provider="openai" в llm_factory — это выбор Instructor-адаптера по ФОРМЕ
    клиента (OpenAI-совместимый), а не по реальному провайдеру; сам клиент
    указывает на Groq через base_url. provider="groq" здесь не подходит —
    Instructor трактует его как настоящий groq-SDK клиент (client.messages.*),
    а не OpenAI-совместимый (проверено вживую: AttributeError на 'messages').
    """
    oa = app_settings.openai
    if not oa.api_key:
        raise RuntimeError("OPENAI__API_KEY не задан — нужен Groq-ключ для judge LLM.")
    client = AsyncOpenAI(api_key=oa.api_key, base_url=oa.host or None)
    return llm_factory(app_settings.eval.judge_model, provider="openai", client=client)


def build_embeddings() -> BaseRagasEmbedding:
    emb = app_settings.embeddings
    return _SymmetricE5Embeddings(model=emb.model, device=emb.device, normalize_embeddings=True)


def build_metrics(
    llm: InstructorBaseRagasLLM | None = None, embeddings: BaseRagasEmbedding | None = None,
) -> dict[str, object]:
    llm = llm or build_judge()
    embeddings = embeddings or build_embeddings()
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


def make_has_citation(answer: str) -> bool:
    """Не-LLM метрика: есть ли хоть одна ссылка [N] в ответе — контракт
    CITATION_SYSTEM_PROMPT (app/services/rag.py) требует ссылку на каждый
    факт. Дёшево ловит грубые нарушения формата без judge-вызова; не
    заменяет Faithfulness (та проверяет содержательную обоснованность)."""
    return bool(_CITATION_RE.search(answer))


async def eval_row(row: dict, metrics: dict[str, object]) -> dict:
    """row — {"user_input", "response", "retrieved_contexts", "reference"}
    (см. RAGService.evaluate_inputs() + golden_dataset.json для reference).
    Возвращает метрики + has_citation в плоском виде, готовом для DataFrame."""
    user_input = row["user_input"]
    response = row["response"]
    retrieved_contexts = row["retrieved_contexts"]
    reference = row["reference"]

    faithfulness = await metrics["faithfulness"].ascore(
        user_input=user_input, response=response, retrieved_contexts=retrieved_contexts,
    )
    answer_relevancy = await metrics["answer_relevancy"].ascore(user_input=user_input, response=response)
    context_precision = await metrics["context_precision"].ascore(
        user_input=user_input, reference=reference, retrieved_contexts=retrieved_contexts,
    )
    context_recall = await metrics["context_recall"].ascore(
        user_input=user_input, retrieved_contexts=retrieved_contexts, reference=reference,
    )

    return {
        "faithfulness": faithfulness.value,
        "answer_relevancy": answer_relevancy.value,
        "context_precision": context_precision.value,
        "context_recall": context_recall.value,
        "has_citation": make_has_citation(response),
    }
