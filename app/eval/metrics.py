"""RAGAS-метрики для оценки RAG-пайплайна (блок 5.6).

Judge — намеренно ДРУГАЯ модель, чем продакшен (settings.openai.model), чтобы
не путать роли "отвечает" и "оценивает" (см. app/settings/eval.py,
docs/rag_evaluation.md). Тот же Groq-провайдер — бесплатно, без
Anthropic/OpenAI ключей.

Embeddings для AnswerRelevancy — self-hosted (HuggingFaceEmbedding, e5-base),
как и везде в проекте, вместо OpenAI-эмбеддингов по умолчанию в RAGAS.
"""

from __future__ import annotations

import asyncio
import logging
import re

from openai import AsyncOpenAI
from pydantic import BaseModel
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics import discrete_metric
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from app.settings import settings as app_settings

logger = logging.getLogger(__name__)

# Instructor (под капотом ragas.llms.llm_factory) по умолчанию делает 0
# повторных попыток при сбое structured-output-валидации (max_retries=1,
# не прокидывается наружу через llm_factory/InstructorModelArgs — см.
# docs/rag_evaluation.md, баг №7). Живой тест показал: сбой Faithfulness на
# Groq/gpt-oss-20b вероятностный (1 из 4 попыток на одном и том же входе), не
# детерминированный — retry на нашей стороне заметно поднимает успех без
# необходимости патчить ragas.
#
# ВАЖНО: два типа сбоя стоят по-разному. TPM-429 (`rate_limit_exceeded`) —
# отказ ДО генерации, токены не тратятся, ретраить его можно щедро. А
# `json_validate_failed` — отказ ПОСЛЕ генерации (модель реально ответила,
# просто невалидным JSON) — токены списываются по-настоящему. На живом
# прогоне ретрай на 3 попытки для ВТОРОГО случая сам по себе разгонял расход
# TPM-бюджета (каждая "бесплатная" попытка на деле платная) и держал лимит
# на нуле дольше, чем нужно. Поэтому лимиты разные: rate-limit ошибки — до
# 3 попыток (используем подсказку "try again in Xs" из текста ошибки для
# паузы), остальные (json_validate_failed и т.п.) — только 1 повтор.
# Живой прогон показал: 28/28 content-сбоев были на faithfulness, ни разу на
# остальных 4 метриках — теперь, когда сбой метрики не убивает всю строку
# (см. _score_or_none), можно себе позволить больше попыток именно для
# content-сбоев (1 из 4 успех на одном и том же входе — 4 попытки дают
# кумулятивно ~70% успеха вместо ~44% у двух).
_METRIC_RETRIES_RATE_LIMIT = 3
_METRIC_RETRIES_OTHER = 4
_METRIC_RETRY_DELAY_S = 5.0
_RETRY_AFTER_RE = re.compile(r"try again in (\d+(?:\.\d+)?)s")


def _retry_delay(exc: Exception) -> float | None:
    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        return float(match.group(1)) + 1.0  # запас, чтобы не влететь в то же окно
    return None


async def _ascore_with_retry(metric, /, **kwargs):
    attempt = 0
    while True:
        try:
            return await metric.ascore(**kwargs)
        except Exception as exc:  # noqa: BLE001 — судья может падать по-разному (json_validate_failed, 429 и т.п.)
            attempt += 1
            rate_limit_delay = _retry_delay(exc)
            max_attempts = _METRIC_RETRIES_RATE_LIMIT if rate_limit_delay is not None else _METRIC_RETRIES_OTHER
            if attempt >= max_attempts:
                raise
            delay = rate_limit_delay if rate_limit_delay is not None else _METRIC_RETRY_DELAY_S
            logger.warning(
                "metric_retry metric=%s attempt=%d/%d delay=%.1fs error=%s",
                getattr(metric, "name", metric), attempt, max_attempts, delay, exc,
            )
            await asyncio.sleep(delay)


_HAS_CITATION_PROMPT = (
    "Содержит ли ответ ссылку на источник: маркер вида '[1]'/'[doc_id]', "
    "имя файла, или фразу 'согласно …'?\n\nОтвет: {response}"
)


class _CitationVerdict(BaseModel):
    has_citation: bool


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
    может указывать и на Groq (settings.eval.judge_provider="groq", через
    base_url), и на настоящий OpenAI (judge_provider="openai", см.
    docs/rag_evaluation.md баг №11 — Groq/gpt-oss-20b ненадёжен именно как
    judge). provider="groq" в llm_factory не подходит ни для того, ни для
    другого случая — Instructor трактует его как настоящий groq-SDK клиент
    (client.messages.*), а не OpenAI-совместимый (проверено вживую:
    AttributeError на 'messages').
    """
    ev = app_settings.eval
    if ev.judge_provider == "openai":
        if not ev.testset_llm_api_key:
            raise RuntimeError(
                "EVAL__TESTSET_LLM_API_KEY не задан — нужен настоящий OpenAI-ключ "
                "для judge_provider='openai'."
            )
        client = AsyncOpenAI(api_key=ev.testset_llm_api_key, base_url=ev.testset_llm_api_base)
        return llm_factory(ev.testset_llm_model, provider="openai", client=client)

    oa = app_settings.openai
    if not oa.api_key:
        raise RuntimeError("OPENAI__API_KEY не задан — нужен Groq-ключ для judge LLM.")
    client = AsyncOpenAI(api_key=oa.api_key, base_url=oa.host or None)
    return llm_factory(ev.judge_model, provider="openai", client=client)


def build_embeddings() -> BaseRagasEmbedding:
    emb = app_settings.embeddings
    return _SymmetricE5Embeddings(model=emb.model, device=emb.device, normalize_embeddings=True)


def make_has_citation(llm: InstructorBaseRagasLLM):
    """LLM-судья через @discrete_metric (ragas.metrics) — критерий 3 чекпоинта 5.

    Не regex на `[N]`: промпт по заданию явно требует ловить и текстовые формы
    ссылки ("согласно …", упоминание имени файла), которые маркер-based проверка
    пропустила бы. Судья — тот же LLM, что и для остальных метрик (та же
    структурированная генерация через llm.agenerate, что и в build_judge)."""

    @discrete_metric(name="has_citation", allowed_values=["yes", "no"])
    async def has_citation(user_input: str, response: str) -> str:
        verdict = await llm.agenerate(
            _HAS_CITATION_PROMPT.format(response=response), _CitationVerdict,
        )
        return "yes" if verdict.has_citation else "no"

    return has_citation


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
        "has_citation": make_has_citation(llm),
    }


# Пауза МЕЖДУ метриками одной строки, не только retry-after-сбоя: judge
# (gpt-oss-20b) видит retrieved_contexts целиком в Faithfulness/ContextPrecision/
# ContextRecall (~1500-2500 токенов/вызов при 5 чанках), и 5-6 вызовов подряд
# (Faithfulness внутри сама делает 2: statement generation + NLI) почти
# всегда сами по себе выжирают весь TPM-бюджет (8000/мин, скользящее окно) за
# одну строку — проверено вживую: почти 100% строк падали по 429 даже при
# concurrency=1 и паузе 6с, потому что суммарная потребность одной строки
# (~6500-8000+ токенов) уже сама на грани бюджета. 6с было мало, чтобы старые
# токены успели "выкатиться" из скользящего окна — подняли паузу до 15с.
_METRIC_PACE_S = 15.0


async def _score_or_none(metric, label: str, /, **kwargs):
    """Каждая метрика считается независимо — если одна упадёт (см.
    _METRIC_FAILURES ниже: на живом прогоне 28/28 ретраев были на
    faithfulness, ни разу на остальных 4), это не должно убивать остальные
    метрики той же строки. Возвращает None + логирует, а не бросает."""
    try:
        return await _ascore_with_retry(metric, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("metric_failed_giving_up metric=%s error=%s", label, exc)
        return None


async def eval_row(row: dict, metrics: dict[str, object]) -> dict:
    """row — {"user_input", "response", "retrieved_contexts", "reference"}
    (см. RAGService.evaluate_inputs() + golden_dataset.json для reference).
    Возвращает метрики + has_citation в плоском виде, готовом для DataFrame;
    метрика, которая не досчиталась после ретраев (см. _score_or_none) —
    None, а не срыв всей строки (см. docs/rag_evaluation.md, баг №9/№10)."""
    user_input = row["user_input"]
    response = row["response"]
    retrieved_contexts = row["retrieved_contexts"]
    reference = row["reference"]

    faithfulness = await _score_or_none(
        metrics["faithfulness"], "faithfulness",
        user_input=user_input, response=response, retrieved_contexts=retrieved_contexts,
    )
    await asyncio.sleep(_METRIC_PACE_S)
    answer_relevancy = await _score_or_none(
        metrics["answer_relevancy"], "answer_relevancy", user_input=user_input, response=response,
    )
    await asyncio.sleep(_METRIC_PACE_S)
    context_precision = await _score_or_none(
        metrics["context_precision"], "context_precision", user_input=user_input, reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    await asyncio.sleep(_METRIC_PACE_S)
    context_recall = await _score_or_none(
        metrics["context_recall"], "context_recall",
        user_input=user_input, retrieved_contexts=retrieved_contexts, reference=reference,
    )
    await asyncio.sleep(_METRIC_PACE_S)
    has_citation = await _score_or_none(
        metrics["has_citation"], "has_citation", user_input=user_input, response=response,
    )

    return {
        "faithfulness": faithfulness.value if faithfulness is not None else None,
        "answer_relevancy": answer_relevancy.value if answer_relevancy is not None else None,
        "context_precision": context_precision.value if context_precision is not None else None,
        "context_recall": context_recall.value if context_recall is not None else None,
        "has_citation": (has_citation.value == "yes") if has_citation is not None else None,
    }
