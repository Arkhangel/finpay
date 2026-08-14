"""Калибровка RAG__SCORE_THRESHOLD_NO_RERANK на реальном finpay_kb (блок 5.5,
пункт из global-аудита: 0.75 был исторической Б5.4-калибровкой на другом,
10-документном корпусе, не измеренной на текущем finpay_kb).

Не требует LLM/Groq — только retrieve() (self-hosted эмбеддинги + Qdrant),
без генерации. Reranker принудительно отключается (service._reranker = None),
чтобы измерить именно ту ветку, где top_score — сырой cosine similarity.

Два уровня негативных примеров: EASY (максимально далёкие от домена темы —
борщ, футбол, погода) и HARD (тематически соседние с FinPay — Stripe,
ЮKassa, PCI DSS вообще, но не про FinPay). Вывод честный: easy отделяются от
in-scope чисто, hard — пересекаются с in-scope диапазоном (см.
docs/rag.md — "Threshold для отказа"). Порог 0.82 ловит только очевидный
шум, не смысловую релевантность — это ожидаемый предел сырого cosine, не
баг калибровки.

    ENVIRONMENT=local uv run scripts/calibrate_no_rerank_threshold.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag import RAGService  # noqa: E402
from app.settings import settings as app_settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Заведомо не по теме FinPay — тот же класс вопросов, что и "борщ"/"футбол" в
# docs/rag.md, для разнообразия тематик (еда, погода, спорт, общие факты).
# "Лёгкие" негативы — максимально далеки от домена, ожидаемо низкий score.
EASY_OUT_OF_SCOPE_QUESTIONS = [
    "Какой рецепт борща?",
    "Кто выиграл чемпионат мира по футболу в 2018 году?",
    "Какая завтра будет погода в Москве?",
    "Сколько планет в Солнечной системе?",
    "Как приготовить плов?",
    "Какая столица Австралии?",
    "Посоветуй книгу для чтения на отпуск",
    "Как научиться играть на гитаре?",
]

# "Сложные" негативы — тематически СОСЕДНИЕ с FinPay (платежи/эквайринг/
# комплаенс), но не про FinPay и не покрыты его базой знаний. Реальный
# пограничный случай выглядит скорее так, а не как борщ — это и есть та
# проверка, которую просил сделать пользователь после первой калибровки.
HARD_OUT_OF_SCOPE_QUESTIONS = [
    "Как настроить приём платежей через Stripe?",
    "Какая комиссия у ЮKassa за приём платежей?",
    "Как оспорить чарджбэк в PayPal?",
    "Что такое PCI DSS и зачем он вообще нужен интернет-магазину?",
    "Как открыть расчётный счёт в банке для ООО?",
    "Какие требования 115-ФЗ к банкам при обслуживании юрлиц?",
    "Как принять оплату криптовалютой через Coinbase Commerce?",
    "Какой банк лучше выбрать для эквайринга интернет-магазина?",
    "Как рассчитать НДС для интернет-магазина на УСН?",
    "Как получить кредит для малого бизнеса в банке?",
    "Как настроить вебхуки в Telegram Bot API?",
    "Как защитить сайт интернет-магазина от DDoS-атак?",
]


async def main() -> None:
    data = json.loads((ROOT / "tests/eval/golden_dataset.json").read_text(encoding="utf-8"))
    in_scope_questions = [item["user_input"] for item in data]

    service = RAGService()
    service.build()
    service._reranker = None  # принудительно — измеряем ветку "reranker выключен"

    async def _scores(label: str, questions: list[str]) -> list[float]:
        print(f"=== {label} ({len(questions)} вопросов) ===")
        scores = []
        for q in questions:
            result = await service.retrieve(q)
            scores.append(result["top_score"])
            print(f"{result['top_score']:.4f}  {q}")
        return scores

    def _report(label: str, scores: list[float]) -> None:
        print(
            f"{label:14s} min={min(scores):.4f} max={max(scores):.4f} "
            f"median={statistics.median(scores):.4f}"
        )

    try:
        in_scope_scores = await _scores(
            "In-scope (tests/eval/golden_dataset.json)", in_scope_questions
        )
        easy_scores = await _scores("Easy out-of-scope", EASY_OUT_OF_SCOPE_QUESTIONS)
        hard_scores = await _scores("Hard out-of-scope", HARD_OUT_OF_SCOPE_QUESTIONS)
        all_negative_scores = easy_scores + hard_scores

        print("\n=== Статистика ===")
        _report("in-scope:", in_scope_scores)
        _report("easy neg:", easy_scores)
        _report("hard neg:", hard_scores)
        _report("all neg:", all_negative_scores)

        gap_low = max(all_negative_scores)
        gap_high = min(in_scope_scores)
        if gap_low < gap_high:
            candidate = round((gap_low + gap_high) / 2, 3)
            print(
                f"\nЧистое разделение: max(все негативы)={gap_low:.4f} < "
                f"min(in-scope)={gap_high:.4f}. Кандидат-порог (середина зазора): {candidate}"
            )
        else:
            print(
                f"\nПересечение диапазонов: max(все негативы)={gap_low:.4f} >= "
                f"min(in-scope)={gap_high:.4f} — чистого порога нет, придётся выбирать "
                "компромисс (баланс false positive/false negative) по распечатке выше."
            )
    finally:
        await service.aclose()


if __name__ == "__main__":
    asyncio.run(main())
