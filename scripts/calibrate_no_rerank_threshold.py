"""Калибровка RAG__SCORE_THRESHOLD_NO_RERANK на реальном finpay_kb (блок 5.5,
пункт из global-аудита: 0.75 был исторической Б5.4-калибровкой на другом,
10-документном корпусе, не измеренной на текущем finpay_kb).

Не требует LLM/Groq — только retrieve() (self-hosted эмбеддинги + Qdrant),
без генерации. Reranker принудительно отключается (service._reranker = None),
чтобы измерить именно ту ветку, где top_score — сырой cosine similarity.

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
OUT_OF_SCOPE_QUESTIONS = [
    "Какой рецепт борща?",
    "Кто выиграл чемпионат мира по футболу в 2018 году?",
    "Какая завтра будет погода в Москве?",
    "Сколько планет в Солнечной системе?",
    "Как приготовить плов?",
    "Какая столица Австралии?",
    "Посоветуй книгу для чтения на отпуск",
    "Как научиться играть на гитаре?",
]


async def main() -> None:
    data = json.loads((ROOT / "tests/eval/golden_dataset.json").read_text(encoding="utf-8"))
    in_scope_questions = [item["user_input"] for item in data]

    service = RAGService()
    service.build()
    service._reranker = None  # принудительно — измеряем ветку "reranker выключен"

    try:
        in_scope_scores: list[float] = []
        print(f"=== In-scope ({len(in_scope_questions)} вопросов, tests/eval/golden_dataset.json) ===")
        for q in in_scope_questions:
            result = await service.retrieve(q)
            in_scope_scores.append(result["top_score"])
            print(f"{result['top_score']:.4f}  {q}")

        out_of_scope_scores: list[float] = []
        print(f"\n=== Out-of-scope ({len(OUT_OF_SCOPE_QUESTIONS)} вопросов) ===")
        for q in OUT_OF_SCOPE_QUESTIONS:
            result = await service.retrieve(q)
            out_of_scope_scores.append(result["top_score"])
            print(f"{result['top_score']:.4f}  {q}")

        print("\n=== Статистика ===")
        print(
            f"in-scope:     min={min(in_scope_scores):.4f} max={max(in_scope_scores):.4f} "
            f"median={statistics.median(in_scope_scores):.4f}"
        )
        print(
            f"out-of-scope: min={min(out_of_scope_scores):.4f} max={max(out_of_scope_scores):.4f} "
            f"median={statistics.median(out_of_scope_scores):.4f}"
        )

        gap_low = max(out_of_scope_scores)
        gap_high = min(in_scope_scores)
        if gap_low < gap_high:
            candidate = round((gap_low + gap_high) / 2, 3)
            print(
                f"\nЧистое разделение: max(out-of-scope)={gap_low:.4f} < "
                f"min(in-scope)={gap_high:.4f}. Кандидат-порог (середина зазора): {candidate}"
            )
        else:
            print(
                f"\nПересечение диапазонов: max(out-of-scope)={gap_low:.4f} >= "
                f"min(in-scope)={gap_high:.4f} — чистого порога нет, см. распечатку выше "
                "для ручного анализа."
            )
    finally:
        await service.aclose()


if __name__ == "__main__":
    asyncio.run(main())
