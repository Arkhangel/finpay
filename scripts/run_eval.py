"""Прогон RAGAS-метрик по golden dataset — оценка качества RAG (блок 5.6).

    python scripts/run_eval.py

Читает tests/eval/golden_dataset.json (см. scripts/generate_testset.py —
как получить сырой датасет и обязательный шаг ручной вычитки), схема —
список {"user_input": str, "reference": str, "reference_contexts": [str, ...]}.

Для каждой строки:
1. RAGService.evaluate_inputs(user_input) — реальный прогон системы:
   retrieval → (score-guard) → генерация с цитатами.
2. app.eval.metrics.eval_row — RAGAS-метрики (judge = settings.eval.judge_model,
   ДРУГАЯ модель, чем продакшен) + has_citation (LLM-судья через
   @discrete_metric, см. app/eval/metrics.py).

Результат — per-row CSV tests/eval/results/{timestamp}_{label}.csv и
агрегированный {timestamp}_{label}_summary.json рядом — audit log с
человекочитаемым label (baseline, chunk_1024, top_k_10 и т.п.), по которому
можно строить временной ряд A/B-прогонов.

Прогон на Groq free tier медленный (пауза между метриками + ретраи, см.
app/eval/metrics.py) и может прерваться (лимит, Ctrl-C, краш) на середине —
каждая строка пишется в results/_wip_{label}.csv СРАЗУ по готовности, не в
конце. Повторный запуск с тем же --label подхватывает уже посчитанные
вопросы из _wip-файла и не пересчитывает их — считать заново можно, только
удалив/переименовав _wip-файл вручную.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from app.eval.metrics import build_metrics, eval_row  # noqa: E402
from app.services.rag import RAGService  # noqa: E402
from app.settings import settings as app_settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_eval")


def _load_golden_dataset(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not data:
        raise RuntimeError(f"Golden dataset пуст: {path}")
    return data


# Полный набор колонок для КАЖДОЙ строки wip-файла — успешные строки не
# заполняют "error", проваленные не заполняют метрики, но обе должны писать
# ОДИНАКОВЫЙ набор колонок. Иначе построчный append (header пишется только
# один раз, для самой первой строки) молча смещает данные под чужими
# заголовками, если вторая строка вдруг короче/длиннее первой.
_ROW_COLUMNS = [
    "user_input", "reference", "response",
    "faithfulness", "answer_relevancy", "context_precision", "context_recall", "has_citation",
    "error",
]


def _load_wip(wip_path: Path) -> list[dict]:
    if not wip_path.exists():
        return []
    return pd.read_csv(wip_path).to_dict("records")


async def _run(dataset: list[dict], concurrency: int, wip_path: Path, done_questions: set[str]) -> None:
    service = RAGService()
    service.build()
    metrics = build_metrics()
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async def _append_wip(row: dict) -> None:
        # Пишем СРАЗУ по готовности строки, не в конце всего прогона — иначе
        # прерывание (TPM/TPD-лимит, Ctrl-C, краш) на 30-й из 36 строк теряет
        # ВСЁ, а не только недосчитанное. write_lock — на случай concurrency>1,
        # чтобы параллельные строки не перемешали строки в CSV. Колонки
        # нормализуем на полный _ROW_COLUMNS (см. комментарий выше).
        full_row = {col: row.get(col) for col in _ROW_COLUMNS}
        async with write_lock:
            header = not wip_path.exists()
            pd.DataFrame([full_row], columns=_ROW_COLUMNS).to_csv(wip_path, mode="a", header=header, index=False)

    async def _one(item: dict) -> None:
        question = item["user_input"]
        base = {"user_input": question, "reference": item["reference"]}
        try:
            # semaphore держим вокруг ВСЕГО (и evaluate_inputs, и eval_row) —
            # раньше он оборачивал только evaluate_inputs, и eval_row (judge,
            # gpt-oss-20b, 8000 токенов/минуту, см. docs/rag_evaluation.md
            # баг №9/№12) уходил в asyncio.gather БЕЗ ограничения: все N
            # строк одновременно ломились к judge, что и держало TPM-бюджет
            # на нуле — не одна строка сама себе лимит, а весь датасет разом.
            # И evaluate_inputs (продакшен-модель), и eval_row (judge) —
            # оба бьют по Groq и оба могут упасть (TPD-лимит, json_validate_
            # failed на вырожденных отказах, и т.п.) — оборачиваем try целиком,
            # чтобы asyncio.gather не ронял весь прогон на одной строке.
            async with sem:
                live = await service.evaluate_inputs(question)
                row = {**live, "reference": item["reference"]}
                scores = await eval_row(row, metrics)
        except Exception as exc:
            logger.warning("eval_row_failed question=%r error=%s", question[:60], exc)
            await _append_wip({**base, "error": str(exc)})
            return
        logger.info("evaluated question=%r scores=%s", question[:60], scores)
        await _append_wip({**base, "response": live["response"], **scores})

    todo = [item for item in dataset if item["user_input"] not in done_questions]
    if len(todo) < len(dataset):
        logger.info("resume_from_wip skipped=%d remaining=%d", len(dataset) - len(todo), len(todo))
    try:
        await asyncio.gather(*(_one(item) for item in todo))
    finally:
        await service.aclose()


def _aggregate(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    ok = df[df["error"].isna()] if "error" in df.columns else df
    errors = len(df) - len(ok)
    # Метрики считаются независимо друг от друга (app/eval/metrics.py::eval_row,
    # см. docs/rag_evaluation.md баг №10) — конкретная метрика может не
    # досчитаться и на "успешной" строке (error is NaN, но faithfulness=None).
    # mean() у pandas сам пропускает None/NaN, но важно явно показать n_scored
    # per metric — иначе непонятно, среднее по 36 строкам или по 12.
    return {
        **{col: round(ok[col].mean(), 4) for col in metric_cols},
        "citation_rate": round(ok["has_citation"].mean(), 4),
        "n": len(df),
        "errors": errors,
        "n_scored": {col: int(ok[col].notna().sum()) for col in (*metric_cols, "has_citation")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Прогон RAGAS-метрик по golden dataset")
    parser.add_argument("--dataset", default=app_settings.eval.golden_dataset_path)
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Параллельных вопросов. judge (gpt-oss-20b) имеет всего 8000 токенов/минуту "
             "(см. docs/rag_evaluation.md) — на 5 метрик/строку этого хватает впритык даже "
             "на concurrency=1, при 2+ почти каждый вызов уходит в retry по TPM.",
    )
    parser.add_argument(
        "--label", default="baseline",
        help="Человекочитаемая метка прогона (baseline, chunk_1024, top_k_10 и т.п.) — "
             "часть имени файла результата, см. докстринг модуля.",
    )
    args = parser.parse_args()

    dataset = _load_golden_dataset(args.dataset)

    results_dir = Path(app_settings.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    wip_path = results_dir / f"_wip_{args.label}.csv"

    already = _load_wip(wip_path)
    done_questions = {row["user_input"] for row in already}

    asyncio.run(_run(dataset, args.concurrency, wip_path, done_questions))

    rows = _load_wip(wip_path)  # финальный набор = то, что было в wip, + всё дописанное за этот прогон
    stem = f"{datetime.now():%Y-%m-%d_%H%M}_{args.label}"
    per_row_path = results_dir / f"{stem}.csv"
    pd.DataFrame(rows).to_csv(per_row_path, index=False)

    summary = _aggregate(rows)
    summary_path = results_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    wip_path.unlink()  # финальный timestamped CSV теперь единственный источник правды
    logger.info("eval_done rows=%d per_row=%s summary=%s", len(rows), per_row_path, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
