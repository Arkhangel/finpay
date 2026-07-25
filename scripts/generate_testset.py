"""RAGAS TestsetGenerator — golden dataset для оценки RAG (блок 5.6).

    python scripts/generate_testset.py --size 30

Корпус — только категориальные папки `data/<category>/...`, которые реально
индексируются в `finpay_kb` (см. scripts/ingest.py); `data/rag-block-03/` —
отдельный исторический корпус Б5.3/Б5.4 (своя коллекция `rag_block_03`),
намеренно исключён, чтобы golden dataset не содержал вопросов по темам вне
продакшен-базы.

Embeddings для knowledge graph — self-hosted (HuggingFaceEmbedding, e5-base),
как и везде в проекте. LLM для генерации — **настоящий OpenAI**
(settings.eval.testset_llm_model, по умолчанию gpt-5.4-mini), а не Groq:
живая проверка показала, что TestsetGenerator делает много LLM-вызовов на
построение knowledge graph (NER/summary/headline-экстракция на каждый
документ) и на free tier Groq (30 RPM/6000 TPM) это ~4-7 часов на 30 пар
(успешные вызовы шли раз в ~2.5 минуты, упираясь в TPM с первого же
ответа на 2048 токенов). Разовая трата на OpenAI — копейки, конкретно для
этого шага. Судья метрик (scripts/run_eval.py) и продакшен остаются на
Groq — см. docs/rag_evaluation.md, раздел про выбор моделей.

Результат — сырой CSV (tests/eval/golden_dataset_raw.csv) с колонками
user_input/reference/reference_contexts. Дальше ОБЯЗАТЕЛЬНА ручная вычитка
(дубли, слишком общие вопросы, доразметка reference) перед тем, как
получившийся файл переименовывается/сохраняется как
tests/eval/golden_dataset.json — см. README в tests/eval/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: E402
from llama_index.llms.openai import OpenAI  # noqa: E402
from llama_index.llms.openai_like import OpenAILike  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402
from ragas.testset import TestsetGenerator  # noqa: E402
from ragas.testset.persona import Persona  # noqa: E402

from app.services.ingestion import build_file_metadata  # noqa: E402
from app.settings import settings as app_settings  # noqa: E402
from scripts.ingest import _READERS, _load_file  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("generate_testset")

# Единственная папка внутри data/, которую сознательно не берём в golden —
# исторический корпус Б5.3/Б5.4, отдельная коллекция, не пересекается с
# finpay_kb (см. docs/data_inventory.md).
_EXCLUDED_DIR_NAME = "rag-block-03"

# OpenAI paid tier — на порядки выше лимиты, чем Groq free tier (см.
# докстринг модуля), можно позволить себе умеренный параллелизм. Всё равно
# держим retry с запасом на случай отдельных 429/5xx.
_TESTSET_RUN_CONFIG_OPENAI = RunConfig(max_workers=4, max_retries=5, max_wait=30, timeout=120)

# Groq free tier (30 RPM / 6000 TPM) — TPM, а не RPM, реальный бутылочный
# горлышко (см. feedback в памяти: успешные вызовы шли раз в ~2.5 минуты даже
# при max_workers=1). max_workers=1 обязателен — параллельные воркеры только
# усугубляют 429 без выигрыша в throughput. Большие max_wait/timeout, чтобы
# скрипт сам пересидел паузу между вызовами, а не падал.
_TESTSET_RUN_CONFIG_GROQ = RunConfig(max_workers=1, max_retries=10, max_wait=180, timeout=180)

# Задаём персон вручную вместо RAGAS-автогенерации (generate_personas_from_kg) —
# найден баг: PersonaGenerationPrompt по умолчанию language="english", из-за
# чего персоны (и по цепочке — сами вопросы) генерировались на английском для
# полностью русскоязычного корпуса (проверено вживую: 15/23 вопросов оказались
# на английском). TestsetGenerator пропускает автогенерацию персон, если
# persona_list уже задан (см. ragas/testset/synthesizers/generate.py) — заодно
# экономит несколько LLM-вызовов на батч.
_RU_PERSONAS = [
    Persona(
        name="Разработчик-интегратор",
        role_description=(
            "Разработчик на стороне мерчанта, интегрирует FinPay API/SDK: платежи, "
            "вебхуки, идемпотентность, коды ошибок."
        ),
    ),
    Persona(
        name="Владелец бизнеса-мерчанта",
        role_description=(
            "Владелец или менеджер небольшого бизнеса, подключает приём платежей FinPay: "
            "тарифы, комиссии, сроки выплат, документы."
        ),
    ),
    Persona(
        name="Специалист поддержки/комплаенс",
        role_description=(
            "Сотрудник поддержки или комплаенс-отдела мерчанта: разбирает инциденты, "
            "возвраты, AML/KYC-требования и договорные условия."
        ),
    ),
]


def _stratified_sample(
    files: list[Path], base_dir: Path, max_files: int | None, file_offset: int = 0,
) -> list[Path]:
    """Round-robin по категориям (data/<category>/...), а не первые N по алфавиту —
    иначе при урезании корпуса легко потерять целые категории вопросов.

    file_offset сдвигает округ-робин на N шагов вперёд — так соседние батчи
    (Б5.6 генерация маленькими порциями на Groq) берут разные файлы вместо
    того, чтобы каждый раз пересчитывать одни и те же первые max_files."""
    by_category: dict[str, list[Path]] = {}
    for path in files:
        parts = path.relative_to(base_dir).parts
        category = parts[0] if len(parts) > 1 else ""
        by_category.setdefault(category, []).append(path)

    buckets = [iter(sorted(paths)) for paths in by_category.values()]
    ordered: list[Path] = []
    while buckets:
        for bucket in list(buckets):
            item = next(bucket, None)
            if item is None:
                buckets.remove(bucket)
                continue
            ordered.append(item)

    window = ordered[file_offset:]
    if max_files is None:
        return window
    return window[:max_files]


def _load_corpus(base_dir: Path, max_files: int | None, file_offset: int = 0):
    files = sorted(
        p
        for p in base_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in _READERS
        and _EXCLUDED_DIR_NAME not in p.relative_to(base_dir).parts
    )
    logger.info("files_found count=%d dir=%s", len(files), base_dir)

    sampled = _stratified_sample(files, base_dir, max_files, file_offset)
    logger.info(
        "files_sampled count=%d (max_files=%s, file_offset=%d)", len(sampled), max_files, file_offset,
    )

    file_metadata = build_file_metadata(base_dir)
    documents = []
    for path in sampled:
        try:
            documents.extend(_load_file(path, file_metadata))
        except Exception:
            logger.warning("skip_unparseable_file path=%s", path, exc_info=True)
    logger.info("documents_loaded count=%d", len(documents))
    return documents


def _build_generator(provider: str) -> TestsetGenerator:
    if provider == "groq":
        # Тот же Groq-клиент, что и продакшен (settings.openai) — бесплатно,
        # но медленно (TPM), см. _TESTSET_RUN_CONFIG_GROQ и докстринг модуля.
        # OpenAILike, а не OpenAI — см. app/services/rag.py: llama_index.llms.openai.OpenAI
        # валидирует model по захардкоженному списку официальных моделей OpenAI и
        # отклоняет "openai/gpt-oss-120b" (Groq-хостинг), даже с правильным api_base.
        oa = app_settings.openai
        if not oa.api_key:
            raise RuntimeError("OPENAI__API_KEY не задан — нужен Groq-ключ для --provider groq.")
        llm = OpenAILike(
            model=oa.model, api_key=oa.api_key, api_base=oa.host or None,
            is_chat_model=True, context_window=8192,
        )
    else:
        # Настоящий OpenAI — см. докстринг модуля (rate limits на Groq).
        ev = app_settings.eval
        if not ev.testset_llm_api_key:
            raise RuntimeError(
                "EVAL__TESTSET_LLM_API_KEY не задан — нужен настоящий OpenAI-ключ "
                "(platform.openai.com) для --provider openai, см. app/settings/eval.py."
            )
        llm = OpenAI(
            model=ev.testset_llm_model,
            api_key=ev.testset_llm_api_key,
            api_base=ev.testset_llm_api_base,
        )
    embed_model = HuggingFaceEmbedding(
        model_name=app_settings.embeddings.model,
        device=app_settings.embeddings.device,
        query_instruction="query: ",
        text_instruction="passage: ",
        normalize=True,
    )
    generator = TestsetGenerator.from_llama_index(llm=llm, embedding_model=embed_model)
    generator.persona_list = _RU_PERSONAS
    return generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация golden dataset через RAGAS TestsetGenerator")
    parser.add_argument("--size", type=int, default=30, help="Число Q/A пар (минимум 30 по заданию)")
    parser.add_argument("--corpus", default="data", help="Корневая директория корпуса")
    parser.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="Сколько файлов брать в knowledge graph (stratified по категориям); "
             "меньше файлов = меньше LLM-вызовов на построение графа. "
             "0 или отрицательное — взять весь корпус (медленно на free-tier Groq).",
    )
    parser.add_argument(
        "--file-offset",
        type=int,
        default=0,
        help="Сдвиг по округ-робин списку файлов — чтобы соседние батчи брали "
             "разные файлы, а не пересчитывали одни и те же первые max_files.",
    )
    parser.add_argument(
        "--provider",
        choices=["groq", "openai"],
        default="groq",
        help="groq — бесплатно, но медленно (TPM), запускать маленькими батчами; "
             "openai — настоящий OpenAI, быстро, но платно и требует EVAL__TESTSET_LLM_API_KEY.",
    )
    parser.add_argument("--out", default="tests/eval/golden_dataset_raw.csv")
    args = parser.parse_args()

    base_dir = Path(args.corpus)
    max_files = args.max_files if args.max_files and args.max_files > 0 else None
    documents = _load_corpus(base_dir, max_files, args.file_offset)
    if not documents:
        logger.error("no_documents_loaded dir=%s", base_dir)
        sys.exit(1)

    generator = _build_generator(args.provider)
    run_config = _TESTSET_RUN_CONFIG_GROQ if args.provider == "groq" else _TESTSET_RUN_CONFIG_OPENAI
    testset = generator.generate_with_llamaindex_docs(
        documents, testset_size=args.size, run_config=run_config,
    )

    df = testset.to_pandas()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        # Батчи (Б5.6, генерация маленькими порциями на Groq) дописываются в
        # один файл, а не перезаписывают его — дедуп по user_input, т.к.
        # соседние батчи не пересекаются по файлам, но RAGAS иногда всё же
        # генерирует похожие вопросы для смежных документов одной категории.
        existing = pd.read_csv(out_path)
        before = len(existing)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(subset=["user_input"])
        logger.info("merged_with_existing existing_rows=%d new_rows_after_dedup=%d", before, len(df) - before)

    df.to_csv(out_path, index=False)
    logger.info("golden_raw_saved rows=%d path=%s", len(df), out_path)
    print(f"Saved {len(df)} raw Q/A pairs total to {out_path}")
    print("СЛЕДУЮЩИЙ ШАГ ОБЯЗАТЕЛЕН: ручная вычитка перед golden_dataset.json (см. tests/eval/README.md)")


if __name__ == "__main__":
    main()
