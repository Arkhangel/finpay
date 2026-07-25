"""RAGAS TestsetGenerator — golden dataset для оценки RAG (блок 5.6).

    python scripts/generate_testset.py --size 30

Корпус — только категориальные папки `data/<category>/...`, которые реально
индексируются в `finpay_kb` (см. scripts/ingest.py); `data/rag-block-03/` —
отдельный исторический корпус Б5.3/Б5.4 (своя коллекция `rag_block_03`),
намеренно исключён, чтобы golden dataset не содержал вопросов по темам вне
продакшен-базы.

LLM и embeddings для генерации — НЕ OpenAI/Anthropic (как в референсном
задании), а тот же self-hosted стек, что и в проде: HuggingFaceEmbedding
(e5-base) + OpenAILike поверх Groq, но на judge-модели
(settings.eval.judge_model), а не на продакшен-модели — чтобы не путать роль
"генерирует вопросы для оценки" с ролью "отвечает пользователю". Осознанное
отклонение от задания, зафиксировано в docs/rag_evaluation.md.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: E402
from llama_index.llms.openai_like import OpenAILike  # noqa: E402
from ragas.testset import TestsetGenerator  # noqa: E402

from app.services.ingestion import build_file_metadata  # noqa: E402
from app.settings import settings as app_settings  # noqa: E402
from scripts.ingest import _READERS, _load_file  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("generate_testset")

# Единственная папка внутри data/, которую сознательно не берём в golden —
# исторический корпус Б5.3/Б5.4, отдельная коллекция, не пересекается с
# finpay_kb (см. docs/data_inventory.md).
_EXCLUDED_DIR_NAME = "rag-block-03"


def _load_corpus(base_dir: Path):
    files = sorted(
        p
        for p in base_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in _READERS
        and _EXCLUDED_DIR_NAME not in p.relative_to(base_dir).parts
    )
    logger.info("files_found count=%d dir=%s", len(files), base_dir)

    file_metadata = build_file_metadata(base_dir)
    documents = []
    for path in files:
        try:
            documents.extend(_load_file(path, file_metadata))
        except Exception:
            logger.warning("skip_unparseable_file path=%s", path, exc_info=True)
    logger.info("documents_loaded count=%d", len(documents))
    return documents


def _build_generator() -> TestsetGenerator:
    # Judge-модель, не продакшен-модель — см. докстринг модуля.
    llm = OpenAILike(
        model=app_settings.eval.judge_model,
        api_key=app_settings.openai.api_key,
        api_base=app_settings.openai.host or None,
        is_chat_model=True,
        context_window=8192,
    )
    embed_model = HuggingFaceEmbedding(
        model_name=app_settings.embeddings.model,
        device=app_settings.embeddings.device,
        query_instruction="query: ",
        text_instruction="passage: ",
        normalize=True,
    )
    return TestsetGenerator.from_llama_index(llm=llm, embedding_model=embed_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация golden dataset через RAGAS TestsetGenerator")
    parser.add_argument("--size", type=int, default=30, help="Число Q/A пар (минимум 30 по заданию)")
    parser.add_argument("--corpus", default="data", help="Корневая директория корпуса")
    parser.add_argument("--out", default="tests/eval/golden_dataset_raw.csv")
    args = parser.parse_args()

    base_dir = Path(args.corpus)
    documents = _load_corpus(base_dir)
    if not documents:
        logger.error("no_documents_loaded dir=%s", base_dir)
        sys.exit(1)

    generator = _build_generator()
    testset = generator.generate_with_llamaindex_docs(documents, testset_size=args.size)

    df = testset.to_pandas()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("golden_raw_saved rows=%d path=%s", len(df), out_path)
    print(f"Saved {len(df)} raw Q/A pairs to {out_path}")
    print("СЛЕДУЮЩИЙ ШАГ ОБЯЗАТЕЛЕН: ручная вычитка перед golden_dataset.json (см. tests/eval/README.md)")


if __name__ == "__main__":
    main()
