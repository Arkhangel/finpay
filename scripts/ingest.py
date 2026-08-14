"""Мультиформатная инкрементальная индексация корпуса (блок 5.5).

    python scripts/ingest.py data/

Каждый файл читается специализированным ридером по расширению
(PyMuPDFReader/.pdf, DocxReader/.docx, HTMLTagReader/.html и .htm,
MarkdownReader/.md — остальные расширения просто пропускаются, не трогаются).
Документы обогащаются метаданными (app/services/ingestion.py) и проходят
через IngestionPipeline с SimpleDocumentStore + DocstoreStrategy.UPSERTS:
docstore персистится в settings.rag.docstore_path между запусками, поэтому
повторный прогон без изменений в файлах не дублирует чанки в Qdrant —
пайплайн видит совпадающий hash и пропускает документ.

Файлы, которые не удалось распарсить (битый PDF, HTML без <section> и
т.п.), переименовываются в "<имя>.failed" и не блокируют индексацию
остальных файлов — ошибка логируется, скрипт продолжает работу.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core import Settings as LlamaSettings  # noqa: E402
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline  # noqa: E402
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter  # noqa: E402
from llama_index.core.schema import Document  # noqa: E402
from llama_index.core.storage.docstore import SimpleDocumentStore  # noqa: E402
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: E402
from llama_index.readers.file import DocxReader, HTMLTagReader, MarkdownReader, PyMuPDFReader  # noqa: E402
from llama_index.vector_stores.qdrant import QdrantVectorStore  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from app.services.ingestion import apply_embedding_exclusions, build_file_metadata  # noqa: E402
from app.settings import settings as app_settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest")

_READERS = {
    ".pdf": PyMuPDFReader(),
    ".docx": DocxReader(),
    ".html": HTMLTagReader(),
    ".htm": HTMLTagReader(),
    ".md": MarkdownReader(),
}
# Публичный список расширений — используется POST /documents/upload
# (app/routers/documents.py) для валидации загружаемого файла.
SUPPORTED_EXTENSIONS = tuple(_READERS.keys())


def _load_file(path: Path, file_metadata) -> list[Document]:
    """Читает один файл специализированным ридером и обогащает метаданными.

    Бросает исключение, если ридер упал ИЛИ вернул пустой список (например,
    HTML без ожидаемого тега <section>) — оба случая для вызывающего кода
    означают "файл не проиндексирован", и должны привести к .failed.
    """
    suffix = path.suffix.lower()
    reader = _READERS[suffix]
    extra = file_metadata(str(path))

    if suffix == ".pdf":
        raw_docs = reader.load_data(file_path=path)
    elif suffix == ".md":
        raw_docs = reader.load_data(file=str(path))
    else:
        raw_docs = reader.load_data(file=path)

    if not raw_docs:
        raise ValueError(f"ридер не извлёк ни одного документа из {path}")

    docs = []
    for index, doc in enumerate(raw_docs):
        # PyMuPDFReader кладёт номер страницы в metadata["source"] — снимаем
        # его до того, как extra (со своим "source" = имя файла) перетрёт
        # ключ, и переносим в отдельное поле "page" (для остальных форматов
        # его просто нет, get() вернёт None).
        page = doc.metadata.pop("source", None)
        doc.metadata.update(extra)
        doc.metadata["page"] = int(page) if page and str(page).isdigit() else None
        # SimpleDirectoryReader сам стабилизирует doc_id по пути файла — при
        # ручном чтении по форматам это нужно сделать явно: без этого каждый
        # запуск получал бы новый случайный doc_id, и UPSERTS никогда не
        # находил бы совпадений с предыдущим прогоном. resolve() — а не сырой
        # path — чтобы один и тот же физический файл давал один и тот же
        # doc_id независимо от того, относительным или абсолютным путём его
        # передал вызывающий (POST /documents/upload и CLI-прогон строят
        # путь по-разному) — иначе UPSERTS считает это разными документами и
        # плодит дубликаты точек в Qdrant (найдено global-аудитом).
        doc.doc_id = f"{path.resolve()}::{index}"
        docs.append(doc)
    return docs


def ingest_files(files: list[Path], base_dir: Path) -> dict:
    """Индексирует уже известный список файлов через IngestionPipeline/UPSERTS.

    Публичная точка входа, переиспользуемая и CLI (main(), после обхода всей
    директории), и POST /documents/upload (app/routers/documents.py) — там
    files это один только что сохранённый файл, base_dir нужен только чтобы
    build_file_metadata мог вычислить category из пути.
    """
    file_metadata = build_file_metadata(base_dir)
    documents: list[Document] = []
    failed: list[Path] = []

    for path in sorted(files):
        try:
            documents.extend(_load_file(path, file_metadata))
        except Exception:
            logger.exception("ingest_file_failed path=%s", path)
            failed_path = path.with_name(path.name + ".failed")
            path.rename(failed_path)
            failed.append(failed_path)

    apply_embedding_exclusions(documents)

    docstore_path = Path(app_settings.rag.docstore_path)
    docstore_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = _build_pipeline(docstore_path)
    changed, unchanged = _count_changed(documents, pipeline.docstore)

    nodes = pipeline.run(documents=documents, show_progress=False)
    pipeline.docstore.persist(str(docstore_path))

    logger.info(
        "ingest_done documents=%d nodes_upserted=%d failed_files=%d",
        len(documents), len(nodes), len(failed),
    )
    return {"changed": changed, "unchanged": unchanged, "nodes_upserted": len(nodes), "failed": failed}


def _build_pipeline(docstore_path: Path) -> IngestionPipeline:
    rag = app_settings.rag
    emb = app_settings.embeddings

    # Тот же конструктор, что и в app/services/rag.py/chunking.py — без
    # query_instruction/text_instruction ретрив на этой модели молча
    # разойдётся с остальным проектом.
    embed_model = HuggingFaceEmbedding(
        model_name=emb.model,
        device=emb.device,
        query_instruction="query: ",
        text_instruction="passage: ",
        normalize=True,
    )
    LlamaSettings.embed_model = embed_model

    # Итог эксперимента блока 5.4 (docs/chunking_experiment.md): semantic дал
    # лучший MRR@10 на golden dataset — та же стратегия используется здесь
    # для продакшен-индексации, а не только в scripts/chunking_experiment.py.
    if rag.chunking_strategy == "semantic":
        node_parser = SemanticSplitterNodeParser(
            buffer_size=rag.semantic_buffer_size,
            breakpoint_percentile_threshold=rag.semantic_breakpoint_percentile_threshold,
            embed_model=embed_model,
        )
    else:
        node_parser = SentenceSplitter(chunk_size=rag.chunk_size, chunk_overlap=rag.chunk_overlap)

    docstore = (
        SimpleDocumentStore.from_persist_path(str(docstore_path))
        if docstore_path.exists()
        else SimpleDocumentStore()
    )

    client = QdrantClient(url=app_settings.qdrant.url, api_key=app_settings.qdrant.api_key or None)
    vector_store = QdrantVectorStore(collection_name=rag.kb_collection, client=client)

    return IngestionPipeline(
        transformations=[node_parser, embed_model],
        docstore=docstore,
        docstore_strategy=DocstoreStrategy.UPSERTS,
        vector_store=vector_store,
    )


def _count_changed(documents: list[Document], docstore: SimpleDocumentStore) -> tuple[int, int]:
    changed = unchanged = 0
    for doc in documents:
        if docstore.get_document_hash(doc.doc_id) == doc.hash:
            unchanged += 1
        else:
            changed += 1
    return changed, unchanged


def main() -> None:
    parser = argparse.ArgumentParser(description="Индексация корпуса FinPay RAG (блок 5.5)")
    parser.add_argument("directory", help="Корневая директория с документами, например data/")
    args = parser.parse_args()

    base_dir = Path(args.directory)
    if not base_dir.is_dir():
        logger.error("directory_not_found path=%s", base_dir)
        sys.exit(1)

    files = sorted(p for p in base_dir.rglob("*") if p.is_file() and p.suffix.lower() in _READERS)
    logger.info("files_found count=%d dir=%s", len(files), base_dir)

    result = ingest_files(files, base_dir)

    print(f"{result['changed']} changed, {result['unchanged']} unchanged")
    if result["failed"]:
        print(f"Не удалось проиндексировать ({len(result['failed'])}), переименованы в .failed:")
        for path in result["failed"]:
            print(f"  {path}")


if __name__ == "__main__":
    main()
