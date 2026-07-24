"""Metadata-обогащение документов для ingestion pipeline (блок 5.5).

SimpleDirectoryReader из коробки кладёт только служебные поля
(file_path/file_name/file_size/file_type/creation_date/last_modified_date).
Здесь добавляются доменные поля (category/version/author) через колбэк
file_metadata и решается, какие поля не должны попадать в текст, который
уходит на эмбеддинг — это делается отдельным шагом после load_data(),
поскольку excluded_embed_metadata_keys — атрибут Document, а не
extra_info-словарь, который читает file_metadata.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from llama_index.core.schema import Document

# "_v1", "_v2.3" и т.п. в конце имени файла (до расширения).
_VERSION_RE = re.compile(r"_v(\d+(?:\.\d+)*)$", re.IGNORECASE)

# Поля, которые не несут смысловой нагрузки для векторного поиска
# (пути, даты, служебные идентификаторы) и только размывают embedding
# шумом, если попадут в текст ноды. В LLM-контексте и для фильтрации
# retrieval они по-прежнему доступны через node.metadata.
NOISY_EMBED_KEYS = [
    "file_path",
    "file_name",
    "file_size",
    "file_type",
    "creation_date",
    "last_modified_date",
    "last_modified",
    "version",
]


def _extract_version(filename: str) -> str | None:
    match = _VERSION_RE.search(Path(filename).stem)
    return match.group(1) if match else None


def _extract_docx_author(path: Path) -> str | None:
    if path.suffix.lower() != ".docx":
        return None
    try:
        from docx import Document as DocxDocument

        author = DocxDocument(str(path)).core_properties.author
    except Exception:
        return None
    return author or None


def build_file_metadata(base_dir: Path) -> Callable[[str], dict]:
    """Фабрика file_metadata-колбэка для SimpleDirectoryReader(file_metadata=...).

    category — имя первой подпапки файла относительно base_dir
    (data/<category>/file.ext -> category). Для файлов, лежащих прямо в
    base_dir без подпапки, category — None.
    """
    resolved_base = base_dir.resolve()

    def _file_metadata(file_path: str) -> dict:
        path = Path(file_path).resolve()
        stat = path.stat()

        try:
            relative_parts = path.relative_to(resolved_base).parts
        except ValueError:
            relative_parts = path.parts
        category = relative_parts[0] if len(relative_parts) > 1 else None

        return {
            "source": path.name,
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "category": category,
            "version": _extract_version(path.name),
            "author": _extract_docx_author(path),
        }

    return _file_metadata


def apply_embedding_exclusions(documents: list[Document]) -> None:
    """Мутирует documents на месте: шумные поля исключаются из текста,
    отдаваемого эмбеддинг-модели, но остаются в metadata."""
    for doc in documents:
        keys_present = [key for key in NOISY_EMBED_KEYS if key in doc.metadata]
        doc.excluded_embed_metadata_keys = list(
            set(doc.excluded_embed_metadata_keys) | set(keys_present)
        )
