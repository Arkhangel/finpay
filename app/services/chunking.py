"""Три стратегии chunking для сравнения качества retrieval (блок 5.4).

Все три функции принимают уже загруженные `Document` (см.
`SimpleDirectoryReader` в `app/services/rag.py::build`) и возвращают список
нод. Сравнение результатов — `docs/chunking_experiment.md`,
оркестрация экспериментов — `scripts/chunking_experiment.py`.
"""

from __future__ import annotations

import re

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)
from llama_index.core.schema import BaseNode, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.settings import Settings as AppSettings
from app.settings import settings as app_settings

# Пунктуация конца предложения в русском тексте (`.`/`!`/`?`), с учётом
# многоточий и переносов строк между предложениями. Дефолтный
# `chunking_tokenizer_fn` у SentenceSplitter — nltk punkt под английский язык
# и рвёт русские сокращения/предложения не там, где нужно.
_RU_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _split_russian_sentences(text: str) -> list[str]:
    return [s for s in _RU_SENTENCE_BOUNDARY.split(text.strip()) if s]


def _embed_model(settings: AppSettings) -> BaseEmbedding:
    # Тот же конструктор, что и в app/services/rag.py::RAGService.build —
    # без query_instruction/text_instruction ретрив на этой модели молча
    # разойдётся с остальным проектом (см. docs/rag.md).
    emb = settings.embeddings
    return HuggingFaceEmbedding(
        model_name=emb.model,
        device=emb.device,
        query_instruction="query: ",
        text_instruction="passage: ",
        normalize=True,
    )


def build_embed_model(settings: AppSettings | None = None) -> BaseEmbedding:
    """Публичный доступ к конфигурации embed_model — переиспользуется
    оркестрацией эксперимента (scripts/chunking_experiment.py), чтобы не
    грузить модель в память повторно для semantic()/индексации/retrieve.
    """
    return _embed_model(settings or app_settings)


def fixed_size(
    documents: list[Document], *, chunk_size: int = 512, chunk_overlap: int = 64
) -> list[BaseNode]:
    """Baseline: режет по токенам без учёта границ предложений."""
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.get_nodes_from_documents(documents)


def recursive(
    documents: list[Document], *, chunk_size: int = 512, chunk_overlap: int = 64
) -> list[BaseNode]:
    """Эквивалент LangChain RecursiveCharacterTextSplitter: границы абзацев,
    затем предложений — не путать с HierarchicalNodeParser (другая ниша,
    3-уровневый сплиттер для AutoMergingRetriever, здесь не используется).
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        chunking_tokenizer_fn=_split_russian_sentences,
    )
    return splitter.get_nodes_from_documents(documents)


def semantic(
    documents: list[Document],
    *,
    embed_model: BaseEmbedding | None = None,
    settings: AppSettings | None = None,
) -> list[BaseNode]:
    """Границы чанков по семантическому разрыву embedding-соседних предложений."""
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model or _embed_model(settings or app_settings),
    )
    return splitter.get_nodes_from_documents(documents)


STRATEGIES = {
    "fixed_size": fixed_size,
    "recursive": recursive,
    "semantic": semantic,
}
