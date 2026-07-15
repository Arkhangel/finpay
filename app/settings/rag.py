from pydantic import BaseModel


class RagSettings(BaseModel):
    corpus_dir: str = "data/rag-block-03"
    collection: str = "rag_block_03"
    collection_baremetal: str = "rag_block_03_baremetal"
    chunk_size: int = 512
    chunk_overlap: int = 64
    similarity_top_k: int = 3
    # Ниже этого score top-1 результата ответ считается "не найдено" (fallback).
    score_threshold: float = 0.75
