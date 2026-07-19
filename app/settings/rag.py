from pydantic import BaseModel


class RagSettings(BaseModel):
    corpus_dir: str = "data/rag-block-03"
    collection: str = "rag_block_03"
    collection_baremetal: str = "rag_block_03_baremetal"
    # Используются fixed_size/recursive в app/services/chunking.py — semantic
    # (победившая стратегия, см. ниже) от них не зависит.
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Итог эксперимента блока 5.4 (docs/chunking_experiment.md): из
    # fixed_size/recursive/semantic на golden dataset (28 вопросов) semantic
    # дал лучший MRR@10 (0.917) при равном с остальными Hit Rate@5/Recall@10
    # (оба 1.000 — корпус из 10 документов небольшой, retrieval насыщается
    # уже на top-10). chunking_strategy — какую из трёх функций в
    # app/services/chunking.py использовать при переиндексации.
    chunking_strategy: str = "semantic"
    semantic_buffer_size: int = 1
    semantic_breakpoint_percentile_threshold: int = 95

    # top-K=10 из grid search (docs/chunking_experiment.md, задача 7):
    # 10 и 20 дали идентичные метрики на этом корпусе — оставлен меньший
    # (дешевле по контексту для LLM-синтеза в RAGService.answer).
    similarity_top_k: int = 10

    # BAAI/bge-reranker-v2-m3 (app/services/reranker.py) поднимает MRR@10
    # semantic-стратегии с 0.917 до 1.000, но на CPU в ~28 раз медленнее
    # (~30ms → ~970ms на вопрос, см. docs/chunking_experiment.md) — на
    # текущем маленьком корпусе выигрыш не окупает задержку, поэтому по
    # умолчанию выключен; включать явно, если корпус вырастет и Hit
    # Rate@5/MRR начнут проседать без него.
    reranker_enabled: bool = False
    reranker_candidate_k: int = 20

    # Ниже этого score top-1 результата ответ считается "не найдено" (fallback).
    score_threshold: float = 0.75
