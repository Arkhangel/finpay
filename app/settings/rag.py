from pydantic import BaseModel


class RagSettings(BaseModel):
    corpus_dir: str = "data/rag-block-03"
    collection: str = "rag_block_03"
    collection_baremetal: str = "rag_block_03_baremetal"

    # Блок 5.5: полный корпоративный корпус (data/<category>/...) в отдельной
    # коллекции — rag_block_03 остаётся историческим корпусом Б5.3/Б5.4
    # (10 документов, docs/chunking_experiment.md), смешивать их некорректно.
    kb_collection: str = "finpay_kb"
    # SimpleDocumentStore, персистится между запусками scripts/ingest.py —
    # без этого DocstoreStrategy.UPSERTS не смог бы определить "changed"/
    # "unchanged" на повторном запуске (сравнение идёт по хешу документа,
    # хранящемуся в docstore с предыдущего запуска).
    docstore_path: str = "storage/docstore_kb.json"
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
    # (~30ms → ~970ms на вопрос, см. docs/chunking_experiment.md). На корпусе
    # из 10 документов Б5.4 выигрыш не окупал задержку, поэтому был выключен.
    # Блок 5.5 поднимает корпус до 50+ документов из многих категорий —
    # retrieval на таком объёме грубее, и re-ranking включён по умолчанию.
    reranker_enabled: bool = True
    reranker_candidate_k: int = 20
    # top_n после re-ranking для kb-пайплайна (Б5.5) — retrieval отдаёт
    # top_k=10 (similarity_top_k выше), re-ranker сужает до top_n=5.
    rerank_top_n: int = 5

    # Ниже этого score top-1 результата ответ считается "не найдено" (fallback).
    # 0.75 был откалиброван для косинусного similarity retrieval (Б5.4, без
    # re-ranker). После включения re-ranker (см. reranker_enabled выше)
    # top_score — это уже сырой score BAAI/bge-reranker-v2-m3 (CrossEncoder),
    # а не косинус — несопоставимая шкала. Замер на finpay_kb (172 документа):
    # нерелевантные вопросы стабильно дают 0.000, слабые, но верные попадания
    # (нужный документ найден, но формулировка вопроса неточная) — 0.001-0.03,
    # уверенные попадания — 0.4-0.99. 0.75 отсекал бы почти все верные ответы.
    # Второй слой защиты (промпт с инструкцией "не выдумывай факты") всё
    # равно ловит настоящий "не знаю" — поэтому порог намеренно низкий.
    # TODO(Б5.6): откалибровать точнее на golden dataset.
    score_threshold: float = 0.005

    # Используется вместо score_threshold, когда top_score — сырой cosine
    # similarity, а не CrossEncoder-скор: reranker_enabled=False (RAGService)
    # или rag_baremetal.py (там reranker'а нет вообще). score_threshold там
    # почти всегда true независимо от релевантности (0.005 << любого
    # реального cosine score) — guard молча становится no-op (global-аудит).
    #
    # Откалибровано вживую на finpay_kb (`scripts/calibrate_no_rerank_threshold.py`,
    # без LLM — только retrieve()). 36 in-scope вопросов golden dataset:
    # top_score 0.832-0.935. Явно нерелевантные ("борщ"-класс, максимально
    # далёкие темы): 0.711-0.802 — от них чисто отделяется. НО тематически
    # соседние с FinPay вопросы (Stripe/ЮKassa/PCI DSS вообще/чарджбэк в
    # PayPal — про платежи/эквайринг, но не про FinPay) дают 0.794-0.859 —
    # ПЕРЕСЕКАЮТСЯ с in-scope диапазоном. Чистого порога математически не
    # существует (макс. "соседнего" негатива 0.859 выше мин. in-scope 0.832):
    # сырой cosine в принципе не различает "похоже по теме" и "тот самый
    # документ", только явный шум. Поднять порог до 0.85+, чтобы отсечь
    # больше таких негативов, значило бы отказывать ~14% (5 из 36) реальных
    # вопросов — хуже для пользователя, чем пропустить соседний вопрос на
    # LLM с его собственной инструкцией "не выдумывай факты" (второй слой
    # защиты, не числовой порог — тот же принцип, что и у rerank-порога
    # выше). 0.82 — с запасом ниже всех наблюдённых in-scope, ловит только
    # очевидный шум; расширить выборку негативов при появлении жалоб.
    score_threshold_no_rerank: float = 0.82
