from pydantic import BaseModel


class EvalSettings(BaseModel):
    # Б5.6: судья RAGAS-метрик — намеренно ДРУГАЯ модель, чем продакшен
    # (settings.openai.model), чтобы не путать роли "отвечает" и "оценивает"
    # (см. docs/rag_evaluation.md). openai/gpt-oss-20b через тот же
    # Groq-провайдер, что и продакшен — бесплатно, без Anthropic/OpenAI
    # ключей (осознанное отклонение от рекомендации задания использовать
    # claude-sonnet-4-6/gpt-5.4-mini для судьи).
    # Было qwen/qwen3.6-27b — падает на структурированном выводе, который
    # RAGAS 1.0-style metrics.collections требуют от judge-LLM (Instructor
    # Mode.JSON, см. app/eval/metrics.py): "json_validate_failed" на каждом
    # вызове. gpt-oss-20b — младший брат продакшен-модели (120b), с тем же
    # семейством/поставщиком, но заведомо другими весами — проверено вживую,
    # структурированный вывод отрабатывает стабильно (в т.ч. на русском).
    judge_model: str = "openai/gpt-oss-20b"
    golden_dataset_path: str = "tests/eval/golden_dataset.json"
    results_dir: str = "tests/eval/results"

    # Эксперимент 2026-07-26: Groq/gpt-oss-20b оказался ненадёжен именно в
    # роли judge — json_validate_failed на Faithfulness (см. баг №11
    # docs/rag_evaluation.md) плюс собственный дневной TPD-лимит, отдельно
    # исчерпанный на отладке. "groq" — текущий дефолт; "openai" — вернуться
    # к рекомендации задания (gpt-5.4-mini), переиспользуя testset_llm_*
    # ключ ниже — настоящий Instructor JSON-mode должен быть надёжнее.
    judge_provider: str = "groq"

    # scripts/generate_testset.py — ОТДЕЛЬНЫЙ разовый шаг, не судья и не
    # продакшен. RAGAS TestsetGenerator строит knowledge graph множеством
    # LLM-вызовов на документ (NER/summary/headline-экстракция) + вопросы —
    # на free tier Groq (30 RPM/6000 TPM) это реально ~4-7 часов на 30 пар
    # (проверено вживую: успешные вызовы шли раз в ~2.5 минуты). Разово
    # используем настоящий OpenAI (выше лимиты, копейки за прогон) только
    # для этого шага; эмбеддинги для knowledge graph остаются self-hosted
    # (HuggingFaceEmbedding), тратится только генерация текста.
    testset_llm_model: str = "gpt-5.4-mini"
    testset_llm_api_key: str = ""
    testset_llm_api_base: str = "https://api.openai.com/v1"
