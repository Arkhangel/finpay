from pydantic import BaseModel


class EvalSettings(BaseModel):
    # Б5.6: судья RAGAS-метрик и LLM для генерации golden dataset —
    # намеренно ДРУГАЯ модель, чем продакшен (settings.openai.model), чтобы
    # не путать роли "отвечает" и "оценивает" (см. docs/rag_evaluation.md).
    # qwen/qwen3.6-27b через тот же Groq-провайдер, что и продакшен —
    # бесплатно, без Anthropic/OpenAI ключей (осознанное отклонение от
    # рекомендации задания использовать claude-sonnet-4-6/gpt-5.4-mini).
    judge_model: str = "qwen/qwen3.6-27b"
    golden_dataset_path: str = "tests/eval/golden_dataset.json"
    results_dir: str = "tests/eval/results"
