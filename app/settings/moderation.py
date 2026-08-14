from pathlib import Path

from pydantic import BaseModel


class ModerationSettings(BaseModel):
    enabled: bool = True
    use_openai_api: bool = False
    keywords_path: Path = Path("app/moderation/moderation_keywords.yaml")
    # category -> порог score для OpenAI Moderation API (0..1); ниже дефолта модели
    category_thresholds: dict[str, float] = {}

    # Второй слой (use_openai_api) зовёт .moderations.create(model="omni-moderation-latest")
    # — это проприетарный эндпоинт настоящего OpenAI, которого нет на Groq
    # (settings.openai.host у нас указывает туда). Раньше ModerationService
    # получал тот же клиент, что и продакшен-LLM (app.state.openai, Groq) —
    # включение use_openai_api просто падало бы на каждом вызове (fail-open,
    # см. ModerationService._check) и не давало реальной модерации (global-
    # аудит, находка №3). Отдельный клиент — тот же паттерн, что
    # EVAL__TESTSET_LLM_API_KEY (см. app/settings/eval.py): настоящий ключ
    # platform.openai.com, не Groq.
    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
