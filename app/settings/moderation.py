from pathlib import Path

from pydantic import BaseModel


class ModerationSettings(BaseModel):
    enabled: bool = True
    use_openai_api: bool = False
    keywords_path: Path = Path("app/moderation/moderation_keywords.yaml")
    # category -> порог score для OpenAI Moderation API (0..1); ниже дефолта модели
    category_thresholds: dict[str, float] = {}
