from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class ChatSettings(BaseModel):
    repository: Literal["json", "postgres"] = "json"
    storage_dir: Path = Path("./var")
    context_strategy: Literal["sliding", "hybrid"] = "sliding"
    context_window: int = 10
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/finpay"

    # Блок 5.5: RAG в чате чинит поиск для follow-up вопросов отдельным
    # вызовом LLM (условный вопрос -> самодостаточный), генерация по-прежнему
    # видит историю целиком независимо от этого флага. Включено по умолчанию —
    # без него follow-up вида "а для них?" не находит нужные чанки в retrieval.
    rag_condense_enabled: bool = True
