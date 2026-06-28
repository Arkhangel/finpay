from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class ChatSettings(BaseModel):
    repository: Literal["json", "postgres"] = "json"
    storage_dir: Path = Path("./var")
    context_strategy: Literal["sliding", "hybrid"] = "sliding"
    context_window: int = 10
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/finpay"
