from pydantic import BaseModel


class BotSettings(BaseModel):
    token: str = ""
    backend_url: str = "http://localhost:8000"
    admin_ids: list[int] = []
