from pydantic import BaseModel, SecretStr


class BotSettings(BaseModel):
    token: str = ""
    backend_url: str = "http://localhost:8000"
    admin_ids: list[int] = []

    # Обратный канал backend -> bot (проактивные уведомления через /notify)
    bot_url: str = "http://localhost:9000"
    internal_token: SecretStr = SecretStr("")
    bot_api_port: int = 9000
