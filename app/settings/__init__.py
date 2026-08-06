import logging
import traceback
from functools import lru_cache

from pydantic import SecretStr, ValidationError
from pydantic_settings import SettingsConfigDict, BaseSettings, PydanticBaseSettingsSource, SettingsError

from app.settings.bot import BotSettings
from app.settings.chat import ChatSettings
from app.settings.embeddings import EmbeddingsSettings
from app.settings.eval import EvalSettings
from app.settings.moderation import ModerationSettings
from app.settings.openai import OpenAISettings
from app.settings.qdrant import QdrantSettings
from app.settings.rag import RagSettings
from app.settings.redis import RedisSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    project_name: str = "FinPay"

    openai: OpenAISettings = OpenAISettings()
    embeddings: EmbeddingsSettings = EmbeddingsSettings()
    qdrant: QdrantSettings = QdrantSettings()
    rag: RagSettings = RagSettings()
    redis: RedisSettings = RedisSettings()
    chat: ChatSettings = ChatSettings()
    bot: BotSettings = BotSettings()
    moderation: ModerationSettings = ModerationSettings()
    eval: EvalSettings = EvalSettings()
    cors_origins: list[str] = ["http://localhost:3000"]

    security_enabled: bool = True
    # Общий секрет для /chats/admin/* (сверяется в app/admin/deps.py::require_admin)
    admin_token: SecretStr = SecretStr("")

    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    reload: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        env_nested_delimiter="__",
    )

    # pylint: disable=too-many-positional-arguments
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Реальные переменные окружения (заданы в шелле/docker-compose) всегда
        # выигрывают у .env — им .env служит только запасным дефолтом, не
        # переопределением. Внутри контейнера .env физически отсутствует
        # (.dockerignore), поэтому dotenv_settings там просто не даёт значений.
        return env_settings, dotenv_settings


@lru_cache
def get_settings():
    logger.info("Set settings to LRU cache.")
    return Settings()


try:
    settings = get_settings()
except ValidationError as e:
    logger.warning("Settings parsing error. Traceback: %s", traceback.format_exc())
    exc_msg = "Invalid settings"
    raise SettingsError(exc_msg) from e
