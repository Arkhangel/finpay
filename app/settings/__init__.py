import logging
import os
import traceback
from functools import lru_cache

from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict, BaseSettings, PydanticBaseSettingsSource, TomlConfigSettingsSource, \
    SettingsError

from app.settings.chat import ChatSettings
from app.settings.openai import OpenAISettings
from app.settings.redis import RedisSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    environment: str = os.getenv("ENVIRONMENT", "test")
    project_name: str = "FinPay"

    openai: OpenAISettings = OpenAISettings()
    redis: RedisSettings = RedisSettings()
    chat: ChatSettings = ChatSettings()
    cors_origins: list[str] = ["http://localhost:3000"]

    security_enabled: bool = True

    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    reload: bool = False

    model_config = SettingsConfigDict(
        toml_file=f".config/{environment}.toml",
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
        return env_settings, TomlConfigSettingsSource(settings_cls)


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
