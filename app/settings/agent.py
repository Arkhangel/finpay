from typing import Literal

from pydantic import BaseModel


class AgentSettings(BaseModel):
    # Блок 6.4: persistent-граф (app/services/agent_persistent.py). "memory" —
    # только для unit-тестов (InMemorySaver, не переживает рестарт процесса).
    # sqlite — дефолт для локальной разработки без Docker. postgres — прод
    # (docker-compose), тот же Postgres-сервис, что для чата (М3Б5) —
    # LangGraph хранит свои чекпоинты в отдельных таблицах (checkpoints/
    # checkpoint_writes/checkpoint_blobs/checkpoint_migrations), новая БД не
    # плодится.
    #
    # Именование отклоняется от буквального AGENT_CHECKPOINTER из ТЗ (без
    # nested-delimiter) в пользу AGENT__CHECKPOINTER — ради консистентности с
    # остальным проектом, где вся конфигурация организована как
    # settings.<область>.<поле> (см. app/settings/__init__.py и остальные
    # settings/*.py). Осознанное отклонение, задокументировано также в
    # docs/agent-persistent-report.md.
    checkpointer: Literal["memory", "sqlite", "postgres"] = "sqlite"

    # storage/ — тот же volume, что docstore_kb.json (Б5.5), переживает
    # рестарт контейнера (см. compose.yaml).
    sqlite_path: str = "storage/agent_checkpoints.sqlite"

    # psycopg (v3) DSN, НЕ путать с settings.chat.database_url
    # (postgresql+asyncpg://... — для SQLAlchemy). AsyncPostgresSaver сам
    # управляет пулом psycopg, отдельный asyncpg-движок ему не нужен.
    # Порт 5433 — хостовый порт compose-сервиса postgres (5432 на хосте
    # занят системным, не-докеровским Postgres, см. compose.yaml).
    postgres_uri: str = "postgresql://postgres:postgres@localhost:5433/finpay"
