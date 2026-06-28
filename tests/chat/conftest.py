"""Fixtures for chat module tests."""

from __future__ import annotations

import pytest
import pytest_asyncio


def _docker_available() -> bool:
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@pytest_asyncio.fixture(params=["json", "postgres"])
async def repo(request, tmp_path):
    """Yields a ChatRepository implementation for each backend."""
    if request.param == "json":
        from app.chat.repositories.json_repo import JsonChatRepository
        yield JsonChatRepository(base_dir=tmp_path)

    else:
        if not _docker_available():
            pytest.skip("Docker not available for Postgres testcontainer")

        from testcontainers.postgres import PostgresContainer
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

        from app.chat.repositories.pg_models import Base
        from app.chat.repositories.pg_repo import PostgresChatRepository

        with PostgresContainer("postgres:16-alpine") as pg:
            url = pg.get_connection_url().replace("psycopg2", "asyncpg")
            engine = create_async_engine(url, echo=False)
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            factory = async_sessionmaker(engine, expire_on_commit=False)
            async with factory() as session:
                yield PostgresChatRepository(session=session)

            await engine.dispose()
