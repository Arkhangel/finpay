"""Integration tests for /chats/admin/* against a real Postgres (testcontainers).

Uses httpx.AsyncClient over ASGITransport (not the sync TestClient) so the app's
asyncpg session and the test both run on the same event loop.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr

from app.chat.domain import ChatMessage
from app.chat.repositories.pg_repo import PostgresChatRepository
from app.settings import settings


def _docker_available() -> bool:
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@pytest.fixture
def admin_token(monkeypatch):
    monkeypatch.setattr(settings, "admin_token", SecretStr("test-admin-token"))
    return "test-admin-token"


@pytest.fixture
async def pg_app(admin_token):
    if not _docker_available():
        pytest.skip("Docker not available for Postgres testcontainer")

    from testcontainers.postgres import PostgresContainer
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

    from app.chat.repositories.pg_models import Base

    with PostgresContainer("postgres:16-alpine") as pg:
        url = pg.get_connection_url().replace("psycopg2", "asyncpg")
        engine = create_async_engine(url, echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        factory = async_sessionmaker(engine, expire_on_commit=False)

        @asynccontextmanager
        async def _noop_lifespan(app):
            yield

        # httpx.ASGITransport never sends lifespan.startup/shutdown, so the real
        # lifespan (which would populate app.state) never runs — set state directly.
        with patch("app.main.lifespan", _noop_lifespan):
            from app.main import create_app
            app = create_app()
        app.state.openai = MagicMock()
        app.state.cache = None
        app.state.pg_engine = engine
        app.state.pg_session_factory = factory
        app.state.canary = "test_canary"

        async with factory() as session:
            yield app, PostgresChatRepository(session=session)

        await engine.dispose()


async def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_stats_reports_totals(pg_app, admin_token):
    app, repo = pg_app
    chat = await repo.create_chat("tg-1", "telegram")
    await repo.append_message(chat.id, ChatMessage(chat_id=chat.id, role="user", content="hi"))
    await repo.append_message(
        chat.id,
        ChatMessage(chat_id=chat.id, role="assistant", content="hello", latency_ms=250),
    )
    await repo.record_moderation_incident(chat.id, "input", "keyword", ["fraud"])

    async with await _client(app) as client:
        resp = await client.get("/chats/admin/stats", headers={"X-Admin-Token": admin_token})

    assert resp.status_code == 200
    body = resp.json()
    assert body["total_messages_24h"] == 2
    assert body["active_users_24h"] == 1
    assert body["avg_latency_ms"] == 250.0
    assert 0 < body["moderation_block_rate"] < 1


async def test_users_lists_recent_chats(pg_app, admin_token):
    app, repo = pg_app
    await repo.create_chat("tg-1", "telegram")
    await repo.create_chat("tg-2", "telegram")

    async with await _client(app) as client:
        resp = await client.get("/chats/admin/users", headers={"X-Admin-Token": admin_token})

    assert resp.status_code == 200
    owners = {u["owner_external_id"] for u in resp.json()}
    assert owners == {"tg-1", "tg-2"}


async def test_broadcast_enqueues_and_pending_resolves_targets(pg_app, admin_token):
    app, repo = pg_app
    await repo.create_chat("111", "telegram")
    await repo.create_chat("222", "telegram")
    await repo.create_chat("not-telegram", "cli")

    async with await _client(app) as client:
        create_resp = await client.post(
            "/chats/admin/broadcast",
            json={"message": "hello everyone", "interface_filter": "telegram"},
            headers={"X-Admin-Token": admin_token},
        )
        assert create_resp.status_code == 201
        broadcast_id = create_resp.json()["id"]

        pending_resp = await client.get(
            "/chats/admin/broadcast/pending", headers={"X-Admin-Token": admin_token}
        )
        assert pending_resp.status_code == 200
        items = pending_resp.json()
        assert len(items) == 1
        assert sorted(items[0]["targets"]) == [111, 222]

        ack_resp = await client.post(
            f"/chats/admin/broadcast/{broadcast_id}/ack",
            json={"status": "sent"},
            headers={"X-Admin-Token": admin_token},
        )
        assert ack_resp.status_code == 200

        pending_after_ack = await client.get(
            "/chats/admin/broadcast/pending", headers={"X-Admin-Token": admin_token}
        )
        assert pending_after_ack.json() == []


async def test_feedback_up_ratio_reflected_in_stats(pg_app, admin_token):
    app, repo = pg_app
    chat = await repo.create_chat("tg-1", "telegram")
    msg = await repo.append_message(
        chat.id, ChatMessage(chat_id=chat.id, role="assistant", content="hi")
    )
    await repo.save_feedback(msg.id, "tg-1", "up")

    async with await _client(app) as client:
        resp = await client.get("/chats/admin/stats", headers={"X-Admin-Token": admin_token})

    assert resp.json()["feedback_up_ratio"] == 1.0
