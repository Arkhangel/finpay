"""Auth/availability checks for /chats/admin/* that don't need a live Postgres."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.settings import settings


@asynccontextmanager
async def _noop_lifespan(app):
    app.state.openai = MagicMock()
    app.state.cache = None
    app.state.pg_engine = None
    app.state.pg_session_factory = None
    app.state.canary = "test_canary"
    yield


@pytest.fixture
def admin_client():
    from app.main import create_app

    with patch("app.main.lifespan", _noop_lifespan):
        app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _admin_token(monkeypatch):
    monkeypatch.setattr(settings, "admin_token", SecretStr("test-admin-token"))


def test_stats_without_token_header_is_rejected(admin_client):
    resp = admin_client.get("/chats/admin/stats")
    assert resp.status_code in (401, 422)  # 422 if FastAPI rejects missing required header


def test_stats_with_wrong_token_returns_401(admin_client):
    resp = admin_client.get("/chats/admin/stats", headers={"X-Admin-Token": "wrong"})
    assert resp.status_code == 401


def test_stats_with_correct_token_but_no_postgres_returns_503(admin_client):
    resp = admin_client.get("/chats/admin/stats", headers={"X-Admin-Token": "test-admin-token"})
    assert resp.status_code == 503


def test_admin_token_unconfigured_rejects_even_empty_header(admin_client, monkeypatch):
    monkeypatch.setattr(settings, "admin_token", SecretStr(""))
    resp = admin_client.get("/chats/admin/stats", headers={"X-Admin-Token": ""})
    assert resp.status_code == 401


def test_broadcast_requires_admin_token(admin_client):
    resp = admin_client.post(
        "/chats/admin/broadcast",
        json={"message": "hi", "interface_filter": "telegram"},
        headers={"X-Admin-Token": "wrong"},
    )
    assert resp.status_code == 401
