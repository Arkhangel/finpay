"""Integration tests for POST /agent/stream (блок 6.4): аутентификация,
graceful degradation при недоступном графе, входная модерация (global-аудит,
находки №1/№4/№7)."""
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
def admin_token(monkeypatch):
    monkeypatch.setattr(settings, "admin_token", SecretStr("test-admin-token"))
    return "test-admin-token"


def _client(graph):
    from app.main import create_app
    from app.deps.providers import get_agent_graph

    with patch("app.main.lifespan", _noop_lifespan):
        app = create_app()

    app.dependency_overrides[get_agent_graph] = lambda: graph
    return TestClient(app)


_REQUEST_BODY = {
    "thread_id": "t1",
    "input": {"messages": [{"role": "user", "content": "привет"}]},
}


def test_agent_stream_requires_admin_token(admin_token):
    # X-Admin-Token обязателен как заголовок (Header()) — без него FastAPI
    # отвечает 422 ещё до вызова require_admin_token; с неверным значением —
    # 401 (см. test_agent_stream_rejects_wrong_admin_token ниже).
    with _client(MagicMock()) as client:
        resp = client.post("/agent/stream", json=_REQUEST_BODY)

    assert resp.status_code == 422


def test_agent_stream_rejects_wrong_admin_token(admin_token):
    with _client(MagicMock()) as client:
        resp = client.post(
            "/agent/stream", json=_REQUEST_BODY, headers={"X-Admin-Token": "wrong"}
        )

    assert resp.status_code == 401


def test_agent_stream_returns_503_when_graph_unavailable(admin_token):
    with _client(None) as client:
        resp = client.post(
            "/agent/stream", json=_REQUEST_BODY, headers={"X-Admin-Token": admin_token}
        )

    assert resp.status_code == 503


def test_agent_stream_blocks_moderated_input_before_touching_graph(admin_token):
    graph = MagicMock()
    graph.astream = MagicMock(side_effect=AssertionError("graph.astream не должен вызываться"))

    with _client(graph) as client:
        resp = client.post(
            "/agent/stream",
            json={
                "thread_id": "t1",
                "input": {"messages": [{"role": "user", "content": "Продам дамп карты, недорого"}]},
            },
            headers={"X-Admin-Token": admin_token},
        )

    assert resp.status_code == 403
    graph.astream.assert_not_called()
