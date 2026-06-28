"""Integration tests for /chats endpoints using TestClient with JSON repo."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


@asynccontextmanager
async def _noop_lifespan(app):
    """Minimal lifespan that sets required app.state fields without real connections."""
    app.state.openai = MagicMock()
    app.state.cache = None
    app.state.pg_engine = None
    app.state.pg_session_factory = None
    app.state.canary = "test_canary"
    yield


@pytest.fixture
def client(tmp_path):
    from app.main import create_app
    from app.chat.deps import get_chat_service
    from app.chat.service import ChatService
    from app.chat.repositories.json_repo import JsonChatRepository

    repo = JsonChatRepository(base_dir=tmp_path)
    svc = ChatService(repository=repo, llm_client=MagicMock())

    with patch("app.main.lifespan", _noop_lifespan):
        app = create_app()

    app.dependency_overrides[get_chat_service] = lambda: svc

    with TestClient(app) as c:
        yield c


def test_create_chat(client):
    resp = client.post("/chats", json={"owner_external_id": "user-1", "interface": "cli"})
    assert resp.status_code == 200
    assert "chat_id" in resp.json()


def test_get_chat(client):
    chat_id = client.post(
        "/chats", json={"owner_external_id": "user-1", "interface": "cli"}
    ).json()["chat_id"]

    resp = client.get(f"/chats/{chat_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == chat_id
    assert data["interface"] == "cli"


def test_get_chat_not_found(client):
    assert client.get(f"/chats/{uuid4()}").status_code == 404


def test_list_messages_empty(client):
    chat_id = client.post(
        "/chats", json={"owner_external_id": "user-1", "interface": "cli"}
    ).json()["chat_id"]

    resp = client.get(f"/chats/{chat_id}/messages")
    assert resp.status_code == 200
    assert resp.json() == []


def test_delete_messages(client):
    chat_id = client.post(
        "/chats", json={"owner_external_id": "user-1", "interface": "cli"}
    ).json()["chat_id"]

    resp = client.delete(f"/chats/{chat_id}/messages")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
