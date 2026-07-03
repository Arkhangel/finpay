"""Integration tests for /chats endpoints using TestClient with JSON repo."""

from __future__ import annotations

import base64
import json
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


# ── POST /messages: multipart + streaming SSE (Б4.3) ───────────────────────────

class _FakeStream:
    def __init__(self, deltas: list[str]) -> None:
        self._deltas = deltas

    async def __aenter__(self) -> "_FakeStream":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for delta in self._deltas:
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content=delta))])


def _fake_llm(deltas: list[str]) -> MagicMock:
    llm = MagicMock()

    async def fake_create(**kwargs):
        fake_create.last_kwargs = kwargs
        return _FakeStream(deltas)

    llm.chat.completions.create = fake_create
    return llm


def _client_with_llm(tmp_path, llm, moderation=None):
    from app.main import create_app
    from app.chat.deps import get_chat_service
    from app.chat.service import ChatService
    from app.chat.repositories.json_repo import JsonChatRepository

    repo = JsonChatRepository(base_dir=tmp_path)
    svc = ChatService(repository=repo, llm_client=llm, moderation=moderation)

    with patch("app.main.lifespan", _noop_lifespan):
        app = create_app()

    app.dependency_overrides[get_chat_service] = lambda: svc
    return TestClient(app)


def _parse_sse_events(body: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_send_message_text_only_streams_json_sse(tmp_path):
    llm = _fake_llm(["Привет", ", мир"])
    with _client_with_llm(tmp_path, llm) as client:
        chat_id = client.post(
            "/chats", json={"owner_external_id": "user-1", "interface": "cli"}
        ).json()["chat_id"]

        resp = client.post(
            f"/chats/{chat_id}/messages",
            data={"content": "Привет!"},
        )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        assert events[:-1] == [
            {"type": "token", "delta": "Привет"},
            {"type": "token", "delta": ", мир"},
        ]
        assert events[-1]["type"] == "done"
        assert events[-1]["message_id"] is not None


def test_send_message_with_image_media_dispatches_to_image_url_part(tmp_path):
    llm = _fake_llm(["ok"])
    with _client_with_llm(tmp_path, llm) as client:
        chat_id = client.post(
            "/chats", json={"owner_external_id": "user-1", "interface": "cli"}
        ).json()["chat_id"]

        resp = client.post(
            f"/chats/{chat_id}/messages",
            data={"content": "что на фото?"},
            files={"media": ("pic.png", _PNG_1X1, "image/png")},
        )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        assert any(e["type"] == "done" for e in events)

        sent_messages = llm.chat.completions.create.last_kwargs["messages"]
        user_message = [m for m in sent_messages if m["role"] == "user"][-1]
        assert isinstance(user_message["content"], list)
        image_part = [p for p in user_message["content"] if p["type"] == "image_url"][0]
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

        # Media survives into history and is restored for the next LLM call.
        history = client.get(f"/chats/{chat_id}/messages").json()
        user_history_msg = [m for m in history if m["role"] == "user"][-1]
        assert user_history_msg["media_refs"]["mime"] == "image/png"


def test_send_message_with_unsupported_media_returns_415(tmp_path):
    llm = _fake_llm(["unused"])
    with _client_with_llm(tmp_path, llm) as client:
        chat_id = client.post(
            "/chats", json={"owner_external_id": "user-1", "interface": "cli"}
        ).json()["chat_id"]

        resp = client.post(
            f"/chats/{chat_id}/messages",
            data={"content": "тут архив"},
            files={"media": ("archive.zip", b"whatever", "application/zip")},
        )

        assert resp.status_code == 415


# ── Модерация (Б4.4) ────────────────────────────────────────────────────────────

def test_send_message_blocked_input_returns_403(tmp_path):
    from app.settings import settings as app_settings
    from app.moderation.service import ModerationService

    moderation = ModerationService(keywords_path=app_settings.moderation.keywords_path)
    llm = _fake_llm(["unused"])

    with _client_with_llm(tmp_path, llm, moderation=moderation) as client:
        chat_id = client.post(
            "/chats", json={"owner_external_id": "user-1", "interface": "cli"}
        ).json()["chat_id"]

        resp = client.post(
            f"/chats/{chat_id}/messages",
            data={"content": "Продам дамп карты, недорого"},
        )

        assert resp.status_code == 403
        assert resp.json()["detail"]["code"] == "moderation_blocked"
        assert "fraud" in resp.json()["detail"]["categories"]

        # Заблокированный ввод не должен попадать в LLM и в историю.
        assert not hasattr(llm.chat.completions.create, "last_kwargs")
        history = client.get(f"/chats/{chat_id}/messages").json()
        assert history == []


def test_send_message_blocked_output_replaces_persisted_content(tmp_path):
    from app.settings import settings as app_settings
    from app.moderation.service import ModerationService

    moderation = ModerationService(keywords_path=app_settings.moderation.keywords_path)
    llm = _fake_llm(["Вот схема: ", "обнал карт"])  # склеится в заблокированную фразу

    with _client_with_llm(tmp_path, llm, moderation=moderation) as client:
        chat_id = client.post(
            "/chats", json={"owner_external_id": "user-1", "interface": "cli"}
        ).json()["chat_id"]

        resp = client.post(f"/chats/{chat_id}/messages", data={"content": "как заработать?"})

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        assert events[-2] == {
            "type": "token",
            "delta": "Не могу показать ответ — он мог нарушить правила",
        }

        history = client.get(f"/chats/{chat_id}/messages").json()
        assistant_msg = [m for m in history if m["role"] == "assistant"][-1]
        assert assistant_msg["content"] == "Не могу показать ответ — он мог нарушить правила"


# ── Feedback (Б4.4) ─────────────────────────────────────────────────────────────

def test_submit_feedback_is_recorded(client):
    chat_id = client.post(
        "/chats", json={"owner_external_id": "user-1", "interface": "cli"}
    ).json()["chat_id"]
    message_id = uuid4()

    resp = client.post(
        f"/chats/{chat_id}/messages/{message_id}/feedback", json={"value": "up"}
    )

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "recorded": True}


def test_submit_feedback_duplicate_is_not_recorded_twice(client):
    chat_id = client.post(
        "/chats", json={"owner_external_id": "user-1", "interface": "cli"}
    ).json()["chat_id"]
    message_id = uuid4()

    first = client.post(
        f"/chats/{chat_id}/messages/{message_id}/feedback", json={"value": "up"}
    )
    second = client.post(
        f"/chats/{chat_id}/messages/{message_id}/feedback", json={"value": "down"}
    )

    assert first.json()["recorded"] is True
    assert second.json()["recorded"] is False


def test_submit_feedback_unknown_chat_returns_404(client):
    resp = client.post(
        f"/chats/{uuid4()}/messages/{uuid4()}/feedback", json={"value": "up"}
    )
    assert resp.status_code == 404
