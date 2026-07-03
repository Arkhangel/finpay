"""Tests for BackendClient using httpx.MockTransport."""

from __future__ import annotations

import json
from uuid import UUID, uuid4

import httpx
import pytest

from app.bot.services.backend_client import BackendClient


def _make_client(transport: httpx.MockTransport) -> BackendClient:
    client = BackendClient(base_url="http://test")
    client._client = httpx.AsyncClient(
        base_url="http://test", transport=transport, timeout=5
    )
    return client


# ── get_or_create_chat ────────────────────────────────────────────────────────

async def test_get_or_create_chat_returns_uuid():
    chat_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"chat_id": str(chat_id)})

    client = _make_client(httpx.MockTransport(handler))
    result = await client.get_or_create_chat("user-1", "telegram")
    assert result == chat_id
    await client.close()


async def test_get_or_create_chat_is_idempotent():
    """Second call with same params must not hit the backend again."""
    call_count = 0
    chat_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json={"chat_id": str(chat_id)})

    client = _make_client(httpx.MockTransport(handler))
    r1 = await client.get_or_create_chat("user-1", "telegram")
    r2 = await client.get_or_create_chat("user-1", "telegram")
    assert r1 == r2
    assert call_count == 1
    await client.close()


# ── send_message (SSE, JSON payloads) ──────────────────────────────────────────

def _sse(*events: dict) -> str:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events)


async def test_send_message_parses_sse_token_events():
    sse_body = _sse(
        {"type": "token", "delta": "Привет"},
        {"type": "token", "delta": " мир"},
        {"type": "done"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse_body, headers={"content-type": "text/event-stream"})

    client = _make_client(httpx.MockTransport(handler))
    chunks = [c async for c in client.send_message(uuid4(), "hello")]
    assert chunks == ["Привет", " мир"]
    await client.close()


async def test_send_message_fills_result_with_message_id():
    message_id = uuid4()
    sse_body = _sse(
        {"type": "token", "delta": "ok"},
        {"type": "done", "message_id": str(message_id)},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse_body, headers={"content-type": "text/event-stream"})

    client = _make_client(httpx.MockTransport(handler))
    result: dict = {}
    [c async for c in client.send_message(uuid4(), "hello", result=result)]
    assert result["message_id"] == message_id
    await client.close()


async def test_send_message_stops_at_done():
    sse_body = _sse(
        {"type": "token", "delta": "chunk1"},
        {"type": "done"},
    ) + _sse({"type": "token", "delta": "chunk2"})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse_body, headers={"content-type": "text/event-stream"})

    client = _make_client(httpx.MockTransport(handler))
    chunks = [c async for c in client.send_message(uuid4(), "hello")]
    assert "chunk2" not in chunks
    assert "chunk1" in chunks
    await client.close()


async def test_send_message_without_media_sends_no_files():
    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(
            200,
            text=_sse({"type": "done"}),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(httpx.MockTransport(handler))
    [c async for c in client.send_message(uuid4(), "hello")]

    assert len(received) == 1
    content_type = received[0].headers.get("content-type", "")
    assert "multipart/form-data" not in content_type
    await client.close()


async def test_send_message_with_media_sends_multipart():
    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(
            200,
            text=_sse({"type": "token", "delta": "ok"}, {"type": "done"}),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(httpx.MockTransport(handler))
    chat_id = uuid4()
    chunks = [
        c
        async for c in client.send_message(
            chat_id, "caption", media=b"\x89PNG\r\n", mime="image/png"
        )
    ]

    assert chunks == ["ok"]
    assert len(received) == 1
    request = received[0]
    assert str(chat_id) in str(request.url)
    assert "multipart/form-data" in request.headers.get("content-type", "")
    assert b"caption" in request.content
    assert b"\x89PNG\r\n" in request.content
    await client.close()


# ── clear_messages ────────────────────────────────────────────────────────────

async def test_clear_messages_sends_delete_to_correct_url():
    chat_id = uuid4()
    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(200, json={"status": "ok"})

    client = _make_client(httpx.MockTransport(handler))
    await client.clear_messages(chat_id)
    assert len(received) == 1
    assert received[0].method == "DELETE"
    assert str(chat_id) in str(received[0].url)
    await client.close()


# ── retry on connect errors ─────────────────────────────────────────────────────

async def test_get_or_create_chat_retries_on_connect_error_then_succeeds():
    attempts = 0
    chat_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json={"chat_id": str(chat_id)})

    client = _make_client(httpx.MockTransport(handler))
    result = await client.get_or_create_chat("user-1", "telegram")
    assert result == chat_id
    assert attempts == 3
    await client.close()


async def test_get_or_create_chat_does_not_retry_on_http_status_error():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, json={"detail": "boom"})

    client = _make_client(httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await client.get_or_create_chat("user-1", "telegram")
    assert attempts == 1
    await client.close()
