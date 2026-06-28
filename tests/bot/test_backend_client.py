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


# ── send_message (SSE) ────────────────────────────────────────────────────────

async def test_send_message_parses_sse_chunks():
    sse_body = "data: Привет\n\ndata:  мир\n\ndata: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse_body, headers={"content-type": "text/event-stream"})

    client = _make_client(httpx.MockTransport(handler))
    chunks = [c async for c in client.send_message(uuid4(), "hello")]
    assert chunks == ["Привет", " мир"]
    await client.close()


async def test_send_message_stops_at_done():
    sse_body = "data: chunk1\n\ndata: [DONE]\n\ndata: chunk2\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse_body, headers={"content-type": "text/event-stream"})

    client = _make_client(httpx.MockTransport(handler))
    chunks = [c async for c in client.send_message(uuid4(), "hello")]
    assert "chunk2" not in chunks
    assert "chunk1" in chunks
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
