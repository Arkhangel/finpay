"""Tests for admin bot commands and the IsAdmin router-level filter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from aiogram.filters import CommandObject

from app.bot.handlers.admin import IsAdmin, cmd_broadcast, cmd_stats, cmd_users, router
from app.settings import settings


@pytest.fixture(autouse=True)
def _admin_ids(monkeypatch):
    monkeypatch.setattr(settings.bot, "admin_ids", [42])


# ── IsAdmin filter ───────────────────────────────────────────────────────────────

async def test_is_admin_allows_configured_id():
    message = MagicMock(from_user=MagicMock(id=42))
    assert await IsAdmin()(message) is True


async def test_is_admin_rejects_other_id():
    message = MagicMock(from_user=MagicMock(id=999))
    assert await IsAdmin()(message) is False


async def test_is_admin_rejects_missing_user():
    message = MagicMock(from_user=None)
    assert await IsAdmin()(message) is False


# ── IsAdmin actually wired at router level (not just the predicate) ─────────────
#
# aiogram checks router-level filters (registered via `router.message.filter(...)`)
# through `observer.check_root_filters(...)`, called by `Router._propagate_event`
# *before* `observer.trigger(...)` ever looks at individual @router.message handlers.
# Calling `trigger()` directly would bypass this gate entirely, so the test below
# exercises the exact method aiogram itself uses to decide "is a router-level
# filter satisfied for this event".

async def test_router_level_filter_blocks_non_admin():
    non_admin = MagicMock(from_user=MagicMock(id=999))
    passed, _ = await router.observers["message"].check_root_filters(non_admin)
    assert passed is False


async def test_router_level_filter_allows_admin():
    admin = MagicMock(from_user=MagicMock(id=42))
    passed, _ = await router.observers["message"].check_root_filters(admin)
    assert passed is True


# ── /stats ────────────────────────────────────────────────────────────────────

async def test_cmd_stats_formats_response():
    message = MagicMock()
    message.answer = AsyncMock()
    backend = MagicMock()
    backend.get_admin_stats = AsyncMock(
        return_value={
            "total_messages_24h": 10,
            "active_users_24h": 3,
            "avg_latency_ms": 512.3,
            "moderation_block_rate": 0.1,
            "feedback_up_ratio": 0.75,
        }
    )

    await cmd_stats(message, backend)

    text = message.answer.call_args[0][0]
    assert "10" in text
    assert "3" in text
    assert "512" in text
    assert "10.0%" in text
    assert "75.0%" in text


async def test_cmd_stats_backend_error_shows_friendly_message():
    message = MagicMock()
    message.answer = AsyncMock()
    backend = MagicMock()
    backend.get_admin_stats = AsyncMock(side_effect=httpx.ConnectError("boom"))

    await cmd_stats(message, backend)

    message.answer.assert_awaited_once_with("Сервис недоступен, попробуйте позже")


# ── /users ────────────────────────────────────────────────────────────────────

async def test_cmd_users_lists_first_ten():
    message = MagicMock()
    message.answer = AsyncMock()
    backend = MagicMock()
    backend.get_admin_users = AsyncMock(
        return_value=[
            {"owner_external_id": f"tg-{i}", "interface": "telegram", "chats_count": 1, "last_seen_at": "now"}
            for i in range(15)
        ]
    )

    await cmd_users(message, backend)

    text = message.answer.call_args[0][0]
    assert "tg-0" in text
    assert "tg-9" in text
    assert "tg-10" not in text  # только первые 10


async def test_cmd_users_empty():
    message = MagicMock()
    message.answer = AsyncMock()
    backend = MagicMock()
    backend.get_admin_users = AsyncMock(return_value=[])

    await cmd_users(message, backend)

    message.answer.assert_awaited_once_with("Пользователей пока нет.")


# ── /broadcast ────────────────────────────────────────────────────────────────

async def test_cmd_broadcast_without_text_shows_usage():
    message = MagicMock()
    message.answer = AsyncMock()
    backend = MagicMock()
    backend.post_admin_broadcast = AsyncMock()
    command = CommandObject(command="broadcast", args=None)

    await cmd_broadcast(message, command, backend)

    backend.post_admin_broadcast.assert_not_called()
    assert "Использование" in message.answer.call_args[0][0]


async def test_cmd_broadcast_enqueues_message():
    message = MagicMock()
    message.answer = AsyncMock()
    backend = MagicMock()
    backend.post_admin_broadcast = AsyncMock(return_value={"id": "abc-123", "status": "pending"})
    command = CommandObject(command="broadcast", args="всем привет")

    await cmd_broadcast(message, command, backend)

    backend.post_admin_broadcast.assert_awaited_once_with("всем привет", interface_filter="telegram")
    assert "abc-123" in message.answer.call_args[0][0]
