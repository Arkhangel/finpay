"""Tests for the bot-side broadcast poller."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.bot.services.broadcast import run_broadcast_worker


async def test_broadcast_worker_sends_to_all_targets_and_acks():
    bot = MagicMock()
    bot.send_message = AsyncMock()

    backend = MagicMock()
    backend.get_pending_broadcasts = AsyncMock(
        return_value=[{"id": "b1", "message": "hi all", "targets": [111, 222]}]
    )
    backend.ack_broadcast = AsyncMock()

    with patch("app.bot.services.broadcast.asyncio.sleep", side_effect=asyncio_cancel_after_first()):
        with pytest.raises(RuntimeError, match="stop-loop"):
            await run_broadcast_worker(bot, backend)

    assert bot.send_message.await_count == 2
    bot.send_message.assert_any_await(chat_id=111, text="hi all")
    bot.send_message.assert_any_await(chat_id=222, text="hi all")
    backend.ack_broadcast.assert_awaited_once_with("b1", "sent")


async def test_broadcast_worker_acks_failed_when_all_sends_fail():
    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=RuntimeError("telegram down"))

    backend = MagicMock()
    backend.get_pending_broadcasts = AsyncMock(
        return_value=[{"id": "b1", "message": "hi", "targets": [111]}]
    )
    backend.ack_broadcast = AsyncMock()

    with patch("app.bot.services.broadcast.asyncio.sleep", side_effect=asyncio_cancel_after_first()):
        with pytest.raises(RuntimeError, match="stop-loop"):
            await run_broadcast_worker(bot, backend)

    backend.ack_broadcast.assert_awaited_once_with("b1", "failed")


async def test_broadcast_worker_survives_poll_error():
    bot = MagicMock()
    backend = MagicMock()
    backend.get_pending_broadcasts = AsyncMock(side_effect=RuntimeError("backend down"))

    bot.send_message = AsyncMock()

    with patch("app.bot.services.broadcast.asyncio.sleep", side_effect=asyncio_cancel_after_first()):
        with pytest.raises(RuntimeError, match="stop-loop"):
            await run_broadcast_worker(bot, backend)

    bot.send_message.assert_not_called()


def asyncio_cancel_after_first():
    """First call to sleep raises to break the infinite polling loop in tests."""
    def _side_effect(*args, **kwargs):
        raise RuntimeError("stop-loop")
    return _side_effect
