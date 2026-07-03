"""Tests for the 👍/👎 feedback callback handler and keyboard."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage

from app.bot.handlers.feedback import handle_feedback
from app.bot.keyboards.feedback import feedback_kb


@pytest.fixture
async def state_with_chat():
    storage = MemoryStorage()
    state = FSMContext(storage=storage, key=StorageKey(bot_id=1, chat_id=100, user_id=42))
    await state.update_data(backend_chat_id=str(uuid4()))
    return state


def test_feedback_kb_has_up_and_down_with_message_id():
    message_id = uuid4()
    kb = feedback_kb(message_id)
    callback_data = [btn.callback_data for row in kb.inline_keyboard for btn in row]
    assert f"fb:up:{message_id}" in callback_data
    assert f"fb:down:{message_id}" in callback_data


async def test_handle_feedback_submits_and_clears_keyboard(state_with_chat):
    message_id = uuid4()
    callback = MagicMock()
    callback.data = f"fb:up:{message_id}"
    callback.message = MagicMock()
    callback.message.edit_reply_markup = AsyncMock()
    callback.answer = AsyncMock()

    backend = MagicMock()
    backend.submit_feedback = AsyncMock()

    await handle_feedback(callback, state_with_chat, backend)

    backend.submit_feedback.assert_awaited_once()
    call_args = backend.submit_feedback.call_args[0]
    assert call_args[1] == message_id
    assert call_args[2] == "up"
    callback.message.edit_reply_markup.assert_awaited_once_with(reply_markup=None)
    callback.answer.assert_awaited_once_with("Спасибо за отзыв!")


async def test_handle_feedback_without_chat_session_shows_alert():
    storage = MemoryStorage()
    state = FSMContext(storage=storage, key=StorageKey(bot_id=1, chat_id=100, user_id=42))

    callback = MagicMock()
    callback.data = f"fb:down:{uuid4()}"
    callback.answer = AsyncMock()
    backend = MagicMock()
    backend.submit_feedback = AsyncMock()

    await handle_feedback(callback, state, backend)

    backend.submit_feedback.assert_not_called()
    callback.answer.assert_awaited_once()
    assert callback.answer.call_args.kwargs.get("show_alert") is True


async def test_handle_feedback_backend_error_shows_friendly_alert(state_with_chat):
    callback = MagicMock()
    callback.data = f"fb:up:{uuid4()}"
    callback.answer = AsyncMock()

    backend = MagicMock()
    backend.submit_feedback = AsyncMock(side_effect=httpx.ConnectError("boom"))

    await handle_feedback(callback, state_with_chat, backend)

    callback.answer.assert_awaited_once_with("Сервис недоступен, попробуйте позже", show_alert=True)
