"""Tests for AskFlow FSM scenario."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.storage.base import StorageKey

from app.bot.states import AskFlow


def _make_fsm_context() -> FSMContext:
    storage = MemoryStorage()
    key = StorageKey(bot_id=1, chat_id=100, user_id=42)
    return FSMContext(storage=storage, key=key)


# ── /ask → waiting_for_topic ──────────────────────────────────────────────────

async def test_ask_command_sets_waiting_for_topic_state():
    from app.bot.handlers.fsm import cmd_ask

    state = _make_fsm_context()
    message = MagicMock()
    message.answer = AsyncMock()

    await cmd_ask(message, state)

    assert await state.get_state() == AskFlow.waiting_for_topic
    message.answer.assert_called_once()


# ── topic callback → waiting_for_question ────────────────────────────────────

async def test_topic_callback_transitions_to_waiting_for_question():
    from app.bot.handlers.fsm import cb_topic_selected

    state = _make_fsm_context()
    await state.set_state(AskFlow.waiting_for_topic)

    callback = MagicMock()
    callback.data = "topic:billing"
    callback.message = MagicMock()
    callback.message.edit_text = AsyncMock()
    callback.answer = AsyncMock()

    await cb_topic_selected(callback, state)

    assert await state.get_state() == AskFlow.waiting_for_question
    data = await state.get_data()
    assert data["topic"] == "billing"
    assert data["topic_label"] == "Биллинг"


async def test_cancel_callback_clears_state():
    from app.bot.handlers.fsm import cb_topic_selected

    state = _make_fsm_context()
    await state.set_state(AskFlow.waiting_for_topic)

    callback = MagicMock()
    callback.data = "topic:cancel"
    callback.message = MagicMock()
    callback.message.edit_text = AsyncMock()
    callback.answer = AsyncMock()

    await cb_topic_selected(callback, state)

    assert await state.get_state() is None


# ── question in waiting_for_question → state cleared after response ───────────

async def test_question_clears_state_after_response():
    from app.bot.handlers.fsm import fsm_question_received

    state = _make_fsm_context()
    await state.set_state(AskFlow.waiting_for_question)
    await state.update_data(
        topic="billing",
        topic_label="Биллинг",
        backend_chat_id=str(uuid4()),
    )

    message = MagicMock()
    message.from_user = MagicMock(id=42)
    message.text = "Как оплатить счёт?"
    message.answer = AsyncMock(return_value=MagicMock(edit_text=AsyncMock()))

    async def fake_stream(*args, **kwargs):
        yield "Ответ на вопрос"

    backend = MagicMock()
    backend.send_message = fake_stream

    await fsm_question_received(message, state, backend)

    assert await state.get_state() is None
