from __future__ import annotations

import logging
from uuid import UUID

import httpx
from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from app.bot.keyboards.inline import topics_kb
from app.bot.services.backend_client import BackendClient
from app.bot.states import AskFlow

router = Router()
logger = logging.getLogger(__name__)

_TOPIC_LABELS: dict[str, str] = {
    "billing": "Биллинг",
    "transactions": "Транзакции",
    "api_errors": "Ошибки API",
    "integration": "Интеграция",
    "refunds": "Возвраты",
}


@router.message(Command("ask"))
async def cmd_ask(message: Message, state: FSMContext) -> None:
    await state.set_state(AskFlow.waiting_for_topic)
    await message.answer("Выберите тему вопроса:", reply_markup=topics_kb())


@router.callback_query(AskFlow.waiting_for_topic, F.data.startswith("topic:"))
async def cb_topic_selected(callback: CallbackQuery, state: FSMContext) -> None:
    slug = callback.data.split(":", 1)[1]

    if slug == "cancel":
        await state.clear()
        await callback.message.edit_text("Отменено.")
        await callback.answer()
        return

    label = _TOPIC_LABELS.get(slug, slug)
    await state.update_data(topic=slug, topic_label=label)
    await state.set_state(AskFlow.waiting_for_question)
    await callback.message.edit_text(
        f"Тема: <b>{label}</b>\n\nТеперь введите ваш вопрос:",
        parse_mode="HTML",
    )
    await callback.answer()


@router.message(AskFlow.waiting_for_question, F.text)
async def fsm_question_received(
    message: Message, state: FSMContext, backend: BackendClient
) -> None:
    data = await state.get_data()
    topic_label = data.get("topic_label", data.get("topic", ""))
    chat_id = data.get("backend_chat_id")

    if chat_id is None:
        try:
            cid = await backend.get_or_create_chat(
                str(message.from_user.id), "telegram"
            )
            await state.update_data(backend_chat_id=str(cid))
            chat_id = str(cid)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError):
            await state.clear()
            await message.answer("Не удалось подключиться к серверу. Попробуйте позже.")
            return

    prompt = f"Тема: {topic_label}. Вопрос: {message.text}"

    sent = await message.answer("⏳")
    buffer = ""

    try:
        import asyncio
        last_edit = asyncio.get_event_loop().time()

        async for chunk in backend.send_message(UUID(chat_id), prompt):
            buffer += chunk
            now = asyncio.get_event_loop().time()
            if now - last_edit >= 0.6:
                await sent.edit_text(buffer)
                last_edit = now

        if buffer:
            await sent.edit_text(buffer)
        else:
            await sent.edit_text("(нет ответа)")

    except (httpx.ConnectError, httpx.ReadTimeout):
        await sent.edit_text("Не удалось подключиться к серверу. Попробуйте позже.")
    except httpx.HTTPStatusError as e:
        await sent.edit_text(f"Ошибка сервера: {e.response.status_code}")
    finally:
        await state.clear()
