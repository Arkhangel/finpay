from __future__ import annotations

from uuid import UUID

import httpx
from aiogram import Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from app.bot.services.backend_client import BackendClient
from app.bot.services.streaming import friendly_error

router = Router()


async def _get_or_init_chat(
    message: Message, state: FSMContext, backend: BackendClient
) -> UUID:
    """Returns the cached/created backend chat id. Raises on network/HTTP errors."""
    data = await state.get_data()
    chat_id = data.get("backend_chat_id")
    if chat_id:
        return UUID(chat_id)
    chat_id = await backend.get_or_create_chat(str(message.from_user.id), "telegram")
    await state.update_data(backend_chat_id=str(chat_id))
    return chat_id


@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext, backend: BackendClient) -> None:
    try:
        await _get_or_init_chat(message, state, backend)
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as exc:
        await message.answer(friendly_error(exc))
        return
    await message.answer(
        "Привет! Я FinPay-ассистент 💳\n\n"
        "Помогу разобраться с биллингом, транзакциями и интеграцией платёжного шлюза.\n\n"
        "Просто напишите вопрос или воспользуйтесь командами:\n"
        "/ask — задать вопрос по теме\n"
        "/clear — очистить историю диалога\n"
        "/help — список команд"
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "Доступные команды:\n"
        "/start — начать диалог\n"
        "/ask — задать вопрос по теме (с выбором категории)\n"
        "/clear — очистить историю диалога\n"
        "/cancel — отменить текущий сценарий\n"
        "/help — это сообщение"
    )


@router.message(Command("clear"))
async def cmd_clear(message: Message, state: FSMContext, backend: BackendClient) -> None:
    data = await state.get_data()
    chat_id = data.get("backend_chat_id")
    if not chat_id:
        await message.answer("История пуста — диалог ещё не начат.")
        return
    try:
        await backend.clear_messages(UUID(chat_id))
        await message.answer("История очищена. Начинаем с чистого листа ✨")
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as exc:
        await message.answer(friendly_error(exc))


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    current = await state.get_state()
    if current is None:
        await message.answer("Нет активного сценария.")
        return
    await state.clear()
    await message.answer("Сценарий отменён.")
