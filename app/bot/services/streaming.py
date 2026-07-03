"""Стриминг ответа в Telegram через нативный sendMessageDraft (Bot API 10.0)."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import httpx
from aiogram import Bot
from aiogram.enums import ChatAction
from aiogram.types import Message

_TYPING_INTERVAL = 4.0  # Telegram "печатает..." индикатор живёт ~5 сек


async def _keep_typing(bot: Bot, chat_id: int) -> None:
    try:
        while True:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(_TYPING_INTERVAL)
    except asyncio.CancelledError:
        pass


async def stream_to_chat(message: Message, tokens: AsyncIterator[str]) -> Message:
    """Возвращает итоговое отправленное сообщение (например, чтобы навесить клавиатуру фидбека)."""
    # Один draft_id на весь стрим — Telegram склеивает вызовы в один черновик;
    # без финального send_message черновик исчезает сам через ~30 сек.
    bot = message.bot
    draft_id = uuid.uuid4().int & 0xFFFFFFFF
    buffer = ""

    typing_task = asyncio.create_task(_keep_typing(bot, message.chat.id))
    try:
        await bot.send_message_draft(chat_id=message.chat.id, text="", draft_id=draft_id)

        async for delta in tokens:
            if not buffer:
                typing_task.cancel()
            buffer += delta
            if buffer.strip():
                await bot.send_message_draft(
                    chat_id=message.chat.id, text=buffer, draft_id=draft_id,
                )
    finally:
        typing_task.cancel()

    return await bot.send_message(chat_id=message.chat.id, text=buffer or "(нет ответа)")


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, httpx.ConnectError):
        return "Сервис недоступен, попробуйте позже"
    if isinstance(exc, httpx.ReadTimeout):
        return "Ответ занимает слишком долго"
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 401:
            return "Доступ запрещён: неверный токен"
        if status == 403:
            return "Действие запрещено правилами модерации"
        if status == 429:
            return "Слишком много запросов, подождите минуту"
        if status == 503:
            return "Функция временно недоступна на бэкенде"
        if status >= 500:
            return "Внутренняя ошибка сервиса"
        return f"Ошибка сервера: {status}"
    return "Произошла непредвиденная ошибка. Попробуйте ещё раз."
