from __future__ import annotations

import asyncio
import logging
from uuid import UUID

import httpx
from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from app.bot.services.backend_client import BackendClient

router = Router()
logger = logging.getLogger(__name__)

_EDIT_INTERVAL = 0.6  # seconds between edit_text calls to avoid Telegram rate limits


@router.message(F.text & ~F.text.startswith("/"))
async def handle_text(
    message: Message, state: FSMContext, backend: BackendClient
) -> None:
    data = await state.get_data()
    chat_id = data.get("backend_chat_id")

    if chat_id is None:
        try:
            cid = await backend.get_or_create_chat(
                str(message.from_user.id), "telegram"
            )
            await state.update_data(backend_chat_id=str(cid))
            chat_id = str(cid)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError):
            await message.answer("Не удалось подключиться к серверу. Попробуйте позже.")
            return

    sent = await message.answer("⏳")
    buffer = ""
    last_edit = asyncio.get_event_loop().time()

    try:
        async for chunk in backend.send_message(UUID(chat_id), message.text):
            buffer += chunk
            now = asyncio.get_event_loop().time()
            if now - last_edit >= _EDIT_INTERVAL:
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
    except Exception:
        logger.exception("unexpected_error_in_text_handler")
        await sent.edit_text("Произошла непредвиденная ошибка. Попробуйте ещё раз.")
