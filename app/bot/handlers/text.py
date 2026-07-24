from __future__ import annotations

import logging
from uuid import UUID

import httpx
from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from app.bot.keyboards.feedback import feedback_kb
from app.bot.services import sources_cache
from app.bot.services.backend_client import BackendClient
from app.bot.services.streaming import friendly_error, stream_to_chat

router = Router()
logger = logging.getLogger(__name__)


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
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as exc:
            await message.answer(friendly_error(exc))
            return

    try:
        result: dict = {}
        sent = await stream_to_chat(message, backend.send_message(UUID(chat_id), message.text, result=result))
        if result.get("message_id"):
            sources_cache.remember(result["message_id"], result.get("sources"))
            await sent.edit_reply_markup(reply_markup=feedback_kb(result["message_id"]))
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as exc:
        await message.answer(friendly_error(exc))
    except Exception:
        logger.exception("unexpected_error_in_text_handler")
        await message.answer("Произошла непредвиденная ошибка. Попробуйте ещё раз.")
