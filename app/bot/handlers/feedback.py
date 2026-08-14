from __future__ import annotations

import logging
from uuid import UUID

import httpx
from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery

from app.bot.services import sources_cache
from app.bot.services.backend_client import BackendClient
from app.bot.services.streaming import friendly_error

router = Router()
logger = logging.getLogger(__name__)


@router.callback_query(F.data.startswith("fb:"))
async def handle_feedback(callback: CallbackQuery, state: FSMContext, backend: BackendClient) -> None:
    _, vote, message_id_raw = callback.data.split(":", 2)

    data = await state.get_data()
    chat_id = data.get("backend_chat_id")
    if not chat_id:
        await callback.answer("Сессия устарела, отзыв не сохранён.", show_alert=True)
        return

    message_id = UUID(message_id_raw)
    sources = sources_cache.pop(message_id)

    try:
        await backend.submit_feedback(UUID(chat_id), message_id, vote, sources=sources)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.HTTPStatusError) as exc:
        await callback.answer(friendly_error(exc), show_alert=True)
        return
    except Exception:
        logger.exception("unexpected_error_in_feedback_handler")
        await callback.answer("Не удалось сохранить отзыв.", show_alert=True)
        return

    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.answer("Спасибо за отзыв!")
