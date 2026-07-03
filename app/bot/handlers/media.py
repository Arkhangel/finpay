from __future__ import annotations

import logging
from io import BytesIO
from uuid import UUID

import httpx
from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from app.bot.services.backend_client import BackendClient
from app.bot.services.streaming import friendly_error, stream_to_chat

router = Router()
logger = logging.getLogger(__name__)

_MAX_PHOTO_BYTES = 2 * 1024 * 1024
_MAX_DOCUMENT_BYTES = 10 * 1024 * 1024


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


async def _download(message: Message, file_id: str) -> bytes:
    bot = message.bot
    file = await bot.get_file(file_id)
    buffer = BytesIO()
    await bot.download_file(file.file_path, destination=buffer)
    return buffer.getvalue()


async def _send_media(
    message: Message,
    state: FSMContext,
    backend: BackendClient,
    media_bytes: bytes,
    mime: str,
) -> None:
    try:
        chat_id = await _get_or_init_chat(message, state, backend)
        caption = message.caption or ""
        await stream_to_chat(
            message, backend.send_message(chat_id, caption, media=media_bytes, mime=mime)
        )
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as exc:
        await message.answer(friendly_error(exc))
    except Exception:
        logger.exception("unexpected_error_in_media_handler")
        await message.answer("Произошла непредвиденная ошибка. Попробуйте ещё раз.")


@router.message(F.photo)
async def handle_photo(message: Message, state: FSMContext, backend: BackendClient) -> None:
    # message.photo отсортирован от меньшего размера к большему.
    candidates = [p for p in message.photo if (p.file_size or 0) <= _MAX_PHOTO_BYTES]
    photo = candidates[-1] if candidates else message.photo[0]

    data = await _download(message, photo.file_id)
    await _send_media(message, state, backend, data, mime="image/jpeg")


@router.message(F.voice)
async def handle_voice(message: Message, state: FSMContext, backend: BackendClient) -> None:
    data = await _download(message, message.voice.file_id)
    await _send_media(message, state, backend, data, mime="audio/ogg")


@router.message(F.audio)
async def handle_audio(message: Message, state: FSMContext, backend: BackendClient) -> None:
    data = await _download(message, message.audio.file_id)
    mime = message.audio.mime_type or "audio/mpeg"
    await _send_media(message, state, backend, data, mime=mime)


@router.message(
    F.document
    & F.document.file_name.func(lambda name: (name or "").lower().endswith((".pdf", ".docx")))
    & F.document.file_size.func(lambda size: (size or 0) <= _MAX_DOCUMENT_BYTES)
)
async def handle_document(message: Message, state: FSMContext, backend: BackendClient) -> None:
    data = await _download(message, message.document.file_id)
    mime = message.document.mime_type or "application/octet-stream"
    await _send_media(message, state, backend, data, mime=mime)
