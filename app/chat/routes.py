from __future__ import annotations

import json
import logging
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.chat.deps import ChatServiceDep, OpenAIClientDep
from app.chat.domain import Chat, ChatMessage
from app.chat.media import media_to_part

router = APIRouter(prefix="/chats", tags=["chats"])
logger = logging.getLogger(__name__)


class CreateChatIn(BaseModel):
    owner_external_id: str
    interface: str
    system_prompt: str | None = None


class CreateChatOut(BaseModel):
    chat_id: UUID


class SystemMessageIn(BaseModel):
    text: str
    notify: bool = False


@router.post("", response_model=CreateChatOut, status_code=200)
async def create_chat(body: CreateChatIn, svc: ChatServiceDep) -> CreateChatOut:
    chat = await svc.create_chat(
        owner_external_id=body.owner_external_id,
        interface=body.interface,
        system_prompt=body.system_prompt,
    )
    return CreateChatOut(chat_id=chat.id)


@router.get("/{chat_id}", response_model=Chat)
async def get_chat(chat_id: UUID, svc: ChatServiceDep) -> Chat:
    chat = await svc.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.post("/{chat_id}/messages")
async def send_message(
    chat_id: UUID,
    svc: ChatServiceDep,
    openai_client: OpenAIClientDep,
    content: str = Form(...),
    media: UploadFile | None = File(None),
) -> StreamingResponse:
    media_part: dict | None = None
    media_meta: dict | None = None

    if media is not None:
        try:
            media_part = await media_to_part(media, openai_client)
        except ValueError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        media_meta = {
            "mime": media.content_type,
            "size": media.size,
            "filename": media.filename,
        }

    async def generator():
        async for delta in svc.send_message(chat_id, content, media_part, media_meta):
            yield f"data: {json.dumps({'type': 'token', 'delta': delta}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@router.post("/{chat_id}/system-message")
async def system_message(chat_id: UUID, body: SystemMessageIn, svc: ChatServiceDep) -> dict:
    """Демо-эндпоинт: симуляция завершения фоновой задачи (статус заявки, подписка и т.п.)."""
    chat = await svc.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    await svc.append_system_message(chat_id, body.text)

    if body.notify:
        from app.services.notifier import notify_user

        try:
            await notify_user(int(chat.owner_external_id), body.text)
        except Exception as exc:  # noqa: BLE001 - notify failure must not break the endpoint
            logger.warning("notify_failed chat_id=%s error=%s", chat_id, exc)

    return {"status": "ok"}


@router.get("/{chat_id}/messages", response_model=list[ChatMessage])
async def list_messages(
    chat_id: UUID, svc: ChatServiceDep, limit: int = 50
) -> list[ChatMessage]:
    return await svc.list_messages(chat_id, limit=limit)


@router.delete("/{chat_id}/messages")
async def clear_messages(chat_id: UUID, svc: ChatServiceDep) -> dict:
    await svc.clear_history(chat_id)
    return {"status": "ok"}
