from __future__ import annotations

import json
import logging
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.chat import feedback
from app.chat.deps import ChatRepositoryDep, ChatServiceDep, OpenAIClientDep
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


class FeedbackIn(BaseModel):
    value: Literal["up", "down"]
    # Источники, показанные пользователю вместе с оценённым ответом (Б5.5) —
    # опционально: клиенты, ещё не подключённые к RAG, могут его не слать.
    sources: list[dict] | None = None


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
    # Проверяется здесь, а не внутри ChatService.send_message: генератор уже
    # успел бы закоммитить статус 200 к моменту, когда там всплыло бы исключение.
    mod_result = await svc.check_input(chat_id, content)
    if not mod_result.allowed:
        raise HTTPException(
            status_code=403,
            detail={"code": "moderation_blocked", "categories": mod_result.categories},
        )

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
        result: dict = {}
        # json.dumps экранирует \n внутри delta как "\n" (два символа) —
        # сырой перевод строки никогда не попадает в SSE data-строку, так что
        # многострочный chunk не рвёт формат события.
        async for delta in svc.send_message(chat_id, content, media_part, media_meta, result=result):
            yield f"data: {json.dumps({'type': 'token', 'delta': delta}, ensure_ascii=False)}\n\n"

        # sources — до done: клиенты (веб/бот), которые останавливают чтение
        # стрима сразу после "done", не должны пропустить финальный event.
        sources = result.get("sources")
        if sources is not None:
            yield f"event: sources\ndata: {json.dumps({'sources': sources}, ensure_ascii=False)}\n\n"

        message_id = result.get("message_id")
        yield f"data: {json.dumps({'type': 'done', 'message_id': str(message_id) if message_id else None})}\n\n"

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


@router.post("/{chat_id}/messages/{message_id}/feedback")
async def submit_feedback(
    chat_id: UUID, message_id: UUID, body: FeedbackIn, svc: ChatServiceDep, repo: ChatRepositoryDep
) -> dict:
    chat = await svc.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    # global-аудит: раньше message_id не проверялся вообще — JSON-бэкенд тихо
    # принимал фидбек на несуществующее/чужое сообщение, а Postgres ронял
    # необработанный IntegrityError (FK message_id -> chat_messages.id) в 500.
    if not await repo.message_exists(chat_id, message_id):
        raise HTTPException(status_code=404, detail="Message not found in this chat")

    created = await feedback.save_feedback(
        repo, message_id, chat.owner_external_id, body.value, body.sources
    )
    return {"status": "ok", "recorded": created}


@router.delete("/{chat_id}/messages")
async def clear_messages(chat_id: UUID, svc: ChatServiceDep) -> dict:
    await svc.clear_history(chat_id)
    return {"status": "ok"}
