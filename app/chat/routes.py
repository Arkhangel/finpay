from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.chat.deps import ChatServiceDep
from app.chat.domain import Chat, ChatMessage

router = APIRouter(prefix="/chats", tags=["chats"])


class CreateChatIn(BaseModel):
    owner_external_id: str
    interface: str
    system_prompt: str | None = None


class CreateChatOut(BaseModel):
    chat_id: UUID


class MessageIn(BaseModel):
    content: str


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
async def send_message(chat_id: UUID, body: MessageIn, svc: ChatServiceDep) -> StreamingResponse:
    async def generator():
        async for chunk in svc.send_message(chat_id, body.content):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@router.get("/{chat_id}/messages", response_model=list[ChatMessage])
async def list_messages(
    chat_id: UUID, svc: ChatServiceDep, limit: int = 50
) -> list[ChatMessage]:
    return await svc.list_messages(chat_id, limit=limit)


@router.delete("/{chat_id}/messages")
async def clear_messages(chat_id: UUID, svc: ChatServiceDep) -> dict:
    await svc.clear_history(chat_id)
    return {"status": "ok"}
