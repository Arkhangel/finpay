from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.deps.providers import LLMServiceDep
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Синхронный ответ LLM",
    responses={200: {}, 422: {}, 429: {}, 502: {}, 504: {}},
)
async def chat_complete(req: ChatRequest, svc: LLMServiceDep) -> ChatResponse:
    return await svc.complete(req)


@router.post(
    "/stream",
    summary="Стриминг ответа LLM (SSE)",
    responses={200: {}, 422: {}, 429: {}, 502: {}, 504: {}},
)
async def chat_stream(req: ChatRequest, svc: LLMServiceDep) -> StreamingResponse:
    async def generator():
        async for delta in svc.stream(req):
            if delta.usage:
                yield f"data: {json.dumps({'usage': delta.usage.model_dump()})}\n\n"
            elif delta.content:
                yield f"data: {delta.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
