from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.chat.deps import get_moderation_service
from app.deps.providers import RAGServiceDep
from app.moderation import ModerationService
from app.schemas.rag import RagQueryRequest, RagQueryResponse

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post(
    "/query",
    response_model=RagQueryResponse,
    summary="RAG-ответ по базе знаний (блок 5.3)",
    responses={200: {}, 403: {}, 422: {}, 503: {}},
)
async def rag_query(
    req: RagQueryRequest,
    service: RAGServiceDep,
    moderation: Annotated[ModerationService | None, Depends(get_moderation_service)],
) -> RagQueryResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="RAG index unavailable")

    # global-аудит, находка №3: /rag/query не проходил ни входную, ни выходную
    # модерацию вообще — тот же ModerationService, что и /chat, /chats/*.
    if moderation is not None:
        mod_result = await moderation.check_input(req.question)
        if not mod_result.allowed:
            raise HTTPException(
                status_code=403,
                detail={"code": "moderation_blocked", "categories": mod_result.categories},
            )

    result = await service.answer(req.question)

    if moderation is not None:
        mod_result = await moderation.check_output(result["answer"])
        if not mod_result.allowed:
            result["answer"] = "Не могу показать ответ — он мог нарушить правила"

    return RagQueryResponse(**result)
