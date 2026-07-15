from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.deps.providers import RAGServiceDep
from app.schemas.rag import RagQueryRequest, RagQueryResponse

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post(
    "/query",
    response_model=RagQueryResponse,
    summary="RAG-ответ по базе знаний (блок 5.3)",
    responses={200: {}, 422: {}, 503: {}},
)
async def rag_query(req: RagQueryRequest, service: RAGServiceDep) -> RagQueryResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="RAG index unavailable")

    result = await service.answer(req.question)
    return RagQueryResponse(**result)
