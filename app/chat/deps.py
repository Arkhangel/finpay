from __future__ import annotations

from typing import Annotated, AsyncGenerator

from fastapi import Depends, Request
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.repository import ChatRepository
from app.chat.service import ChatService
from app.deps.providers import get_rag_service
from app.moderation import ModerationService
from app.services.rag import RAGService
from app.settings import settings


def get_openai_client(request: Request) -> AsyncOpenAI:
    return request.app.state.openai


def get_moderation_openai_client(request: Request) -> AsyncOpenAI | None:
    # НЕ get_openai_client — тот указывает на Groq (settings.openai.host),
    # а Moderation API — проприетарный эндпоинт настоящего OpenAI, которого
    # на Groq нет (global-аудит, находка №3; см. app/settings/moderation.py).
    return getattr(request.app.state, "moderation_openai_client", None)


def get_moderation_service(
    moderation_client: Annotated[AsyncOpenAI | None, Depends(get_moderation_openai_client)],
) -> ModerationService | None:
    if not settings.moderation.enabled:
        return None
    return ModerationService(
        keywords_path=settings.moderation.keywords_path,
        use_openai_api=settings.moderation.use_openai_api,
        openai_client=moderation_client,
        category_thresholds=settings.moderation.category_thresholds,
    )


async def get_pg_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    factory = getattr(request.app.state, "pg_session_factory", None)
    if factory is None:
        yield None
        return
    async with factory() as session:
        yield session


def get_repository(
    session: Annotated[AsyncSession | None, Depends(get_pg_session)],
) -> ChatRepository:
    repo_type = settings.chat.repository
    if repo_type == "json":
        from app.chat.repositories.json_repo import JsonChatRepository
        return JsonChatRepository(base_dir=settings.chat.storage_dir)
    if repo_type == "postgres":
        if session is None:
            raise ValueError("Postgres session is not available (engine not initialized in lifespan)")
        from app.chat.repositories.pg_repo import PostgresChatRepository
        return PostgresChatRepository(session=session)
    raise ValueError(
        f"Unknown CHAT__REPOSITORY value: {repo_type!r}. Expected 'json' or 'postgres'."
    )


def get_chat_service(
    repo: Annotated[ChatRepository, Depends(get_repository)],
    llm: Annotated[AsyncOpenAI, Depends(get_openai_client)],
    moderation: Annotated[ModerationService | None, Depends(get_moderation_service)],
    rag: Annotated[RAGService | None, Depends(get_rag_service)],
) -> ChatService:
    return ChatService(repository=repo, llm_client=llm, moderation=moderation, rag_service=rag)


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
ChatRepositoryDep = Annotated[ChatRepository, Depends(get_repository)]
OpenAIClientDep = Annotated[AsyncOpenAI, Depends(get_openai_client)]
