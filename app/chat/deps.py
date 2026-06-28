from __future__ import annotations

from typing import Annotated, AsyncGenerator

from fastapi import Depends, Request
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.repository import ChatRepository
from app.chat.service import ChatService
from app.settings import settings


def get_openai_client(request: Request) -> AsyncOpenAI:
    return request.app.state.openai


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
) -> ChatService:
    return ChatService(repository=repo, llm_client=llm)


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
