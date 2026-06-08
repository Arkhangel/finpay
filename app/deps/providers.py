"""DI-провайдеры для FastAPI.

Клиенты достаются из app.state, собранного в lifespan.
Никаких глобальных переменных-клиентов на уровне модуля.
"""

from __future__ import annotations

from typing import Annotated

import redis.asyncio as aioredis
from fastapi import Depends, Request
from openai import AsyncOpenAI

from app.services.llm import LLMService
from app.settings import settings


def get_openai(request: Request) -> AsyncOpenAI:
    return request.app.state.openai


def get_cache(request: Request) -> aioredis.Redis | None:
    return request.app.state.cache


def get_llm_service(
    openai: Annotated[AsyncOpenAI, Depends(get_openai)],
    cache: Annotated[aioredis.Redis | None, Depends(get_cache)],
) -> LLMService:
    return LLMService(openai, cache, settings)


LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
CacheDep = Annotated[aioredis.Redis | None, Depends(get_cache)]
