"""DI-провайдеры для FastAPI.

Клиенты достаются из app.state, собранного в lifespan.
Никаких глобальных переменных-клиентов на уровне модуля.
"""

from __future__ import annotations

from typing import Annotated

import redis.asyncio as aioredis
from fastapi import Depends, Request
from langgraph.graph.state import CompiledStateGraph
from openai import AsyncOpenAI

from app.services.llm import LLMService
from app.services.rag import RAGService
from app.services.vector_store import VectorStore
from app.settings import settings


def get_openai(request: Request) -> AsyncOpenAI:
    return request.app.state.openai


def get_cache(request: Request) -> aioredis.Redis | None:
    return request.app.state.cache


def get_vector_store(request: Request) -> VectorStore | None:
    return request.app.state.vector_store


def get_rag_service(request: Request) -> RAGService | None:
    return request.app.state.rag_service


def get_agent_graph(request: Request) -> CompiledStateGraph | None:
    return getattr(request.app.state, "agent_graph", None)


def get_canary(request: Request) -> str:
    return getattr(request.app.state, "canary", "")


def get_llm_service(
    openai: Annotated[AsyncOpenAI, Depends(get_openai)],
    cache: Annotated[aioredis.Redis | None, Depends(get_cache)],
) -> LLMService:
    return LLMService(openai, cache, settings)


LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
CacheDep = Annotated[aioredis.Redis | None, Depends(get_cache)]
CanaryDep = Annotated[str, Depends(get_canary)]
VectorStoreDep = Annotated[VectorStore | None, Depends(get_vector_store)]
RAGServiceDep = Annotated[RAGService | None, Depends(get_rag_service)]
AgentGraphDep = Annotated[CompiledStateGraph | None, Depends(get_agent_graph)]
