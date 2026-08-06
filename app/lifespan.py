from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from app.observability.tracing import setup_tracing
from app.services.agent_persistent import agent_lifespan, build_agent
from app.services.rag import RAGService
from app.services.vector_store import VectorStore
from app.settings import settings
from app.settings.logging import setup_logging

logger = logging.getLogger("llm-service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(level="INFO")
    setup_tracing()

    # Canary token injected into every system prompt to detect leakage
    app.state.canary = f"CANARY_{secrets.token_hex(4)}"
    logger.info("canary_token_generated token=%s", app.state.canary)

    app.state.openai = AsyncOpenAI(
        api_key=settings.openai.api_key,
        base_url=settings.openai.host or None,
        timeout=30,
        max_retries=3,
    )

    try:
        app.state.cache = aioredis.from_url(
            settings.redis.url,
            password=settings.redis.password,
            decode_responses=True,
        )
        await app.state.cache.ping()
        logger.info("Redis connected: %s", settings.redis.url)
    except Exception:
        logger.warning("Redis unavailable — cache disabled")
        app.state.cache = None

    if settings.chat.repository == "postgres":
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
        pg_engine = create_async_engine(settings.chat.database_url, echo=False)
        app.state.pg_engine = pg_engine
        app.state.pg_session_factory = async_sessionmaker(pg_engine, expire_on_commit=False)
        logger.info("Postgres engine created: %s", settings.chat.database_url)
    else:
        app.state.pg_engine = None
        app.state.pg_session_factory = None

    app.state.qdrant_client = AsyncQdrantClient(
        url=settings.qdrant.url,
        api_key=settings.qdrant.api_key or None,
    )
    app.state.vector_store = VectorStore(
        app.state.qdrant_client, settings.qdrant.collection, settings.embeddings.dim
    )
    try:
        await app.state.vector_store.ensure_collection()
        logger.info("Qdrant collection ready: %s", settings.qdrant.collection)
    except Exception:
        logger.warning("Qdrant unavailable — vector search disabled")
        app.state.vector_store = None

    # RAG-сервис (Б5.5) подключается к коллекции, наполненной
    # scripts/ingest.py — сама индексация здесь не выполняется.
    rag_service = RAGService()
    try:
        await asyncio.to_thread(rag_service.build)
        app.state.rag_service = rag_service
        logger.info("RAG index ready: %s", settings.rag.kb_collection)
    except Exception:
        logger.warning("RAG index unavailable — /rag/query disabled")
        app.state.rag_service = None

    # Блок 6.4: checkpointer.setup() вызывается ОДИН раз здесь (внутри
    # agent_lifespan()), не на каждый запрос — см. docs/agent-persistent-report.md.
    async with agent_lifespan() as checkpointer:
        app.state.agent_graph = build_agent(checkpointer)
        logger.info("Persistent agent graph ready: checkpointer=%s", settings.agent.checkpointer)

        yield

    await app.state.openai.close()
    if app.state.cache:
        await app.state.cache.aclose()
    if app.state.pg_engine:
        await app.state.pg_engine.dispose()
    if app.state.rag_service:
        await app.state.rag_service.aclose()
    await app.state.qdrant_client.close()
