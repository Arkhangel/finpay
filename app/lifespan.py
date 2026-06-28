from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from openai import AsyncOpenAI

from app.observability.tracing import setup_tracing
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

    yield

    await app.state.openai.close()
    if app.state.cache:
        await app.state.cache.aclose()
    if app.state.pg_engine:
        await app.state.pg_engine.dispose()
