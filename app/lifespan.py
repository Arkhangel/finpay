from __future__ import annotations

import logging
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

    yield

    await app.state.openai.close()
    if app.state.cache:
        await app.state.cache.aclose()
