"""LLMService — сервисный слой для FastAPI (блок 3.4).

complete: запрос с Redis-кешем (ключ = sha256 тела запроса).
stream:   async-генератор дельт, без кеша.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI

from app.core.exceptions import LLMAuthError, LLMRateLimitError, LLMTimeoutError
from app.schemas.chat import ChatDelta, ChatRequest, ChatResponse, Usage
from app.settings import Settings

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self, client: AsyncOpenAI, cache, settings: Settings) -> None:
        self._client = client
        self._cache = cache
        self._settings = settings

    # ── кеш ──────────────────────────────────────────────────────────────

    def _cache_key(self, req: ChatRequest) -> str:
        data = req.model_dump(exclude={"user_id", "session_id"})
        digest = hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()
        return f"chat:{digest}"

    # ── публичные методы ─────────────────────────────────────────────────

    async def complete(self, req: ChatRequest) -> ChatResponse:
        key = self._cache_key(req)

        if self._cache is not None:
            try:
                cached = await self._cache.get(key)
                if cached:
                    resp = ChatResponse.model_validate_json(cached)
                    return resp.model_copy(update={"cached": True})
            except Exception:
                logger.warning("Redis get failed, skipping cache")

        model = req.model or self._settings.openai.model
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[m.model_dump() for m in req.messages],
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except openai.APITimeoutError as e:
            raise LLMTimeoutError(str(e)) from e
        except openai.AuthenticationError as e:
            raise LLMAuthError(str(e)) from e

        result = ChatResponse.from_openai(response)

        if self._cache is not None:
            try:
                await self._cache.setex(
                    key, self._settings.redis.ttl, result.model_dump_json()
                )
            except Exception:
                logger.warning("Redis setex failed, skipping cache write")

        return result

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]:
        model = req.model or self._settings.openai.model
        try:
            openai_stream = await self._client.chat.completions.create(
                model=model,
                messages=[m.model_dump() for m in req.messages],
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except openai.APITimeoutError as e:
            raise LLMTimeoutError(str(e)) from e
        except openai.AuthenticationError as e:
            raise LLMAuthError(str(e)) from e

        async for chunk in openai_stream:
            if not chunk.choices:
                if chunk.usage:
                    yield ChatDelta(
                        usage=Usage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens,
                        )
                    )
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield ChatDelta(content=delta)
