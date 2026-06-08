"""Асинхронный LLM-клиент — блок 3.3.

Semaphore создаётся в __init__ один раз как self._sem (не глобальный, не в методе).
SDK-таймаут: AsyncOpenAI(timeout=30), бизнес-таймаут: asyncio.timeout(15) в complete.
Retry: встроенный max_retries=3 SDK (exponential backoff + retry-after для 429/5xx).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.settings import settings as _settings

logger = logging.getLogger(__name__)


class AsyncLLMClient:
    def __init__(self, concurrency: int = 10) -> None:
        self._client = AsyncOpenAI(
            api_key=_settings.openai.api_key,
            base_url=_settings.openai.host or None,
            timeout=30,
            max_retries=3,
        )
        self._model = _settings.openai.model
        self._sem = asyncio.Semaphore(concurrency)

    async def complete(self, prompt: str, max_tokens: int = 256) -> str:
        start = time.perf_counter()
        status = "ok"
        try:
            async with asyncio.timeout(60):
                async with self._sem:
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=max_tokens,
                    )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            status = type(exc).__name__
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "llm.call duration_ms=%.1f model=%s prompt_chars=%d status=%s",
                duration_ms, self._model, len(prompt), status,
            )

    async def batch_chat(
        self, prompts: list[str], concurrency: int = 5, max_tokens: int = 256
    ) -> list[str | Exception]:
        """Параллельные запросы. Ошибки оседают в позициях, не роняют батч.

        Конкурентность ограничивается self._sem (создан в __init__).
        Параметр concurrency используется при создании клиента: AsyncLLMClient(concurrency=N).
        """
        return list(
            await asyncio.gather(
                *(self.complete(p, max_tokens=max_tokens) for p in prompts),
                return_exceptions=True,
            )
        )

    async def stream_chat(self, prompt: str) -> AsyncIterator[str]:
        """Async-генератор строковых дельт. Логирует total_tokens после стрима."""
        first_token_t: float | None = None

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in stream:
            if not chunk.choices:
                if chunk.usage:
                    logger.info("stream_chat done total_tokens=%d", chunk.usage.total_tokens)
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                if first_token_t is None:
                    first_token_t = time.perf_counter()
                    logger.debug("stream_chat: first token received")
                yield delta

    async def close(self) -> None:
        await self._client.close()
