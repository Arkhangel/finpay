"""Бенчмарк sync vs async LLM-клиент — блок 3.3.

Запуск из корня проекта:
    ENVIRONMENT=local uv run scripts/benchmark.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from app.services.llm_client import AsyncLLMClient
from app.settings import settings

logging.basicConfig(level=logging.WARNING)

MAX_TOKENS = 64  # одинаково для sync и async

CONCEPTS = [
    "event loop в asyncio",
    "semaphore как примитив синхронизации",
    "exponential backoff при retry",
    "connection pooling в HTTP-клиентах",
    "cache-aside паттерн",
    "circuit breaker паттерн",
    "rate limiting алгоритм token bucket",
    "SSE (Server-Sent Events)",
    "dependency injection в FastAPI",
    "pydantic-settings и SecretStr",
    "asyncio.gather с return_exceptions",
    "asyncio.timeout vs asyncio.wait_for",
    "CORS и preflight-запросы",
    "JWT-токены и их структура",
    "HTTP/2 мультиплексирование",
    "streaming response в HTTP",
    "LRU-кеш и его применение",
    "health check эндпоинт",
    "middleware в ASGI-приложениях",
    "request_id для трассировки запросов",
]
PROMPTS = [f"Объясни одним абзацем концепцию: {c}" for c in CONCEPTS]


def run_sync() -> float:
    client = OpenAI(
        api_key=settings.openai.api_key,
        base_url=settings.openai.host or None,
    )
    start = time.perf_counter()
    for prompt in PROMPTS:
        client.chat.completions.create(
            model=settings.openai.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
        )
    return time.perf_counter() - start


async def run_async(concurrency: int) -> float:
    client = AsyncLLMClient(concurrency=concurrency)
    start = time.perf_counter()
    results = await client.batch_chat(PROMPTS, max_tokens=MAX_TOKENS)
    elapsed = time.perf_counter() - start
    await client.close()
    errors = sum(1 for r in results if isinstance(r, Exception))
    if errors:
        print(f"   ⚠️  {errors}/{len(PROMPTS)} ошибок (вероятно rate limit)")
    return elapsed


def main() -> None:
    print(f"\n{'=' * 60}")
    print(f"  Бенчмарк: {len(PROMPTS)} промптов | модель: {settings.openai.model}")
    print(f"  max_tokens={MAX_TOKENS} (одинаково для sync и async)")
    print(f"{'=' * 60}\n")

    print("⏳ Sync (последовательно)...")
    sync_time = run_sync()
    print(f"   sync:              {sync_time:6.2f}s\n")

    results: dict[int, float] = {}
    for c in [1, 5, 10]:
        print(f"⏳ Async concurrency={c}...")
        t = asyncio.run(run_async(c))
        results[c] = t
        speedup = sync_time / t
        print(f"   async c={c:2d}:         {t:6.2f}s  (speedup {speedup:.1f}x)\n")

    print(f"{'=' * 60}")
    print("Итог:")
    print(f"  sync:          {sync_time:.2f}s  (1.0x)")
    for c, t in results.items():
        print(f"  async c={c:2d}:     {t:.2f}s  ({sync_time / t:.1f}x)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
