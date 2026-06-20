"""Unit-тесты для LLMService: кеш, порядок ролей, обработка ошибок.

Все тесты работают без сетевых вызовов и API-ключей.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

import openai

from app.core.exceptions import LLMRateLimitError
from app.schemas.chat import ChatRequest, ChatResponse, Message, Usage
from app.services.llm import LLMService


@pytest.fixture
def mock_settings():
    s = MagicMock()
    s.openai.model = "gpt-4o-mini"
    s.redis.ttl = 300
    return s


@pytest.fixture
def openai_response():
    choice = MagicMock()
    choice.message.content = "Транзакция успешно завершена."
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20
    usage.total_tokens = 30
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    resp.model = "gpt-4o-mini"
    return resp


async def test_cache_hit_skips_llm_call(mock_settings, openai_response):
    """Кеш-хит: OpenAI не вызывается, в ответе cached=True."""
    stored = ChatResponse(
        content="cached answer",
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        finish_reason="stop",
    )
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=stored.model_dump_json())

    client = AsyncMock()
    svc = LLMService(client, cache, mock_settings)
    req = ChatRequest(messages=[Message(role="user", content="Привет")])

    result = await svc.complete(req)

    assert result.cached is True
    assert result.content == "cached answer"
    client.chat.completions.create.assert_not_called()


async def test_cache_miss_calls_llm(mock_settings, openai_response):
    """Кеш-мисс: OpenAI вызывается, cached=False."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.setex = AsyncMock()

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=openai_response)

    svc = LLMService(client, cache, mock_settings)
    req = ChatRequest(messages=[Message(role="user", content="Привет")])

    result = await svc.complete(req)

    assert result.cached is False
    client.chat.completions.create.assert_called_once()


async def test_successful_response_stored_in_cache(mock_settings, openai_response):
    """После LLM-ответа результат сохраняется в Redis с нужным ключом."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.setex = AsyncMock()

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=openai_response)

    svc = LLMService(client, cache, mock_settings)
    req = ChatRequest(messages=[Message(role="user", content="Статус транзакции?")])

    await svc.complete(req)

    cache.setex.assert_called_once()
    key_arg = cache.setex.call_args[0][0]
    assert key_arg.startswith("chat:")


async def test_rate_limit_mapped_to_custom_exception(mock_settings):
    """openai.RateLimitError → LLMRateLimitError (кастомное исключение)."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        side_effect=openai.RateLimitError("rate limit", response=MagicMock(), body={})
    )

    svc = LLMService(client, cache, mock_settings)
    req = ChatRequest(messages=[Message(role="user", content="Тест")])

    with pytest.raises(LLMRateLimitError):
        await svc.complete(req)


async def test_message_role_order_preserved(mock_settings, openai_response):
    """system → user — порядок сообщений сохраняется при вызове OpenAI API."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.setex = AsyncMock()

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=openai_response)

    svc = LLMService(client, cache, mock_settings)
    req = ChatRequest(messages=[
        Message(role="system", content="Ты ассистент поддержки FinPay"),
        Message(role="user", content="Какой статус у транзакции TXN-1001?"),
    ])

    await svc.complete(req)

    call_kwargs = client.chat.completions.create.call_args[1]
    msgs = call_kwargs["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


async def test_curly_braces_in_user_message_not_interpolated(mock_settings, openai_response):
    """Фигурные скобки в сообщении пользователя не вызывают KeyError/ValueError."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.setex = AsyncMock()

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=openai_response)

    svc = LLMService(client, cache, mock_settings)
    req = ChatRequest(messages=[
        Message(role="user", content="Что такое {service_name} и {api_key}?"),
    ])

    await svc.complete(req)

    call_kwargs = client.chat.completions.create.call_args[1]
    content = call_kwargs["messages"][0]["content"]
    assert "{service_name}" in content
    assert "{api_key}" in content
