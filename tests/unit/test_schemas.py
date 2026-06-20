"""Unit-тесты для Pydantic-схем: валидация сообщений, PII в repr, расчёт стоимости."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest, Message
from app.schemas.models import AVAILABLE_MODELS, ModelInfo


def test_message_empty_content_raises_validation_error():
    with pytest.raises(ValidationError):
        Message(role="user", content="")


def test_message_too_long_content_raises_validation_error():
    with pytest.raises(ValidationError):
        Message(role="user", content="x" * 32001)


def test_message_repr_masks_email():
    msg = Message(role="user", content="Напишите на ivan@example.com пожалуйста")
    r = repr(msg)
    assert "ivan@example.com" not in r
    assert "[EMAIL]" in r


def test_message_repr_masks_card_number():
    msg = Message(role="user", content="Карта 4111 1111 1111 1111")
    r = repr(msg)
    assert "4111 1111 1111 1111" not in r
    assert "[CARD]" in r


def test_chat_request_empty_messages_raises():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[])


def test_chat_request_temperature_above_max_raises():
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=[Message(role="user", content="hi")],
            temperature=3.0,
        )


def test_chat_request_max_tokens_below_min_raises():
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=[Message(role="user", content="hi")],
            max_tokens=0,
        )


# ── calculate_cost ───────────────────────────────────────────────────────────

def test_calculate_cost_basic():
    model = ModelInfo(
        id="test", name="Test", context_window=4096,
        input_price_per_1k=0.001, output_price_per_1k=0.002,
    )
    cost = model.calculate_cost(prompt_tokens=1000, completion_tokens=500)
    assert abs(cost - (1.0 * 0.001 + 0.5 * 0.002)) < 1e-9


def test_calculate_cost_zero_tokens():
    model = ModelInfo(
        id="test", name="Test", context_window=4096,
        input_price_per_1k=0.005, output_price_per_1k=0.015,
    )
    assert model.calculate_cost(0, 0) == 0.0


def test_calculate_cost_gpt4o_mini_in_available_models():
    mini = next(m for m in AVAILABLE_MODELS if m.id == "gpt-4o-mini")
    cost = mini.calculate_cost(prompt_tokens=10000, completion_tokens=2000)
    expected = 10 * 0.00015 + 2 * 0.0006
    assert abs(cost - expected) < 1e-9
