"""Unit-тесты для app/moderation/service.py: keyword-слой и OpenAI-слой."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.moderation.service import ModerationService
from app.settings import settings


@pytest.fixture
def moderation() -> ModerationService:
    return ModerationService(keywords_path=settings.moderation.keywords_path)


async def test_check_input_clean_text_allowed(moderation):
    result = await moderation.check_input("Какой статус у транзакции TXN-1001?")
    assert result.allowed is True
    assert result.categories == []


async def test_check_input_keyword_blocked(moderation):
    result = await moderation.check_input("Продам дамп карты, недорого")
    assert result.allowed is False
    assert result.blocked_by == "keyword"
    assert "fraud" in result.categories


async def test_check_output_keyword_blocked(moderation):
    result = await moderation.check_output("Вот как обналичить карты через дроп-схему: обнал карт")
    assert result.allowed is False
    assert "fraud" in result.categories


async def test_check_input_case_insensitive(moderation):
    result = await moderation.check_input("ДАМП КАРТЫ срочно")
    assert result.allowed is False


async def test_openai_layer_flags_above_threshold():
    client = MagicMock()
    category_scores = MagicMock()
    category_scores.model_dump.return_value = {"violence": 0.9, "hate": 0.1}
    client.moderations.create = AsyncMock(
        return_value=MagicMock(results=[MagicMock(category_scores=category_scores)])
    )

    moderation = ModerationService(
        keywords_path=settings.moderation.keywords_path,
        use_openai_api=True,
        openai_client=client,
    )

    result = await moderation.check_input("что-то нейтральное на вид")

    assert result.allowed is False
    assert result.blocked_by == "openai_moderation"
    assert result.categories == ["violence"]


async def test_openai_layer_not_called_when_disabled():
    client = MagicMock()
    client.moderations.create = AsyncMock()

    moderation = ModerationService(keywords_path=settings.moderation.keywords_path, use_openai_api=False, openai_client=client)
    result = await moderation.check_input("нейтральный текст")

    assert result.allowed is True
    client.moderations.create.assert_not_called()


async def test_openai_layer_outage_fails_open():
    client = MagicMock()
    client.moderations.create = AsyncMock(side_effect=RuntimeError("timeout"))

    moderation = ModerationService(
        keywords_path=settings.moderation.keywords_path,
        use_openai_api=True,
        openai_client=client,
    )

    result = await moderation.check_input("нейтральный текст")

    assert result.allowed is True
