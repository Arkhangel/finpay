import pytest

from app.observability.pii import redact_pii


@pytest.mark.parametrize("pii,placeholder", [
    ("ivan@mail.ru", "[EMAIL]"),
    ("user.name+tag@example.co.uk", "[EMAIL]"),
])
def test_email_redacted(pii, placeholder):
    result = redact_pii(f"Напишите на {pii} пожалуйста")
    assert pii not in result
    assert placeholder in result


@pytest.mark.parametrize("pii", [
    "+7 (999) 123-45-67",
    "+7 999 123 45 67",
    "8 999 123-45-67",
])
def test_phone_redacted(pii):
    result = redact_pii(f"Позвоните {pii}")
    assert "[PHONE_RU]" in result


def test_card_redacted():
    result = redact_pii("карта 4111 1111 1111 1111")
    assert "4111 1111 1111 1111" not in result
    assert "[CARD]" in result


def test_combined():
    text = "Мой email ivan@mail.ru, тел +7 (999) 123-45-67, карта 4111 1111 1111 1111"
    result = redact_pii(text)
    assert "ivan@mail.ru" not in result
    assert "123-45-67" not in result
    assert "4111 1111 1111 1111" not in result
    assert "[EMAIL]" in result
    assert "[PHONE_RU]" in result
    assert "[CARD]" in result


def test_no_pii_unchanged():
    text = "Статус транзакции TXN-1001?"
    assert redact_pii(text) == text


def test_fails_without_masking():
    """Тест падает если убрать маскирование — доказывает что тест рабочий."""
    text = "email: ivan@mail.ru"
    result = redact_pii(text)
    assert "ivan@mail.ru" not in result
