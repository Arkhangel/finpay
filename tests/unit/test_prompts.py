"""Unit-тесты для app/prompts/loader.py.

render_system_prompt имеет @lru_cache — сбрасываем кеш в фикстуре,
чтобы тесты не влияли друг на друга.
"""
from __future__ import annotations

import pytest

from app.prompts.loader import load_tool_description, render_system_prompt


@pytest.fixture(autouse=True)
def clear_prompt_cache():
    render_system_prompt.cache_clear()
    yield
    render_system_prompt.cache_clear()


def test_render_system_prompt_contains_project_name():
    result = render_system_prompt(project_name="TestProject")
    assert "TestProject" in result


def test_render_system_prompt_is_non_empty_string():
    result = render_system_prompt(project_name="FinPay")
    assert isinstance(result, str)
    assert len(result) > 100


def test_render_system_prompt_literal_braces_preserved():
    """{service_name} в шаблоне — одинарные скобки, Jinja2 не трогает их.
    Если кто-то случайно вызовет .format() — получит KeyError.
    """
    result = render_system_prompt(project_name="FinPay")
    assert "{service_name}" in result


def test_render_system_prompt_not_safe_for_format_string():
    """Гарантируем, что .format() на промпте взрывается — proof of injection risk."""
    result = render_system_prompt(project_name="FinPay")
    with pytest.raises(KeyError):
        result.format()


def test_load_tool_description_returns_non_empty():
    for tool in ("check_transaction_status", "get_payment_system_status"):
        desc = load_tool_description(tool)
        assert isinstance(desc, str)
        assert len(desc) > 10
