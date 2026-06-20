"""Unit-тесты для app/llm/client.py — синхронный агент с tool calling.

Мокается app.llm.client.client — там, где импортирован (правило: патчить
по месту импорта, а не по месту определения).
"""
from __future__ import annotations

import json

import pytest
from unittest.mock import MagicMock, call

from app.llm.client import run


def _response(content: str | None = None, tool_calls: list | None = None) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls if tool_calls is not None else []
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    resp.usage.total_tokens = 10
    return resp


def _tool_call(name: str, args: dict, call_id: str = "call_1") -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def test_run_no_tool_call_returns_content(mocker):
    """Если модель не вызвала tool, run() возвращает content без второго запроса."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _response(content="Комиссия 1.8%.")
    mocker.patch("app.llm.client.client", mock_client)
    mocker.patch("app.llm.client.render_system_prompt", return_value="sys")

    result = run("Какая комиссия?")

    assert result == "Комиссия 1.8%."
    assert mock_client.chat.completions.create.call_count == 1


def test_run_with_tool_call_makes_two_requests(mocker):
    """При tool call делается два запроса к модели."""
    tc = _tool_call("get_payment_system_status", {"component": "api"})
    first = _response(tool_calls=[tc])
    second = _response(content="API работает штатно.")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [first, second]
    mocker.patch("app.llm.client.client", mock_client)
    mocker.patch("app.llm.client.render_system_prompt", return_value="sys")

    result = run("Статус API?")

    assert result == "API работает штатно."
    assert mock_client.chat.completions.create.call_count == 2


def test_run_tool_result_added_to_messages(mocker):
    """Результат tool добавляется в messages со role='tool' перед вторым запросом."""
    tc = _tool_call("get_payment_system_status", {"component": "api"}, "call_xyz")
    first = _response(tool_calls=[tc])
    second = _response(content="Ок.")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [first, second]
    mocker.patch("app.llm.client.client", mock_client)
    mocker.patch("app.llm.client.render_system_prompt", return_value="sys")

    run("Статус API?")

    second_call_messages = mock_client.chat.completions.create.call_args_list[1][1]["messages"]
    tool_msgs = [m for m in second_call_messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_xyz"


def test_run_unknown_tool_puts_error_in_messages(mocker):
    """Неизвестный tool → сообщение об ошибке добавляется в messages."""
    tc = _tool_call("nonexistent_tool", {}, "call_bad")
    first = _response(tool_calls=[tc])
    second = _response(content="Не могу помочь.")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [first, second]
    mocker.patch("app.llm.client.client", mock_client)
    mocker.patch("app.llm.client.render_system_prompt", return_value="sys")

    run("что-то")

    second_call_messages = mock_client.chat.completions.create.call_args_list[1][1]["messages"]
    tool_msgs = [m for m in second_call_messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    payload = json.loads(tool_msgs[0]["content"])
    assert "error" in payload


def test_run_first_request_includes_system_and_user_messages(mocker):
    """Первый запрос содержит system и user сообщения в правильном порядке."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _response(content="Ответ.")
    mocker.patch("app.llm.client.client", mock_client)
    mocker.patch("app.llm.client.render_system_prompt", return_value="SYSTEM")

    run("Вопрос пользователя")

    messages = mock_client.chat.completions.create.call_args[1]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "SYSTEM"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Вопрос пользователя"
