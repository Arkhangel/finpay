"""Unit-тесты для app/services/agent_naive.py (блок 6.1).

Мокается app.services.agent_naive._client — там, где импортирован (тот же
принцип, что и в tests/unit/test_llm_agent.py).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from app.services.agent_naive import run_agent


def _response(content: str | None = None, tool_calls: list | None = None) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls if tool_calls is not None else []
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 20
    return resp


def _tool_call(name: str, args: dict, call_id: str = "call_1") -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def test_unknown_tool_name_does_not_crash_and_loop_continues(mocker):
    """Критерий самопроверки: неизвестный call.name -> строка-ошибка модели,
    цикл продолжается (не падает с KeyError), возвращается финальный ответ."""
    unknown_call = _response(tool_calls=[_tool_call("get_user_balance", {"user_id": 42})])
    final = _response(content="У меня нет доступа к балансу пользователя.")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [unknown_call, final]
    mocker.patch("app.services.agent_naive._client", mock_client)

    result = run_agent("Проверь баланс пользователя 42")

    assert result["answer"] == "У меня нет доступа к балансу пользователя."
    assert "error" not in result
    trace_entry = result["trace"][0]
    assert trace_entry["tool_name"] == "get_user_balance"
    assert "нет инструмента" in trace_entry["tool_result"]


def test_max_steps_exhausted_returns_error_field(mocker):
    """Если модель зацикливается на tool-вызовах до max_steps, результат
    содержит error и answer=None, а не падает/висит бесконечно."""
    looping = _response(tool_calls=[_tool_call("get_current_time", {})])
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = looping
    mocker.patch("app.services.agent_naive._client", mock_client)

    result = run_agent("Зациклись", max_steps=3)

    assert result["answer"] is None
    assert result["steps"] == 3
    assert "max_steps=3" in result["error"]
    assert len(result["trace"]) == 3
