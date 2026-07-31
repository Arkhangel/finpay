"""Unit-тесты для app/services/agent_react.py (блок 6.2): жёсткие лимиты и
self-reflection.

Мокается app.services.agent_react._client — там, где импортирован (тот же
принцип, что и в tests/unit/test_agent_naive.py).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APITimeoutError

from app.services.agent_react import run_react_agent


def _response(content: str | None = None, tool_calls: list | None = None,
               prompt: int = 100, completion: int = 20) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls if tool_calls is not None else []
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    resp.usage.prompt_tokens = prompt
    resp.usage.completion_tokens = completion
    resp.usage.total_tokens = prompt + completion
    return resp


def _tool_call(name: str, args: dict, call_id: str = "call_1") -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def _critic_verdict(text: str) -> MagicMock:
    return _response(content=text, prompt=30, completion=10)


def test_max_iterations_exhausted_returns_explicit_error_not_hang(mocker):
    """Критерий самопроверки: agent_react.py падает с явным сообщением при
    max_iterations, а не висит."""
    looping = _response(tool_calls=[_tool_call("get_current_time", {"timezone": "Europe/Moscow"})])
    ok_verdict = _critic_verdict("OK")
    mock_client = MagicMock()
    # 8 итераций, на каждой main + critic(OK, лимит ревизий не задет)
    mock_client.chat.completions.create.side_effect = [looping, ok_verdict] * 8
    mocker.patch("app.services.agent_react._client", mock_client)

    result = run_react_agent("Зациклись", max_iterations=8, timeout_per_iteration_sec=5.0)

    assert result["answer"] is None
    assert "Превышен лимит итераций" in result["error"]
    assert "max_iterations=8" in result["error"]
    assert len(result["trace"]) == 8


def test_revision_counter_caps_and_never_resets_between_iterations(mocker):
    """Self-reflection срабатывает не больше max_revisions раз за запуск;
    после исчерпания лимита критик больше не вызывается (счётчик не
    сбрасывается между итерациями)."""
    looping = _response(tool_calls=[_tool_call("get_current_time", {"timezone": "Europe/Moscow"})])
    revise = _critic_verdict("REVISE: наблюдение не по делу")
    mock_client = MagicMock()
    # step0: main, critic(REVISE #1) | step1: main, critic(REVISE #2) |
    # step2..7 (6 итераций): критик больше не вызывается — лимит (2) исчерпан
    mock_client.chat.completions.create.side_effect = (
        [looping, revise, looping, revise] + [looping] * 6
    )
    mocker.patch("app.services.agent_react._client", mock_client)

    result = run_react_agent("Зациклись", max_iterations=8, timeout_per_iteration_sec=5.0,
                              max_revisions=2)

    assert result["revisions_used"] == 2
    assert mock_client.chat.completions.create.call_count == 10


def test_timeout_returns_explicit_message_not_exception(mocker):
    """Критерий самопроверки: agent_react.py падает с явным сообщением при
    timeout, а не пробрасывает исключение и не висит."""
    mock_client = MagicMock()
    dummy_request = httpx.Request("POST", "http://test")
    mock_client.chat.completions.create.side_effect = APITimeoutError(dummy_request)
    mocker.patch("app.services.agent_react._client", mock_client)

    result = run_react_agent("Задача", max_iterations=8, timeout_per_iteration_sec=5.0)

    assert result["answer"] is None
    assert result["error"] == "Timeout"


def test_empty_content_without_tool_call_does_not_finish_silently(mocker):
    """Найдено эмпирически: модель иногда отдаёт content="" без tool_calls.
    Такой ответ не должен молча завершать цикл с answer="" — агент должен
    переспросить и получить реальный финальный ответ."""
    empty = _response(content="")
    final = _response(content="Реальный ответ.")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [empty, final]
    mocker.patch("app.services.agent_react._client", mock_client)

    result = run_react_agent("Задача", max_iterations=8, timeout_per_iteration_sec=5.0)

    assert result["answer"] == "Реальный ответ."
    assert result["steps"] == 2


def test_out_of_range_limits_rejected():
    with pytest.raises(ValueError):
        run_react_agent("x", max_iterations=30)
    with pytest.raises(ValueError):
        run_react_agent("x", timeout_per_iteration_sec=1.0)
