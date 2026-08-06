"""Unit-тесты для app/services/agent_graph.py (блок 6.3): stop-кран
(force_finish на max_iterations) и router-логика.

Мокается app.services.agent_graph.model_with_tools.ainvoke — там, где
call_model его вызывает (тот же принцип патчинга в точке импорта/
использования, что в tests/unit/test_agent_react.py).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage

from app.services.agent_graph import (
    MAX_ITERATIONS,
    AgentState,
    custom_graph,
    force_finish,
    route_after_model,
)


def _tool_call_message(name: str = "get_current_time", args: dict | None = None,
                        call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args or {"timezone": "НЕ_РЕАЛЬНЫЙ_ПОЯС"}, "id": call_id},
    ])


async def test_stop_crank_reaches_force_finish_with_broken_tool(mocker):
    """Критерий самопроверки: iteration_count >= 6 уводит в force_finish, а
    не в бесконечный цикл — даже если модель бесконечно зовёт заведомо
    бесполезный (ломаный) tool (get_current_time с некорректным поясом)."""
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(side_effect=lambda *_a, **_k: _tool_call_message())
    mocker.patch("app.services.agent_graph.model_with_tools", mock_model)

    result = await custom_graph.ainvoke({
        "messages": [{"role": "user", "content": "Зациклись на ломаном tool"}],
        "iteration_count": 0,
        "tool_results": [],
    })

    assert result["iteration_count"] == MAX_ITERATIONS
    assert "Превышен лимит итераций" in result["messages"][-1].content
    # execute_tool (реальный, не мокнутый) выполняется МЕЖДУ вызовами
    # call_model — на MAX_ITERATIONS-ю итерацию роутер уже отправляет в
    # force_finish, не дав execute_tool отработать последний tool_call.
    # Итого execute_tool реально выполнился MAX_ITERATIONS-1 раз, каждый —
    # с заведомо бесполезным результатом (некорректный часовой пояс).
    assert len(result["tool_results"]) == MAX_ITERATIONS - 1
    assert all("Ошибка" in r["result"] for r in result["tool_results"])


def test_router_is_pure_and_typed():
    """Router — отдельная функция, читает state, не пишет, без сетевых
    вызовов, типизирован через Literal."""
    state_final: AgentState = {
        "messages": [AIMessage(content="готовый ответ")],
        "iteration_count": 1,
        "tool_results": [],
    }
    assert route_after_model(state_final) == "force_finish"
    # исходный state не мутирован вызовом router
    assert state_final["iteration_count"] == 1

    state_tool: AgentState = {
        "messages": [_tool_call_message()],
        "iteration_count": 1,
        "tool_results": [],
    }
    assert route_after_model(state_tool) == "execute_tool"

    state_over_limit: AgentState = {
        "messages": [_tool_call_message()],
        "iteration_count": MAX_ITERATIONS,
        "tool_results": [],
    }
    assert route_after_model(state_over_limit) == "force_finish"


async def test_force_finish_passes_through_real_final_answer_unchanged():
    """Если модель уже дала настоящий финальный ответ (нет tool_calls),
    force_finish не подменяет его — просто прокидывает {}."""
    state: AgentState = {
        "messages": [AIMessage(content="Настоящий финальный ответ.")],
        "iteration_count": 2,
        "tool_results": [],
    }
    delta = await force_finish(state)
    assert delta == {}
