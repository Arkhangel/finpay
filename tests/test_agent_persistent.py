"""Smoke-тесты для app/services/agent_persistent.py (блок 6.4): HIL через
interrupt()+Command(resume=...) на AsyncSqliteSaver(":memory:") — без
Postgres, без реального LLM (модель мокается) и без реального Telegram API
(aiogram.Bot мокается).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.services.agent_persistent import SYSTEM_PROMPT, build_agent

TASK = "Отправь клиенту в чат 555 сообщение 'привет'."


def _tool_call_message() -> AIMessage:
    return AIMessage(content="", tool_calls=[{
        "name": "send_telegram_message",
        "args": {"chat_id": "555", "text": "привет"},
        "id": "call_1",
    }])


def _initial_state() -> dict:
    return {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=TASK)],
        "iteration_count": 0, "tool_results": [], "pending_action": None,
    }


@pytest.fixture
def mock_model(mocker):
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=_tool_call_message())
    mocker.patch("app.services.agent_persistent.model_with_tools", mock)
    return mock


@pytest.fixture
def mock_bot(mocker):
    bot_instance = MagicMock()
    bot_instance.send_message = AsyncMock()
    bot_instance.session.close = AsyncMock()
    mock_bot_cls = MagicMock(return_value=bot_instance)
    mocker.patch("app.services.agent_persistent.Bot", mock_bot_cls)
    return bot_instance


async def test_graph_reaches_interrupt_with_approve_node_pending(mock_model, mock_bot):
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        graph = build_agent(checkpointer)
        config = {"configurable": {"thread_id": "t-interrupt"}}
        result = await graph.ainvoke(_initial_state(), config=config)

        assert "__interrupt__" in result
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("confirm_and_execute_send_telegram",)
        mock_bot.send_message.assert_not_called()


async def test_resume_true_sends_real_message(mock_model, mock_bot):
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        graph = build_agent(checkpointer)
        config = {"configurable": {"thread_id": "t-approve"}}
        await graph.ainvoke(_initial_state(), config=config)

        result = await graph.ainvoke(Command(resume=True), config=config)

        mock_bot.send_message.assert_called_once_with(chat_id=555, text="привет")
        sent_result = result["tool_results"][-1]
        assert sent_result["name"] == "send_telegram_message"
        assert "отправлено" in sent_result["result"]


async def test_resume_false_skips_side_effect(mock_model, mock_bot):
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        graph = build_agent(checkpointer)
        config = {"configurable": {"thread_id": "t-deny"}}
        await graph.ainvoke(_initial_state(), config=config)

        result = await graph.ainvoke(Command(resume=False), config=config)

        mock_bot.send_message.assert_not_called()
        sent_result = result["tool_results"][-1]
        assert sent_result["result"] == "Отклонено пользователем — сообщение не отправлено."


async def test_user_role_full_skips_interrupt_entirely(mock_model, mock_bot):
    """Permission policy: роль full пропускает interrupt (см. отчёт,
    раздел про policy) — граф не должен приостанавливаться вовсе."""
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        graph = build_agent(checkpointer)
        config = {"configurable": {"thread_id": "t-full", "user_role": "full"}}
        result = await graph.ainvoke(_initial_state(), config=config)

        assert "__interrupt__" not in result
        mock_bot.send_message.assert_called_once_with(chat_id=555, text="привет")


async def test_two_threads_same_input_diverge_approve_vs_deny(mock_model, mock_bot):
    """Time-travel-нюанс из ТЗ: одна и та же задача на ДВУХ разных
    thread_id может дать разные исходы — resume детерминирован по
    thread-lineage, а не по значению, поэтому ветвление требует разных
    thread_id, а не повторного resume одного треда."""
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        graph = build_agent(checkpointer)

        cfg_a = {"configurable": {"thread_id": "t-branch-approve"}}
        await graph.ainvoke(_initial_state(), config=cfg_a)
        result_a = await graph.ainvoke(Command(resume=True), config=cfg_a)

        cfg_b = {"configurable": {"thread_id": "t-branch-deny"}}
        await graph.ainvoke(_initial_state(), config=cfg_b)
        result_b = await graph.ainvoke(Command(resume=False), config=cfg_b)

        assert "отправлено" in result_a["tool_results"][-1]["result"]
        assert result_b["tool_results"][-1]["result"] == "Отклонено пользователем — сообщение не отправлено."
        assert mock_bot.send_message.call_count == 1
