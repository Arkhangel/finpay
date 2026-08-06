"""Персистентный ReAct-граф (блок 6.4) — custom_graph из agent_graph.py
(Б6.3) получает checkpointer (SQLite/Postgres/memory) и human-in-the-loop
для одного опасного tool (send_telegram_message — теперь настоящая отправка
в Telegram через aiogram, а не print).

agent_graph.py НЕ меняется — остаётся in-memory вариантом для unit-тестов
(Б6.3 checklist). Безопасные tools (search_knowledge_base/get_current_time)
и SYSTEM_PROMPT переиспользуются оттуда напрямую, без дублирования.

HIL: prepare_send_telegram (idempotent — только формирует payload из
tool_call, никакого side-effect) -> confirm_and_execute_send_telegram
(interrupt() + реальная отправка ПОСЛЕ resume). До interrupt — только
подготовка; после — сам side-effect. Если это нарушить, при resume узел
перезапустится с начала и сообщение отправится дважды (см.
docs/agent-persistent-report.md, раздел про идемпотентность).

    python -m app.services.agent_persistent "<задача>" --thread-id demo-1
"""

from __future__ import annotations

import argparse
import asyncio
import operator
from contextlib import asynccontextmanager
from typing import Annotated, Literal, TypedDict

from aiogram import Bot
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from app.services.agent_graph import (
    SYSTEM_PROMPT,
    get_current_time,
    model,
    search_knowledge_base,
)
from app.settings import settings as app_settings

MAX_ITERATIONS = 6
DANGEROUS_TOOL_NAME = "send_telegram_message"


@tool
def send_telegram_message(chat_id: str, text: str) -> str:
    """Отправляет готовый текст сообщения клиенту в Telegram по его chat_id.
    Вызывай только последним шагом, когда ответ уже полностью сформулирован
    из результатов предыдущих инструментов — это финальное действие, а не
    способ получить информацию. Параметры chat_id (идентификатор чата) и
    text (готовый текст сообщения) оба обязательны. Возвращает
    строку-подтверждение отправки.
    """
    # Тело никогда не вызывается напрямую — исполнение разнесено на
    # prepare_send_telegram/confirm_and_execute_send_telegram ниже. Функция
    # существует только чтобы дать LLM JSON Schema через model.bind_tools().
    msg = "send_telegram_message исполняется через HIL-узлы, не напрямую"
    raise NotImplementedError(msg)


SAFE_TOOLS = [search_knowledge_base, get_current_time]
ALL_TOOLS = [*SAFE_TOOLS, send_telegram_message]
model_with_tools = model.bind_tools(ALL_TOOLS)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int
    tool_results: Annotated[list[dict], operator.add]
    # Payload, подготовленный prepare_send_telegram (idempotent) для
    # confirm_and_execute_send_telegram — обычный replace-reducer, читается
    # и пишется только этой парой узлов.
    pending_action: dict | None


async def call_model(state: AgentState) -> dict:
    response = await model_with_tools.ainvoke(state["messages"])
    return {"messages": [response], "iteration_count": state["iteration_count"] + 1}


async def execute_tool(state: AgentState) -> dict:
    """Только безопасные tools — SAFE_TOOLS, без send_telegram_message даже
    защитно (если бы сюда как-то попал вызов опасного tool, DISPATCH его
    просто не найдёт, а не выполнит)."""
    last = state["messages"][-1]
    by_name = {t.name: t for t in SAFE_TOOLS}
    new_messages: list[AnyMessage] = []
    new_results: list[dict] = []
    for tc in last.tool_calls:
        if tc["name"] not in by_name:
            content = f"Ошибка: нет инструмента '{tc['name']}'"
        else:
            try:
                content = str(await by_name[tc["name"]].ainvoke(tc["args"]))
            except Exception as exc:  # noqa: BLE001
                content = f"Ошибка при вызове {tc['name']}: {exc}"
        new_messages.append({"role": "tool", "content": content, "tool_call_id": tc["id"]})
        new_results.append({"name": tc["name"], "args": tc["args"], "result": content})
    return {"messages": new_messages, "tool_results": new_results}


async def prepare_send_telegram(state: AgentState) -> dict:
    """Idempotent: только рендерит payload из уже сделанного моделью
    tool_call. Никакого side-effect — безопасно перезапускать сколько угодно
    раз (в т.ч. при resume, когда граф проигрывает узлы заново)."""
    last = state["messages"][-1]
    call = next(tc for tc in last.tool_calls if tc["name"] == DANGEROUS_TOOL_NAME)
    return {"pending_action": {"tool_call_id": call["id"], "args": call["args"]}}


async def _send_real_telegram(chat_id: str, text: str) -> str:
    bot = Bot(token=app_settings.bot.token)
    try:
        await bot.send_message(chat_id=int(chat_id), text=text)
        return f"Сообщение отправлено в {chat_id}"
    except Exception as exc:  # noqa: BLE001 — не должно ронять граф
        return f"Ошибка при отправке в Telegram: {exc}"
    finally:
        await bot.session.close()


async def confirm_and_execute_send_telegram(state: AgentState, config: RunnableConfig) -> dict:
    """Всё, что ДО interrupt() ниже — уже сделано в prepare_send_telegram
    (idempotent). Сам side-effect (реальный вызов Telegram API) — СТРОГО
    после interrupt(), то есть только после подтверждения человеком."""
    payload = state["pending_action"]
    user_role = (config.get("configurable") or {}).get("user_role", "write-with-approve")

    if user_role == "full":
        decision = True
    else:
        decision = interrupt({"preview": payload, "type": "approve_send_telegram_message"})

    if not decision:
        result = "Отклонено пользователем — сообщение не отправлено."
    else:
        result = await _send_real_telegram(payload["args"]["chat_id"], payload["args"]["text"])

    return {
        "messages": [{"role": "tool", "content": result, "tool_call_id": payload["tool_call_id"]}],
        "tool_results": [{"name": DANGEROUS_TOOL_NAME, "args": payload["args"], "result": result}],
        "pending_action": None,
    }


async def force_finish(state: AgentState) -> dict:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return {"messages": [AIMessage(
            content=f"Превышен лимит итераций (max_iterations={MAX_ITERATIONS}) — "
                    "не удалось завершить задачу доступными инструментами.",
        )]}
    return {}


def route_after_model(state: AgentState) -> Literal["execute_tool", "prepare_send_telegram", "force_finish"]:
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "force_finish"
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return "force_finish"
    if last.tool_calls[0]["name"] == DANGEROUS_TOOL_NAME:
        return "prepare_send_telegram"
    return "execute_tool"


def build_agent(checkpointer):
    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tool", execute_tool)
    builder.add_node("prepare_send_telegram", prepare_send_telegram)
    builder.add_node("confirm_and_execute_send_telegram", confirm_and_execute_send_telegram)
    builder.add_node("force_finish", force_finish)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model", route_after_model,
        {
            "execute_tool": "execute_tool",
            "prepare_send_telegram": "prepare_send_telegram",
            "force_finish": "force_finish",
        },
    )
    builder.add_edge("execute_tool", "call_model")
    # Отдельный edge между prepare и confirm (не один узел) — checklist Б6.4.
    builder.add_edge("prepare_send_telegram", "confirm_and_execute_send_telegram")
    builder.add_edge("confirm_and_execute_send_telegram", "call_model")
    builder.add_edge("force_finish", END)

    return builder.compile(checkpointer=checkpointer)


@asynccontextmanager
async def agent_lifespan():
    """await checkpointer.setup() вызывается здесь РОВНО ОДИН раз — вызывающий
    код (app/lifespan.py) держит этот контекст на весь lifetime приложения,
    не на каждый запрос."""
    backend = app_settings.agent.checkpointer
    if backend == "memory":
        yield InMemorySaver()
        return
    if backend == "sqlite":
        async with AsyncSqliteSaver.from_conn_string(app_settings.agent.sqlite_path) as saver:
            await saver.setup()
            yield saver
        return
    if backend == "postgres":
        async with AsyncPostgresSaver.from_conn_string(app_settings.agent.postgres_uri) as saver:
            await saver.setup()
            yield saver
        return
    msg = f"Неизвестный AGENT__CHECKPOINTER={backend!r}"
    raise ValueError(msg)


async def _run_cli(task: str, thread_id: str) -> None:
    async with agent_lifespan() as checkpointer:
        graph = build_agent(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        result = await graph.ainvoke(
            {
                "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=task)],
                "iteration_count": 0, "tool_results": [], "pending_action": None,
            },
            config=config,
        )
        if "__interrupt__" in result:
            print(f"[INTERRUPT] {result['__interrupt__']}")
            answer = input("Подтвердить действие? [y/N]: ").strip().lower() == "y"
            result = await graph.ainvoke(Command(resume=answer), config=config)
        print(result["messages"][-1].content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Персистентный ReAct-агент (блок 6.4)")
    parser.add_argument("task")
    parser.add_argument("--thread-id", default="cli-demo")
    args = parser.parse_args()
    asyncio.run(_run_cli(args.task, args.thread_id))


if __name__ == "__main__":
    main()
