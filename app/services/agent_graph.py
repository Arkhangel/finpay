"""ReAct-агент из agent_react.py (Б6.2) переписан на LangGraph 1.x (блок 6.3):
явный StateGraph, типизированный State, отдельные узлы для модели и
инструментов, conditional edge с router-функцией. Концепция ReAct не
меняется — только orchestration-форма.

Тулы, домен и системный промпт — переиспользованы из agent_react.py (тот же
3-tool набор: search_knowledge_base/get_current_time/send_telegram_message),
чтобы сравнение в scripts/bench_agents.py было честным (одна и та же задача
и промпт на всех трёх реализациях).

Два независимых runnable на одном наборе tools:
- custom_graph — StateGraph руками (задачи 4-5 из ТЗ);
- prebuilt_graph — langchain.agents.create_agent (задача 6 из ТЗ).

    python -m app.services.agent_graph "<задача>"
"""

from __future__ import annotations

import argparse
import asyncio
import operator
from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.services.agent_react import SYSTEM_PROMPT
from app.services.rag import RAGService
from app.settings import settings as app_settings

MAX_ITERATIONS = 6

_rag: RAGService | None = None


def _get_rag() -> RAGService:
    global _rag
    if _rag is None:
        _rag = RAGService()
        _rag.build()
    return _rag


# 1. Tools — те же 3, что и в agent_react.py (Б6.2), тот же текст description
# (там — JSON Schema "description", здесь — docstring: LangChain берёт его
# как description для LLM автоматически через @tool).
@tool
async def search_knowledge_base(query: str) -> str:
    """Ищет в базе знаний FinPay (тарифы, правила, инструкции) фрагмент текста,
    релевантный запросу, и возвращает один самый релевантный (top-1) кусок.
    Вызывай перед ответом на вопрос о конкретных фактах, правилах или тарифах
    FinPay — если вопрос не про факты компании (арифметика, общие знания,
    действия без привязки к фактам), не вызывай. Параметр query — короткий
    поисковый запрос на русском (переформулированный для полнотекстового
    поиска, а не дословная копия вопроса). Возвращает текстовый фрагмент
    документа или строку 'В базе знаний ничего не найдено' — в этом случае
    не выдумывай факты, а честно сообщи об отсутствии данных.
    """
    sources = (await _get_rag().retrieve(query))["sources"]
    return sources[0]["snippet"] if sources else "В базе знаний ничего не найдено."


@tool
def get_current_time(timezone: str) -> str:
    """Возвращает текущие дату и время в формате ISO 8601 для указанного часового
    пояса. Вызывай, когда для ответа или для другого инструмента (например
    отправки сообщения) нужна метка текущего времени. Параметр timezone —
    обязательное IANA-имя пояса (например Europe/Moscow, Asia/Almaty); если
    пользователь не назвал пояс явно, передай 'Europe/Moscow'. Возвращает
    строку ISO 8601 или сообщение об ошибке, если имя пояса некорректно.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    try:
        return datetime.now(ZoneInfo(timezone)).isoformat()
    except Exception as exc:  # noqa: BLE001 — некорректное имя таймзоны от модели
        return f"Ошибка: некорректный часовой пояс '{timezone}' ({exc})"


@tool
def send_telegram_message(chat_id: str, text: str) -> str:
    """Отправляет готовый текст сообщения клиенту в Telegram по его chat_id.
    Вызывай только последним шагом, когда ответ уже полностью сформулирован
    из результатов предыдущих инструментов — это финальное действие, а не
    способ получить информацию. Параметры chat_id (идентификатор чата) и
    text (готовый текст сообщения) оба обязательны. Возвращает
    строку-подтверждение отправки.
    """
    print(f"[TELEGRAM → {chat_id}] {text}")
    return f"Сообщение отправлено в {chat_id}"


TOOLS = [search_knowledge_base, get_current_time, send_telegram_message]

model = ChatOpenAI(
    model=app_settings.openai.model,
    api_key=app_settings.openai.api_key,
    base_url=app_settings.openai.host or None,
    temperature=0,
)
model_with_tools = model.bind_tools(TOOLS)


# 2. State — только сериализуемые данные, без SDK-клиентов/API-ключей.
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int
    tool_results: Annotated[list[dict], operator.add]


# 3. Nodes
async def call_model(state: AgentState) -> dict:
    response = await model_with_tools.ainvoke(state["messages"])
    return {"messages": [response], "iteration_count": state["iteration_count"] + 1}


async def execute_tool(state: AgentState) -> dict:
    last = state["messages"][-1]
    by_name = {t.name: t for t in TOOLS}
    new_messages: list[AnyMessage] = []
    new_results: list[dict] = []
    for tc in last.tool_calls:
        if tc["name"] not in by_name:
            content = f"Ошибка: нет инструмента '{tc['name']}'"
        else:
            try:
                content = str(await by_name[tc["name"]].ainvoke(tc["args"]))
            except Exception as exc:  # noqa: BLE001 — инструмент не должен ронять граф
                content = f"Ошибка при вызове {tc['name']}: {exc}"
        new_messages.append({"role": "tool", "content": content, "tool_call_id": tc["id"]})
        new_results.append({"name": tc["name"], "args": tc["args"], "result": content})
    return {"messages": new_messages, "tool_results": new_results}


async def force_finish(state: AgentState) -> dict:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        # Достигнут max_iterations с незакрытым tool_call — граф не должен
        # молча уйти в END без ответа пользователю.
        return {"messages": [AIMessage(
            content=f"Превышен лимит итераций (max_iterations={MAX_ITERATIONS}) — "
                    "не удалось завершить задачу доступными инструментами.",
        )]}
    return {}  # последний AIMessage уже реальный финальный ответ — прокидываем как есть


# 4. Router — читает state, не пишет в него, без сетевых вызовов.
def route_after_model(state: AgentState) -> Literal["execute_tool", "force_finish"]:
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "force_finish"
    last = state["messages"][-1]
    return "execute_tool" if getattr(last, "tool_calls", None) else "force_finish"


# 5. Сборка кастомного графа
builder = StateGraph(AgentState)
builder.add_node("call_model", call_model)
builder.add_node("execute_tool", execute_tool)
builder.add_node("force_finish", force_finish)
builder.add_edge(START, "call_model")
builder.add_conditional_edges(
    "call_model", route_after_model,
    {"execute_tool": "execute_tool", "force_finish": "force_finish"},
)
builder.add_edge("execute_tool", "call_model")
builder.add_edge("force_finish", END)
custom_graph = builder.compile()

# 6. Prebuilt-вариант — те же model/tools/system_prompt, независимый runnable.
prebuilt_graph = create_agent(
    model=model,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)


async def run_custom(task: str, thread_id: str | None = None) -> dict:
    config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
    result = await custom_graph.ainvoke(
        {
            "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=task)],
            "iteration_count": 0,
            "tool_results": [],
        },
        config=config,
    )
    return result


async def run_prebuilt(task: str) -> dict:
    return await prebuilt_graph.ainvoke({"messages": [HumanMessage(content=task)]})


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph ReAct-агент (блок 6.3)")
    parser.add_argument("task")
    parser.add_argument("--variant", choices=["custom", "prebuilt"], default="custom")
    args = parser.parse_args()

    runner = run_custom if args.variant == "custom" else run_prebuilt
    result = asyncio.run(runner(args.task))
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
