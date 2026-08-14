"""Мультиагентная система на LangGraph (блок 6.5): supervisor + researcher +
writer, сравнение с single_agent_baseline.py на тех же 5 вопросах и том же
tool (experiments/common.py).

Сборка через langgraph_supervisor.create_supervisor (не ручной Command(goto=...))
— канонiчный туториал LangGraph для этого паттерна, супервизор сам создаёт
handoff-инструменты transfer_to_<name> и ведёт общий messages-state; ручной
supervisor даёт больше контроля, но для пары агентов с одним общим tool это
не требуется — см. docs/multi-agent-report.md.

    python -m experiments.multi_agent_langgraph
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_supervisor import create_supervisor

from experiments.common import MODEL_NAME, QUESTIONS, search_knowledge_base, usage_from_messages
from app.settings import settings as app_settings

RESEARCHER_PROMPT = (
    "Ты researcher. Найди факты по вопросу через search_knowledge_base и "
    "верни маркированный список фактов с номерами источников [N] из ответа "
    "инструмента — строго обычные квадратные скобки ASCII ( [ и ] ), не "
    "круглые и не полноширинные символы вроде 【1】. Финальный ответ "
    "пользователю НЕ пиши — этим займётся writer. Если вопрос не про факты "
    "FinPay — прямо сообщи об этом, не выдумывай."
)
WRITER_PROMPT = (
    "Ты writer. Получаешь список фактов от researcher и собираешь связный "
    "ответ на русском с цитированием источников в формате [1], [2] (номера "
    "из фактов researcher) — строго обычные квадратные скобки ASCII "
    "( [ и ] ), не круглые и не полноширинные символы вроде 【1】. Если "
    "фактов недостаточно или researcher сообщил об отсутствии данных — "
    "честно откажись отвечать, не выдумывай."
)
SUPERVISOR_PROMPT = (
    "Ты супервизор команды из researcher и writer. Сначала ВСЕГДА передавай "
    "задачу researcher для сбора фактов, затем writer — для финального "
    "ответа. Сам не отвечай пользователю, только делегируй."
)

model = ChatOpenAI(
    model=MODEL_NAME, api_key=app_settings.openai.api_key,
    base_url=app_settings.openai.host or None, temperature=0,
)


def build_supervisor_app():
    researcher = create_agent(model=model, tools=[search_knowledge_base],
                               name="researcher", system_prompt=RESEARCHER_PROMPT)
    writer = create_agent(model=model, tools=[], name="writer", system_prompt=WRITER_PROMPT)
    workflow = create_supervisor(agents=[researcher, writer], model=model,
                                  prompt=SUPERVISOR_PROMPT, output_mode="last_message")
    return workflow.compile(checkpointer=InMemorySaver())


def count_handoffs(messages: list) -> int:
    count = 0
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            if tc["name"].startswith("transfer_to_"):
                count += 1
    return count


async def run_one(app, question: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    t0 = time.perf_counter()
    async for event in app.astream({"messages": [HumanMessage(content=question)]},
                                     config, stream_mode="updates"):
        for node in event:
            print(f"    -> {node}")
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    state = await app.aget_state(config)
    messages = state.values["messages"]
    usage = usage_from_messages(messages)
    handoffs = count_handoffs(messages)
    answer = messages[-1].content
    return {"answer": answer, "latency_ms": latency_ms, "handoff_count": handoffs, **usage}


async def main() -> None:
    app = build_supervisor_app()

    mermaid = app.get_graph().draw_mermaid()
    mermaid_path = Path(__file__).resolve().parents[1] / "docs" / "architecture-multi-agent.md"
    mermaid_path.write_text(
        "# Mermaid-схема supervisor-графа (блок 6.5)\n\n```mermaid\n" + mermaid + "\n```\n",
        encoding="utf-8",
    )
    print(f"Mermaid-схема сохранена в {mermaid_path}\n")

    results = []
    for q in QUESTIONS:
        print(f"=== {q['id']}: {q['question']}")
        # Отдельный thread_id на каждый вопрос (не единый "exp-langgraph" из
        # ТЗ) — иначе InMemorySaver копит контекст между вопросами и токены/
        # ответы одного вопроса зависят от предыдущих, сравнение с
        # single-agent (который стартует с чистого листа на каждый вопрос)
        # перестаёт быть честным. Отклонение задокументировано в отчёте.
        r = await run_one(app, q["question"], thread_id=f"exp-langgraph-{q['id']}")
        print(f"  tokens={r['total_tokens']} calls={r['llm_calls']} "
              f"latency={r['latency_ms']:.0f}ms handoffs={r['handoff_count']}")
        print(f"  answer: {r['answer'][:200]!r}")
        results.append({"impl": "multi_agent", **q, **r})

    out_path = Path(__file__).resolve().parent / "results_multi_agent.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
