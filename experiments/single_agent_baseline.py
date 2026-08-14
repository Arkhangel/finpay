"""Single-agent baseline (блок 6.5) — точка отсчёта для сравнения с
supervisor-графом из multi_agent_langgraph.py. Один create_agent с тем же
tool search_knowledge_base (experiments/common.py) и промптом, объединяющим
обе роли: найти факты И оформить ответ с цитированием.

    python -m experiments.single_agent_baseline
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from experiments.common import MODEL_NAME, QUESTIONS, search_knowledge_base, usage_from_messages
from app.settings import settings as app_settings

SYSTEM_PROMPT = (
    "Ты ассистент поддержки FinPay. Сначала найди нужные факты через "
    "search_knowledge_base, затем оформи связный ответ на русском с "
    "цитированием источников в формате [1], [2] (номера из ответа "
    "инструмента) — строго обычные квадратные скобки ASCII ( [ и ] ), не "
    "круглые и не полноширинные символы вроде 【1】. Если вопрос не про "
    "факты FinPay (например требует общих знаний, не связанных с FinPay) — "
    "не вызывай инструмент и честно объясни, что не можешь ответить по базе "
    "знаний FinPay. Не выдумывай факты, которых нет в результате инструмента."
)

model = ChatOpenAI(
    model=MODEL_NAME, api_key=app_settings.openai.api_key,
    base_url=app_settings.openai.host or None, temperature=0,
)


def build_agent():
    return create_agent(model=model, tools=[search_knowledge_base], system_prompt=SYSTEM_PROMPT)


async def run_one(agent, question: str) -> dict:
    t0 = time.perf_counter()
    result = await agent.ainvoke({"messages": [HumanMessage(content=question)]})
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    usage = usage_from_messages(result["messages"])
    answer = result["messages"][-1].content
    return {"answer": answer, "latency_ms": latency_ms, "handoff_count": 0, **usage}


async def main() -> None:
    agent = build_agent()
    results = []
    for q in QUESTIONS:
        r = await run_one(agent, q["question"])
        print(f"[{q['id']}] tokens={r['total_tokens']} calls={r['llm_calls']} "
              f"latency={r['latency_ms']:.0f}ms")
        print(f"  answer: {r['answer'][:200]!r}")
        results.append({"impl": "single_agent", **q, **r})

    out_path = Path(__file__).resolve().parent / "results_single_agent.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
