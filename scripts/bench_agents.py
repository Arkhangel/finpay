"""Бенчмарк Б6.2 (agent_react.py, наивный loop) vs custom_graph vs
prebuilt_graph (обе — LangGraph, agent_graph.py) на тех же 5 задачах, что и
scripts/run_agent_comparison.py (Б6.2). 3 повтора на задачу × реализацию,
усреднение по latency (time.perf_counter()) и token usage.

    python scripts/bench_agents.py

Сырые данные — docs/agent-graph-bench-raw.json, таблица (среднее по 3
повторам) печатается в stdout в готовом markdown для вставки в
docs/agent-graph-report.md.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.services.agent_graph as graph_mod  # noqa: E402
import app.services.agent_react as react_mod  # noqa: E402
from app.services.agent_graph import run_custom, run_prebuilt  # noqa: E402
from app.services.agent_react import run_react_agent  # noqa: E402

OUT_RAW = Path(__file__).resolve().parents[1] / "docs" / "agent-graph-bench-raw.json"
REPEATS = 3

SCENARIOS = [
    ("01_simple_fact",
     "Найди в базе знаний стандартную комиссию за транзакцию в FinPay и назови её."),
    ("02_simple_time",
     "Который сейчас час в Алматы?"),
    ("03_composability_kb_to_telegram",
     "Найди в базе знаний контакты поддержки FinPay для Казахстана и отправь эту "
     "информацию клиенту в чат 555."),
    ("04_composability_time_to_telegram",
     "Узнай текущее время в Алматы и отправь его клиенту в чат 777 одной короткой фразой."),
    ("05_provocative_no_tool",
     "Проверь баланс пользователя с id 42."),
]


async def _warm_up() -> None:
    print("=== warm-up (RAGService cold start для всех трёх реализаций)")
    # react_mod.search_knowledge_base сам делает asyncio.run() внутри (Б6.2,
    # синхронная функция) — нельзя звать напрямую из уже работающего event
    # loop этого скрипта, поэтому уводим в отдельный поток.
    await asyncio.to_thread(react_mod.search_knowledge_base, "прогрев")
    await graph_mod.search_knowledge_base.ainvoke({"query": "прогрев"})


def _ai_messages(messages: list) -> list:
    return [m for m in messages if type(m).__name__ == "AIMessage"]


async def _run_react(task: str) -> dict:
    t0 = time.perf_counter()
    result = await asyncio.to_thread(run_react_agent, task)
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "latency_ms": latency_ms,
        "prompt_tokens": result["usage"]["prompt"],
        "completion_tokens": result["usage"]["completion"],
        "total_tokens": result["usage"]["total"],
        "total_steps": result["steps"],
    }


async def _run_custom(task: str, thread_id: str) -> dict:
    t0 = time.perf_counter()
    result = await run_custom(task, thread_id=thread_id)
    latency_ms = (time.perf_counter() - t0) * 1000
    ai_msgs = _ai_messages(result["messages"])
    prompt = sum(m.usage_metadata["input_tokens"] for m in ai_msgs if m.usage_metadata)
    completion = sum(m.usage_metadata["output_tokens"] for m in ai_msgs if m.usage_metadata)
    return {
        "latency_ms": latency_ms,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "total_steps": result["iteration_count"],
    }


async def _run_prebuilt(task: str) -> dict:
    t0 = time.perf_counter()
    result = await run_prebuilt(task)
    latency_ms = (time.perf_counter() - t0) * 1000
    ai_msgs = _ai_messages(result["messages"])
    prompt = sum(m.usage_metadata["input_tokens"] for m in ai_msgs if m.usage_metadata)
    completion = sum(m.usage_metadata["output_tokens"] for m in ai_msgs if m.usage_metadata)
    return {
        "latency_ms": latency_ms,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "total_steps": len(ai_msgs),
    }


async def main() -> None:
    await _warm_up()
    raw: list[dict] = []

    for name, task in SCENARIOS:
        for impl, runner in (
            ("react", lambda t: _run_react(t)),
            ("custom", lambda t, n=name: _run_custom(t, thread_id=f"bench-{n}")),
            ("prebuilt", lambda t: _run_prebuilt(t)),
        ):
            for repeat in range(1, REPEATS + 1):
                measurement = await runner(task)
                row = {"task": name, "impl": impl, "repeat": repeat, **measurement}
                raw.append(row)
                print(f"{name} | {impl} | run {repeat}/{REPEATS} | "
                      f"latency={measurement['latency_ms']:.0f}ms "
                      f"tokens={measurement['total_tokens']} steps={measurement['total_steps']}")

    OUT_RAW.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    # Агрегация: среднее по REPEATS для каждой (task, impl)
    print("\n\n| # | Задача | Реализация | latency_ms (avg) | prompt_tokens (avg) | "
          "completion_tokens (avg) | total_steps (avg) |")
    print("|---|--------|------------|-------------------|----------------------|"
          "--------------------------|--------------------|")
    for i, (name, _task) in enumerate(SCENARIOS, start=1):
        for impl in ("react", "custom", "prebuilt"):
            rows = [r for r in raw if r["task"] == name and r["impl"] == impl]
            n = len(rows)
            avg_latency = sum(r["latency_ms"] for r in rows) / n
            avg_prompt = sum(r["prompt_tokens"] for r in rows) / n
            avg_completion = sum(r["completion_tokens"] for r in rows) / n
            avg_steps = sum(r["total_steps"] for r in rows) / n
            print(f"| {i} | {name} | {impl} | {avg_latency:.0f} | {avg_prompt:.0f} | "
                  f"{avg_completion:.0f} | {avg_steps:.1f} |")


if __name__ == "__main__":
    asyncio.run(main())
