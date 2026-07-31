"""Сравнение agent_naive.py (baseline, Б6.1) и agent_react.py (Б6.2) на 5
обязательных тестовых задачах из ТЗ: 2 простых (1 tool), 2 средних
(composability — выход одного tool идёт во вход другого), 1 провокационная
(правильное поведение — НЕ вызывать tool вовсе).

    python scripts/run_agent_comparison.py

Каждая задача прогоняется дважды (naive и react), полный результат (answer +
trace + usage) сохраняется в docs/agent-react-traces/{name}_{agent}.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.services.agent_naive as naive_mod  # noqa: E402
import app.services.agent_react as react_mod  # noqa: E402
from app.services.agent_naive import run_agent as run_naive  # noqa: E402
from app.services.agent_react import run_react_agent as run_react  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "agent-react-traces"

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


def _naive_usage(result: dict) -> int:
    return sum(
        (e.get("llm_input_tokens") or 0) + (e.get("llm_output_tokens") or 0)
        for e in result["trace"]
    )


def _warm_up() -> None:
    """Прогревает RAGService (эмбеддинги + reranker) для обоих агентов ДО
    замера времени — иначе холодный старт (загрузка модели эмбеддингов на
    первом вызове search_knowledge_base) может сам по себе исчерпать
    timeout_per_iteration_sec у agent_react и исказить сравнение."""
    print("=== warm-up (RAGService cold start для naive и react)")
    naive_mod.search_knowledge_base("прогрев")
    react_mod.search_knowledge_base("прогрев")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _warm_up()
    for name, task in SCENARIOS:
        print(f"=== {name}: {task}")

        naive_result = run_naive(task)
        naive_tokens = _naive_usage(naive_result)
        print(f"  [naive] steps={naive_result['steps']} tokens~{naive_tokens} "
              f"answer={str(naive_result.get('answer'))[:150]!r}")
        (OUT_DIR / f"{name}_naive.json").write_text(
            json.dumps({"task": task, **naive_result, "tokens_total": naive_tokens},
                        ensure_ascii=False, indent=2), encoding="utf-8",
        )

        react_result = run_react(task)
        print(f"  [react] steps={react_result['steps']} revisions={react_result['revisions_used']} "
              f"tokens={react_result['usage']['total']} "
              f"answer={str(react_result.get('answer'))[:150]!r}")
        (OUT_DIR / f"{name}_react.json").write_text(
            json.dumps({"task": task, **react_result}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
