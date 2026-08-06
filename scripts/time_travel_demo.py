"""Time travel по чек-пойнтам персистентного графа (блок 6.4) — офлайн,
sqlite in-memory, модель и Telegram-отправка замоканы (детерминированный
демо-прогон без затрат Groq-квоты и без спама в реальный Telegram; реальная
отправка уже проверена вживую и описана в docs/agent-persistent-report.md).

    python scripts/time_travel_demo.py

Показывает: (1) __interrupt__ payload; (2) историю чек-пойнтов треда;
(3) чтение состояния на ПРОШЛОМ чек-пойнте — уже после того, как тред продолжился
дальше (настоящий time travel, не просто "текущий стейт до resume");
(4) две ветки (отказ/одобрение) из ОДИНАКОВОГО входа на двух РАЗНЫХ thread_id
— повторный resume одного треда с другим значением НЕ даёт вторую ветку
(resume детерминирован по thread-lineage, см. отчёт).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from app.services.agent_persistent import SYSTEM_PROMPT, build_agent  # noqa: E402

TASK = "Отправь клиенту в чат 555 сообщение 'привет от тайм-тревел демо'."


def _tool_call_message() -> AIMessage:
    return AIMessage(content="", tool_calls=[{
        "name": "send_telegram_message",
        "args": {"chat_id": "555", "text": "привет от тайм-тревел демо"},
        "id": "call_1",
    }])


def _initial_state() -> dict:
    return {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=TASK)],
        "iteration_count": 0, "tool_results": [], "pending_action": None,
    }


def _short_id(checkpoint_id: str) -> str:
    # UUIDv7 (time-sortable) — первые символы у соседних чек-пойнтов
    # совпадают, поэтому для читаемости показываем начало И конец.
    return f"{checkpoint_id[:8]}…{checkpoint_id[-6:]}"


def _fmt_snapshot(snap) -> str:
    outcome = "?"
    if snap.values.get("tool_results"):
        outcome = snap.values["tool_results"][-1]["result"][:40]
    cid = _short_id(snap.config["configurable"]["checkpoint_id"])
    return f"checkpoint_id={cid} next={snap.next} outcome={outcome!r}"


async def main() -> None:
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=_tool_call_message())
    mock_bot_instance = MagicMock()
    mock_bot_instance.send_message = AsyncMock()
    mock_bot_instance.session.close = AsyncMock()

    with patch("app.services.agent_persistent.model_with_tools", mock_model), \
         patch("app.services.agent_persistent.Bot", MagicMock(return_value=mock_bot_instance)):

        async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
            graph = build_agent(checkpointer)

            print("=== (1) Запуск до interrupt (thread A) ===")
            cfg_a = {"configurable": {"thread_id": "demo-A"}}
            result = await graph.ainvoke(_initial_state(), config=cfg_a)
            print("__interrupt__ payload:", result["__interrupt__"])

            snap_before_resume = await graph.aget_state(cfg_a)
            old_checkpoint_id = snap_before_resume.config["configurable"]["checkpoint_id"]
            print(f"\n(запомнили checkpoint_id ДО resume: {_short_id(old_checkpoint_id)})")

            print("\n=== (2) История чек-пойнтов thread A (сразу после interrupt) ===")
            history = [s async for s in graph.aget_state_history(cfg_a)]
            for snap in history:
                print(" ", _fmt_snapshot(snap))

            print("\n=== Резюмируем thread A (Command(resume=True)) — тред продолжается ===")
            await graph.ainvoke(Command(resume=True), config=cfg_a)

            print("\n=== (3) Time travel: читаем СТАРЫЙ checkpoint_id ПОСЛЕ того, как тред уже продолжился ===")
            past_state = await graph.aget_state({
                "configurable": {"thread_id": "demo-A", "checkpoint_id": old_checkpoint_id},
            })
            print("  next:", past_state.next, "(узел подтверждения — ещё ничего не отправлено)")
            print("  pending_action:", past_state.values.get("pending_action"))
            print("  tool_results на этом чек-пойнте:", past_state.values.get("tool_results"))
            print("  (в БОЛЕЕ НОВЫХ чек-пойнтах того же треда tool_results уже содержит реальный "
                  "результат отправки — вот и путешествие во времени: чек-пойнт не изменился, "
                  "хотя тред давно ушёл вперёд)")

            print("\n=== (4) Две ветки из ОДИНАКОВОГО входа — на РАЗНЫХ thread_id ===")
            cfg_approve = {"configurable": {"thread_id": "demo-B-approve"}}
            await graph.ainvoke(_initial_state(), config=cfg_approve)
            result_approve = await graph.ainvoke(Command(resume=True), config=cfg_approve)

            cfg_deny = {"configurable": {"thread_id": "demo-C-deny"}}
            await graph.ainvoke(_initial_state(), config=cfg_deny)
            result_deny = await graph.ainvoke(Command(resume=False), config=cfg_deny)

            print("  demo-B (resume=True): ", result_approve["tool_results"][-1]["result"])
            print("  demo-C (resume=False):", result_deny["tool_results"][-1]["result"])
            print(f"\n  Bot.send_message вызван всего {mock_bot_instance.send_message.call_count} раза "
                  f"за весь прогон (thread A + demo-B, оба resume=True) — demo-C (resume=False) "
                  f"вклада не внёс, как и ожидалось")


if __name__ == "__main__":
    asyncio.run(main())
