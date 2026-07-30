"""Наивный agent loop на Chat Completions (блок 6.1) — фундамент агентного
слоя. Модель — settings.openai.model (Groq openai/gpt-oss-120b), не
gpt-5.4-mini из задания — тот же сдвиг "Groq вместо OpenAI", что и везде
в проекте (см. docs/rag.md).

    python -m app.services.agent_naive "<задача>" [--max-steps N] [--trace]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from openai import OpenAI

from app.services.rag import RAGService
from app.settings import settings as app_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agent_naive")

_client = OpenAI(api_key=app_settings.openai.api_key, base_url=app_settings.openai.host or None)
_rag: RAGService | None = None


def search_knowledge_base(query: str) -> str:
    global _rag
    if _rag is None:
        _rag = RAGService()
        _rag.build()
    sources = asyncio.run(_rag.retrieve(query))["sources"]
    return sources[0]["snippet"] if sources else "В базе знаний ничего не найдено."

def get_current_time(timezone: str = "Europe/Moscow") -> str:
    return datetime.now(ZoneInfo(timezone)).isoformat()

def send_telegram_message(chat_id: str, text: str) -> str:
    print(f"[TELEGRAM → {chat_id}] {text}")
    return f"Сообщение отправлено в {chat_id}"


DISPATCH = {
    "search_knowledge_base": search_knowledge_base,
    "get_current_time": get_current_time,
    "send_telegram_message": send_telegram_message,
}

TOOLS = [
    {"type": "function", "function": {
        "name": "search_knowledge_base",
        "description": "Ищет в базе знаний FinPay фрагмент, релевантный запросу, и возвращает "
                        "один самый подходящий (top-1) кусок текста. Вызывай перед ответом на вопрос "
                        "о конкретных фактах, правилах или инструкциях FinPay.",
        "parameters": {"type": "object", "required": ["query"],
                        "properties": {"query": {"type": "string", "description": "Поисковый запрос"}}}}},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "Возвращает текущие дату и время в указанном часовом поясе в формате ISO 8601. "
                        "Вызывай, когда нужно узнать текущее время, например для метки в сообщении.",
        "parameters": {"type": "object", "required": [],
                        "properties": {"timezone": {"type": "string",
                                                     "description": "IANA-имя таймзоны, например Europe/Moscow"}}}}},
    {"type": "function", "function": {
        "name": "send_telegram_message",
        "description": "Отправляет готовый текст сообщения клиенту в Telegram по его chat_id. Вызывай, "
                        "только когда ответ уже сформулирован и точно нужно его отправить — это финальное "
                        "действие, а не поиск информации.",
        "parameters": {"type": "object", "required": ["chat_id", "text"],
                        "properties": {"chat_id": {"type": "string", "description": "Идентификатор чата Telegram"},
                                        "text": {"type": "string", "description": "Текст сообщения"}}}}},
]


def run_agent(task: str, max_steps: int = 6) -> dict:
    messages = [{"role": "user", "content": task}]
    trace = []
    for step in range(max_steps):
        start = time.perf_counter()
        response = _client.chat.completions.create(model=app_settings.openai.model, messages=messages, tools=TOOLS)
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        message = response.choices[0].message
        messages.append(message)
        if not message.tool_calls:
            return {"answer": message.content, "steps": step + 1, "trace": trace}

        for call in message.tool_calls:
            tool_args = json.loads(call.function.arguments)
            fn = DISPATCH.get(call.function.name)
            try:
                result = fn(**tool_args) if fn else f"Ошибка: нет инструмента '{call.function.name}'"
            except Exception as exc:  # noqa: BLE001 — инструмент не должен ронять весь цикл
                result = f"Ошибка при вызове {call.function.name}: {exc}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
            trace.append({
                "step": step, "tool_name": call.function.name, "tool_args": tool_args,
                "tool_result": str(result)[:200], "duration_ms": duration_ms,
                "llm_input_tokens": response.usage.prompt_tokens,
                "llm_output_tokens": response.usage.completion_tokens,
            })
            logger.info("step=%d tool=%s duration_ms=%.1f", step, call.function.name, duration_ms)

    return {"answer": None, "steps": max_steps, "trace": trace, "error": f"Достигнут max_steps={max_steps}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Наивный agent loop (блок 6.1)")
    parser.add_argument("task")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    result = run_agent(args.task, args.max_steps)
    print(result.get("answer") or f"Остановлено: {result.get('error', 'причина неизвестна')}")
    if args.trace:
        print(json.dumps(result["trace"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
