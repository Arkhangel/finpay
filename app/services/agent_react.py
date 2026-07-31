"""ReAct-агент с self-reflection (блок 6.2) — управляемая версия наивного
agent_naive.py (Б6.1): жёсткие лимиты (max_iterations, timeout на итерацию)
вместо голого for-цикла, critic-вызов (Reflexion-light) после каждой tool
observation, строгие JSON Schema инструментов вместо произвольных.

agent_naive.py НЕ меняется — он остаётся baseline для сравнения (см.
docs/agent-react-report.md).

Модели — тот же сдвиг "Groq вместо OpenAI/gpt-5.4-mini", что и везде в
проекте: model_main/model_premium — settings.openai.model (Groq
openai/gpt-oss-120b, продакшен-модель); model_critic — settings.eval.judge_model
(openai/gpt-oss-20b) — та же младшая модель, что уже играет роль независимого
судьи в Б5.6, здесь — роль критика. model_premium по умолчанию совпадает с
model_main: на бесплатном Groq tier модели мощнее продакшена под рукой нет,
поэтому "премиум-шаг" — это управляемая, но по умолчанию неактивная точка
расширения (документируемое отклонение от задания), а не выдумывание
несуществующей модели.

    python -m app.services.agent_react "<задача>" [--max-iterations N] [--trace]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import structlog
from openai import APITimeoutError, OpenAI

from app.services.rag import RAGService
from app.settings import settings as app_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = structlog.get_logger("agent_react")

_client = OpenAI(api_key=app_settings.openai.api_key, base_url=app_settings.openai.host or None)
_rag: RAGService | None = None


def search_knowledge_base(query: str) -> str:
    global _rag
    if _rag is None:
        _rag = RAGService()
        _rag.build()
    sources = asyncio.run(_rag.retrieve(query))["sources"]
    return sources[0]["snippet"] if sources else "В базе знаний ничего не найдено."


def get_current_time(timezone: str) -> str:
    try:
        return datetime.now(ZoneInfo(timezone)).isoformat()
    except Exception as exc:  # noqa: BLE001 — некорректное имя таймзоны от модели
        return f"Ошибка: некорректный часовой пояс '{timezone}' ({exc})"


def send_telegram_message(chat_id: str, text: str) -> str:
    print(f"[TELEGRAM → {chat_id}] {text}")
    return f"Сообщение отправлено в {chat_id}"


DISPATCH = {
    "search_knowledge_base": search_knowledge_base,
    "get_current_time": get_current_time,
    "send_telegram_message": send_telegram_message,
}

# Строгая JSON Schema (additionalProperties: false, все параметры в required —
# strict-режим OpenAI/Groq function calling не допускает опциональных
# полей). Описания отвечают на 4 вопроса: что делает / когда вызывать /
# что значат параметры / что возвращает (в т.ч. при пустом результате).
TOOLS = [
    {"type": "function", "function": {
        "name": "search_knowledge_base",
        "description": (
            "Ищет в базе знаний FinPay (тарифы, правила, инструкции) фрагмент текста, "
            "релевантный запросу, и возвращает один самый релевантный (top-1) кусок. "
            "Вызывай перед ответом на вопрос о конкретных фактах, правилах или тарифах "
            "FinPay — если вопрос не про факты компании (арифметика, общие знания, "
            "действия без привязки к фактам), не вызывай. Параметр query — короткий "
            "поисковый запрос на русском (переформулированный для полнотекстового "
            "поиска, а не дословная копия вопроса). Возвращает текстовый фрагмент "
            "документа или строку 'В базе знаний ничего не найдено' — в этом случае "
            "не выдумывай факты, а честно сообщи об отсутствии данных."
        ),
        "strict": True,
        "parameters": {
            "type": "object", "additionalProperties": False,
            "required": ["query"],
            "properties": {"query": {"type": "string", "description": "Поисковый запрос"}},
        },
    }},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": (
            "Возвращает текущие дату и время в формате ISO 8601 для указанного часового "
            "пояса. Вызывай, когда для ответа или для другого инструмента (например "
            "отправки сообщения) нужна метка текущего времени. Параметр timezone — "
            "обязательное IANA-имя пояса (например Europe/Moscow, Asia/Almaty); если "
            "пользователь не назвал пояс явно, передай 'Europe/Moscow'. Возвращает "
            "строку ISO 8601 или сообщение об ошибке, если имя пояса некорректно."
        ),
        "strict": True,
        "parameters": {
            "type": "object", "additionalProperties": False,
            "required": ["timezone"],
            "properties": {"timezone": {"type": "string", "description": "IANA-имя таймзоны"}},
        },
    }},
    {"type": "function", "function": {
        "name": "send_telegram_message",
        "description": (
            "Отправляет готовый текст сообщения клиенту в Telegram по его chat_id. "
            "Вызывай только последним шагом, когда ответ уже полностью сформулирован "
            "из результатов предыдущих инструментов — это финальное действие, а не "
            "способ получить информацию. Параметры chat_id (идентификатор чата) и "
            "text (готовый текст сообщения) оба обязательны. Возвращает "
            "строку-подтверждение отправки."
        ),
        "strict": True,
        "parameters": {
            "type": "object", "additionalProperties": False,
            "required": ["chat_id", "text"],
            "properties": {
                "chat_id": {"type": "string", "description": "Идентификатор чата Telegram"},
                "text": {"type": "string", "description": "Текст сообщения"},
            },
        },
    }},
]

SYSTEM_PROMPT = (
    "Ты — ReAct-агент FinPay. На каждом шаге сначала одним предложением "
    "поясни, что и зачем собираешься сделать, затем вызови ровно один "
    "инструмент и опирайся на его результат. Как только данных достаточно — "
    "дай финальный ответ без вызова инструментов. Не выдумывай данные: "
    "используй только то, что вернули инструменты. Если доступными "
    "инструментами задачу решить нельзя (например арифметика или действие, "
    "для которого нет инструмента) — не вызывай инструменты вообще и прямо "
    "сообщи об этом или ответь напрямую, если это в твоих силах без инструментов."
)

CRITIC_SYSTEM_PROMPT = (
    "Ты — критик агента (Reflexion-light). Тебе даны: вопрос пользователя, "
    "thought агента перед действием, вызванный инструмент с аргументами и "
    "observation (результат инструмента). Проверь: (1) observation "
    "релевантен thought и вопросу; (2) агент не собирается выдумать данные, "
    "которых observation не содержит; (3) наблюдение не является пустым/"
    "отказом, который агент планирует проигнорировать. Ответь СТРОГО одной "
    "строкой: 'OK' если всё в порядке, или 'REVISE: <короткая причина>', "
    "если план нужно пересмотреть."
)


def _accumulate_usage(usage_total: dict, usage, role: str) -> None:
    if usage is None:
        return
    usage_total["prompt"] += usage.prompt_tokens
    usage_total["completion"] += usage.completion_tokens
    usage_total["total"] += usage.total_tokens
    usage_total.setdefault("by_role", {}).setdefault(role, 0)
    usage_total["by_role"][role] += usage.total_tokens


def _critic_verdict(model_critic: str, question: str, thought: str, tool_name: str,
                     tool_args: dict, observation: str, usage_total: dict,
                     timeout: float) -> str:
    prompt = (
        f"Вопрос пользователя: {question}\n"
        f"Thought перед действием: {thought or '(пусто)'}\n"
        f"Action: {tool_name}({tool_args})\n"
        f"Observation: {observation}"
    )
    response = _client.chat.completions.create(
        model=model_critic,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=80, temperature=0, timeout=timeout,
    )
    _accumulate_usage(usage_total, response.usage, role="critic")
    return (response.choices[0].message.content or "OK").strip()


def run_react_agent(
    task: str,
    max_iterations: int = 10,
    timeout_per_iteration_sec: float = 10.0,
    max_revisions: int = 2,
    model_main: str | None = None,
    model_premium: str | None = None,
    model_critic: str | None = None,
) -> dict:
    if not 8 <= max_iterations <= 20:
        msg = f"max_iterations={max_iterations} вне допустимого диапазона 8-20"
        raise ValueError(msg)
    if not 5 <= timeout_per_iteration_sec <= 15:
        msg = f"timeout_per_iteration_sec={timeout_per_iteration_sec} вне диапазона 5-15"
        raise ValueError(msg)

    model_main = model_main or app_settings.openai.model
    model_premium = model_premium or app_settings.openai.model
    model_critic = model_critic or app_settings.eval.judge_model

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    revisions_used = 0
    usage_total = {"prompt": 0, "completion": 0, "total": 0, "by_role": {}}
    trace: list[dict] = []
    current_model = model_main

    for step in range(max_iterations):
        t0 = time.monotonic()
        try:
            response = _client.chat.completions.create(
                model=current_model, messages=messages, tools=TOOLS,
                tool_choice="auto", timeout=timeout_per_iteration_sec,
            )
        except APITimeoutError:
            log.warning("react.timeout", step=step, model=current_model)
            return {"answer": None, "error": "Timeout", "steps": step, "trace": trace,
                    "usage": usage_total, "revisions_used": revisions_used}

        _accumulate_usage(usage_total, response.usage, role="main")
        message = response.choices[0].message
        messages.append(message)
        thought = message.content or ""

        if not message.tool_calls:
            latency_ms = round((time.monotonic() - t0) * 1000, 1)
            log.info("react.step", step=step, tool_name=None, latency_ms=latency_ms,
                      total_tokens=response.usage.total_tokens if response.usage else None)
            if message.content and message.content.strip():
                log.info("react.usage_total", **usage_total)
                return {"answer": message.content, "steps": step + 1, "trace": trace,
                        "usage": usage_total, "revisions_used": revisions_used}
            # Пустой content без tool_calls — модель ничего не сделала и ничего
            # не ответила (обнаружено эмпирически: изредка случается на Groq
            # под нагрузкой). Не завершаем цикл молча с answer="" — просим
            # модель либо дать ответ, либо вызвать инструмент.
            log.warning("react.empty_response", step=step)
            messages.append({
                "role": "user",
                "content": "Ты не вызвал инструмент и не дал ответа. Дай финальный ответ "
                            "или вызови подходящий инструмент.",
            })
            continue

        if len(message.tool_calls) > 1:
            log.warning("react.multi_tool_call", step=step, count=len(message.tool_calls))

        for call in message.tool_calls:
            tool_args = json.loads(call.function.arguments)
            fn = DISPATCH.get(call.function.name)
            try:
                observation = fn(**tool_args) if fn else f"Ошибка: нет инструмента '{call.function.name}'"
            except Exception as exc:  # noqa: BLE001 — инструмент не должен ронять весь цикл
                observation = f"Ошибка при вызове {call.function.name}: {exc}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(observation)})

            verdict = None
            if revisions_used < max_revisions:
                verdict = _critic_verdict(
                    model_critic, task, thought, call.function.name, tool_args,
                    str(observation), usage_total, timeout_per_iteration_sec,
                )
                if verdict.upper().startswith("REVISE"):
                    revisions_used += 1
                    messages.append({"role": "system", "content": f"[critic] {verdict}"})
                    current_model = model_premium
                    log.info("react.revise", step=step, revisions_used=revisions_used, verdict=verdict)

            latency_ms = round((time.monotonic() - t0) * 1000, 1)
            trace.append({
                "step": step, "thought": thought, "tool_name": call.function.name,
                "tool_args": tool_args, "observation": str(observation)[:200],
                "critic_verdict": verdict, "latency_ms": latency_ms,
                "llm_input_tokens": response.usage.prompt_tokens if response.usage else None,
                "llm_output_tokens": response.usage.completion_tokens if response.usage else None,
            })
            log.info("react.step", step=step, tool_name=call.function.name, latency_ms=latency_ms,
                      critic_verdict=verdict)

        if time.monotonic() - t0 > timeout_per_iteration_sec:
            log.warning("react.timeout", step=step, model=current_model)
            return {"answer": None, "error": "Timeout", "steps": step + 1, "trace": trace,
                    "usage": usage_total, "revisions_used": revisions_used}

    log.warning("react.max_iterations_exceeded", max_iterations=max_iterations)
    log.info("react.usage_total", **usage_total)
    return {"answer": None, "error": f"Превышен лимит итераций (max_iterations={max_iterations})",
            "steps": max_iterations, "trace": trace, "usage": usage_total,
            "revisions_used": revisions_used}


def main() -> None:
    parser = argparse.ArgumentParser(description="ReAct-агент с self-reflection (блок 6.2)")
    parser.add_argument("task")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    result = run_react_agent(args.task, max_iterations=args.max_iterations)
    print(result.get("answer") or f"Остановлено: {result.get('error', 'причина неизвестна')}")
    print(f"steps={result['steps']} revisions_used={result['revisions_used']} usage={result['usage']}")
    if args.trace:
        print(json.dumps(result["trace"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
