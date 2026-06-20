import json
import logging
from openai import OpenAI

from app.prompts.loader import render_system_prompt
from app.settings import settings
from app.settings.logging import config_logging
from app.tools.schemas import TOOLS
from app.tools.handlers import get_payment_system_status, check_transaction_status

config_logging()

logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=settings.openai.api_key or None,
    base_url=settings.openai.host or None,
)

# Маршрутизация: имя tool → функция
HANDLERS = {
    "get_payment_system_status": get_payment_system_status,
    "check_transaction_status": check_transaction_status,
}


def run(user_input: str) -> str:
    logger.info(json.dumps({"event": "user_input", "text": user_input}, ensure_ascii=False))

    system_prompt = render_system_prompt(project_name=settings.project_name)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    # --- Шаг 1: первый запрос к модели ---
    response = client.chat.completions.create(
        model=settings.openai.model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    message = response.choices[0].message

    # --- Шаг 2: проверяем, вызвала ли модель tool ---
    if not message.tool_calls:
        # Модель ответила текстом — возвращаем как есть
        final = message.content
        logger.info(json.dumps({
            "event": "no_tool_called",
            "final_answer": final,
            "total_tokens": response.usage.total_tokens,
        }, ensure_ascii=False))
        return final

    # --- Шаг 3: выполняем все вызванные tools ---
    # Добавляем ассистентский ход с tool_calls в историю (обязательно!)
    messages.append(message)

    for tool_call in message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        logger.info(json.dumps({
            "event": "tool_called",
            "tool": tool_name,
            "args": tool_args,
        }, ensure_ascii=False))

        # Вызываем обработчик
        handler = HANDLERS.get(tool_name)
        if handler is None:
            tool_result = {"error": f"Неизвестный инструмент: {tool_name}"}
        else:
            tool_result = handler(**tool_args)

        logger.info(json.dumps({
            "event": "tool_result",
            "tool": tool_name,
            "result": tool_result,
        }, ensure_ascii=False))

        # Добавляем результат в историю
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(tool_result, ensure_ascii=False),
        })

    # --- Шаг 4: второй запрос — финальный ответ пользователю ---
    second_response = client.chat.completions.create(
        model=settings.openai.model,
        messages=messages,
    )

    final = second_response.choices[0].message.content

    logger.info(json.dumps({
        "event": "final_answer",
        "text": final,
        "total_tokens": second_response.usage.total_tokens,
    }, ensure_ascii=False))

    return final
