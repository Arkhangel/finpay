"""Context strategies and token budget utilities for ChatService."""

from __future__ import annotations

import logging
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

_ENCODING = None


def _get_encoding() -> tiktoken.Encoding:
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("o200k_base")
    return _ENCODING


def _content_text(content: Any) -> str:
    # Мультимодальный content — список parts; image/audio в токен-бюджет не считаем.
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def count_tokens(messages: list[dict[str, Any]]) -> int:
    """Count tokens for a list of OpenAI-format messages (ChatML overhead included)."""
    enc = _get_encoding()
    total = 2  # +2 overhead for the conversation
    for msg in messages:
        total += 4  # +4 per message (role + content framing)
        total += len(enc.encode(_content_text(msg.get("content"))))
        total += len(enc.encode(msg.get("role") or ""))
    return total


def fit_to_budget(messages: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Trim messages from the beginning to fit within token budget.

    System messages are always preserved.
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    system_tokens = count_tokens(system_msgs)
    remaining = budget - system_tokens

    if not non_system:
        return system_msgs

    result: list[dict[str, Any]] = []
    # Walk from the end so we keep the most recent messages
    for msg in reversed(non_system):
        msg_tokens = count_tokens([msg])
        if msg_tokens > remaining:
            break
        result.insert(0, msg)
        remaining -= msg_tokens

    if not result:
        # Даже самое последнее (текущее) сообщение не влезло в остаток бюджета —
        # это гарантированно последний элемент non_system (текущий вопрос
        # пользователя, всегда добавляется последним перед вызовом
        # fit_to_budget). Отдать его всё равно лучше, чем уйти в LLM без
        # единого user-сообщения — усечение вместо полного выпадения турна.
        logger.warning(
            "fit_to_budget_current_turn_exceeds_budget tokens=%d remaining=%d",
            count_tokens([non_system[-1]]), remaining,
        )
        result = [non_system[-1]]

    return system_msgs + result


def build_sliding_window_context(
    history: list[dict[str, Any]],
    system_prompt: str | None,
    context_window: int,
) -> list[dict[str, Any]]:
    """Return the last `context_window` messages prepended with system prompt."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history[-context_window:])
    return messages
