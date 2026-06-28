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


def count_tokens(messages: list[dict[str, Any]]) -> int:
    """Count tokens for a list of OpenAI-format messages (ChatML overhead included)."""
    enc = _get_encoding()
    total = 2  # +2 overhead for the conversation
    for msg in messages:
        total += 4  # +4 per message (role + content framing)
        total += len(enc.encode(msg.get("content") or ""))
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

    result: list[dict[str, Any]] = []
    # Walk from the end so we keep the most recent messages
    for msg in reversed(non_system):
        msg_tokens = count_tokens([msg])
        if msg_tokens > remaining:
            break
        result.insert(0, msg)
        remaining -= msg_tokens

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
