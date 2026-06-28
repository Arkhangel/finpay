"""Tests for context building and token budget utilities."""

from __future__ import annotations

import pytest

from app.chat.context import build_sliding_window_context, count_tokens, fit_to_budget


def test_count_tokens_returns_positive():
    messages = [{"role": "user", "content": "hello world"}]
    assert count_tokens(messages) > 0


def test_count_tokens_overhead():
    # Empty conversation should still have the 2-token overhead
    assert count_tokens([]) == 2


def test_count_tokens_per_message_overhead():
    # One message: 2 (global) + 4 (per-msg) + tokens for "hi" + tokens for "user"
    result = count_tokens([{"role": "user", "content": "hi"}])
    assert result >= 7  # 2 + 4 + at least 1


def test_fit_to_budget_trims_from_start():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]
    # Very tight budget — should keep only the last message
    single_msg_tokens = count_tokens([messages[-1]])
    result = fit_to_budget(messages, single_msg_tokens + 5)
    assert len(result) == 1
    assert result[0]["content"] == "third"


def test_fit_to_budget_preserves_system():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "old message that won't fit"},
        {"role": "user", "content": "recent"},
    ]
    system_tokens = count_tokens([messages[0]])
    recent_tokens = count_tokens([messages[-1]])
    budget = system_tokens + recent_tokens + 5

    result = fit_to_budget(messages, budget)
    roles = [m["role"] for m in result]
    assert "system" in roles
    assert result[-1]["content"] == "recent"


def test_build_sliding_window_with_system_prompt():
    history = [{"role": "user", "content": f"msg {i}"} for i in range(15)]
    result = build_sliding_window_context(history, "Be helpful", context_window=5)
    assert result[0] == {"role": "system", "content": "Be helpful"}
    assert len(result) == 6  # 1 system + 5 history


def test_build_sliding_window_without_system_prompt():
    history = [{"role": "user", "content": "hi"}]
    result = build_sliding_window_context(history, None, context_window=10)
    assert len(result) == 1
    assert result[0]["role"] == "user"
