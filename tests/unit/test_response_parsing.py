"""Unit-тесты для парсинга ответов LLM.

Покрывает: JSON в markdown-fence, JSON без fence, malformed JSON, tool_calls.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.utils.response_parser import extract_json, parse_tool_calls


def test_extract_json_from_markdown_fence():
    content = '```json\n{"reasoning": "ok", "score": 5}\n```'
    result = extract_json(content)
    assert result == {"reasoning": "ok", "score": 5}


def test_extract_json_from_fence_without_language_tag():
    content = '```\n{"status": "succeeded"}\n```'
    result = extract_json(content)
    assert result == {"status": "succeeded"}


def test_extract_json_without_fence():
    content = '{"correctness": 4, "relevance": 5}'
    result = extract_json(content)
    assert result == {"correctness": 4, "relevance": 5}


def test_extract_json_malformed_raises_value_error():
    content = "```json\n{not: valid json\n```"
    with pytest.raises(ValueError, match="Invalid JSON"):
        extract_json(content)


def test_extract_json_plain_malformed_raises_value_error():
    content = "это не JSON вообще"
    with pytest.raises(ValueError, match="Invalid JSON"):
        extract_json(content)


def test_parse_tool_calls_single_call():
    tc = MagicMock()
    tc.id = "call_abc"
    tc.function.name = "check_transaction_status"
    tc.function.arguments = '{"transaction_id": "TXN-1001"}'

    result = parse_tool_calls([tc])

    assert len(result) == 1
    assert result[0]["name"] == "check_transaction_status"
    assert result[0]["arguments"] == {"transaction_id": "TXN-1001"}
    assert result[0]["id"] == "call_abc"


def test_parse_tool_calls_multiple_calls():
    tc1 = MagicMock()
    tc1.id = "call_1"
    tc1.function.name = "check_transaction_status"
    tc1.function.arguments = '{"transaction_id": "TXN-1001"}'

    tc2 = MagicMock()
    tc2.id = "call_2"
    tc2.function.name = "get_payment_system_status"
    tc2.function.arguments = '{"system": "visa"}'

    result = parse_tool_calls([tc1, tc2])

    assert len(result) == 2
    assert result[0]["name"] == "check_transaction_status"
    assert result[1]["name"] == "get_payment_system_status"


def test_parse_tool_calls_malformed_arguments_returns_empty_dict():
    tc = MagicMock()
    tc.id = "call_bad"
    tc.function.name = "some_tool"
    tc.function.arguments = "not json at all"

    result = parse_tool_calls([tc])

    assert result[0]["arguments"] == {}
