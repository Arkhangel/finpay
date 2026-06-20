"""Утилиты для парсинга ответов LLM."""
from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_json(content: str) -> dict:
    """Extract and parse JSON from LLM response (with or without markdown fence).

    Raises ValueError if the content cannot be parsed as JSON.
    """
    stripped = content.strip()
    match = _FENCE_RE.search(stripped)
    json_str = match.group(1).strip() if match else stripped
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in LLM response: {e}") from e


def parse_tool_calls(tool_calls: list) -> list[dict]:
    """Parse OpenAI tool_calls into a plain list of dicts."""
    result = []
    for tc in tool_calls:
        func = tc.function
        try:
            args = json.loads(func.arguments)
        except (json.JSONDecodeError, AttributeError):
            args = {}
        result.append({"name": func.name, "arguments": args, "id": tc.id})
    return result
