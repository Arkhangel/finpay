"""Unit-тесты для app/eval/metrics.py::make_has_citation — LLM-judged
@discrete_metric (критерий 3 чекпоинта 5), judge мокается (без реального
API-вызова)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.eval.metrics import _CitationVerdict, make_has_citation


def _fake_llm(has_citation: bool):
    llm = AsyncMock()
    llm.agenerate.return_value = _CitationVerdict(has_citation=has_citation)
    return llm


@pytest.mark.asyncio
async def test_has_citation_true_when_judge_says_yes():
    metric = make_has_citation(_fake_llm(True))
    result = await metric.ascore(user_input="q", response="Комиссия 1% от суммы [1].")
    assert result.value == "yes"


@pytest.mark.asyncio
async def test_has_citation_false_when_judge_says_no():
    metric = make_has_citation(_fake_llm(False))
    result = await metric.ascore(user_input="q", response="Комиссия 1% от суммы.")
    assert result.value == "no"


@pytest.mark.asyncio
async def test_has_citation_calls_judge_with_response_text():
    llm = _fake_llm(True)
    metric = make_has_citation(llm)
    await metric.ascore(user_input="q", response="согласно нашей политике возвратов")
    prompt = llm.agenerate.call_args.args[0]
    assert "согласно нашей политике возвратов" in prompt
