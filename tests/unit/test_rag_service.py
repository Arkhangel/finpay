"""Unit-тесты для app/services/rag.py::RAGService.answer() (блок 5.5).

Мокаются retriever (self._index.as_retriever(...).aretrieve), реранкер и
service._llm.achat — реальные Qdrant/HuggingFace/LLM не поднимаются.
Фокус — на контракте: score-guard должен сработать ДО вызова LLM, реранкер
подключается только если включён, sources собираются с нужными полями.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from app.services.rag import REFUSAL_ANSWER, RAGService


def _node(text: str, score: float, **metadata) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=text, metadata=metadata), score=score)


def _make_service(settings, reranker=None, llm=None) -> RAGService:
    service = RAGService(settings=settings)
    service._reranker = reranker
    service._llm = llm or _mock_llm("ответ")
    return service


def _mock_llm(content: str) -> MagicMock:
    response = MagicMock()
    response.message.content = content
    llm = MagicMock()
    llm.achat = AsyncMock(return_value=response)
    return llm


@pytest.fixture
def rag_settings(monkeypatch):
    from app.settings import settings as app_settings

    monkeypatch.setattr(app_settings.rag, "similarity_top_k", 10)
    monkeypatch.setattr(app_settings.rag, "rerank_top_n", 5)
    monkeypatch.setattr(app_settings.rag, "score_threshold", 0.75)
    monkeypatch.setattr(app_settings.rag, "reranker_enabled", False)
    return app_settings


def _mock_index(nodes: list[NodeWithScore]) -> MagicMock:
    retriever = MagicMock()
    retriever.aretrieve = AsyncMock(return_value=nodes)
    index = MagicMock()
    index.as_retriever.return_value = retriever
    return index


async def test_score_guard_skips_llm_call(rag_settings):
    nodes = [_node("нерелевантный текст", 0.2, source="01_off_topic.md")]
    mock_llm = _mock_llm("не должно вызваться")
    service = _make_service(rag_settings, llm=mock_llm)
    service._index = _mock_index(nodes)

    result = await service.answer("вопрос не по теме")

    mock_llm.achat.assert_not_called()
    assert result["answer"] == REFUSAL_ANSWER
    assert result["confident"] is False
    assert result["top_score"] == 0.2


async def test_confident_answer_calls_llm_with_numbered_context(rag_settings):
    nodes = [
        _node("Возврат оформляется в течение 30 дней.", 0.9, source="05_refunds.md", page=1),
        _node("Комиссия 1.8%.", 0.85, source="02_tariffs.md", page=None),
    ]
    mock_llm = _mock_llm("Возврат — 30 дней [1].")
    service = _make_service(rag_settings, llm=mock_llm)
    service._index = _mock_index(nodes)

    result = await service.answer("Каков срок возврата?")

    mock_llm.achat.assert_called_once()
    sent_messages = mock_llm.achat.call_args[0][0]
    user_message = sent_messages[-1].content
    assert "[1] (05_refunds.md)" in user_message
    assert "[2] (02_tariffs.md)" in user_message

    assert result["answer"] == "Возврат — 30 дней [1]."
    assert result["confident"] is True
    assert result["top_score"] == 0.9
    assert result["sources"] == [
        {"id": 1, "file_name": "05_refunds.md", "page": 1, "score": 0.9, "snippet": "Возврат оформляется в течение 30 дней."},
        {"id": 2, "file_name": "02_tariffs.md", "page": None, "score": 0.85, "snippet": "Комиссия 1.8%."},
    ]


async def test_reranker_reorders_and_truncates_candidates(rag_settings):
    nodes = [_node(f"текст {i}", 0.9 - i * 0.01, source=f"{i}.md") for i in range(10)]
    reordered = list(reversed(nodes))[:5]

    reranker = MagicMock()
    reranker.rerank.return_value = reordered

    service = _make_service(rag_settings, reranker=reranker)
    service._index = _mock_index(nodes)

    result = await service.answer("вопрос")

    reranker.rerank.assert_called_once_with("вопрос", nodes, top_n=5)
    assert len(result["sources"]) == 5
    assert result["sources"][0]["file_name"] == reordered[0].node.metadata["source"]


async def test_reranker_not_called_when_disabled(rag_settings):
    nodes = [_node("текст", 0.9, source="01.md")]
    service = _make_service(rag_settings, reranker=None)
    service._index = _mock_index(nodes)

    result = await service.answer("вопрос")

    assert len(result["sources"]) == 1


async def test_answer_before_build_raises() -> None:
    service = RAGService.__new__(RAGService)
    service._index = None

    with pytest.raises(RuntimeError):
        await service.answer("вопрос")
