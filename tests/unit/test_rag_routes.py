"""Integration tests for POST /rag/query (блок 5.5 response contract)."""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@asynccontextmanager
async def _noop_lifespan(app):
    app.state.openai = MagicMock()
    app.state.cache = None
    app.state.pg_engine = None
    app.state.pg_session_factory = None
    app.state.canary = "test_canary"
    yield


def _client(rag_service):
    from app.main import create_app
    from app.deps.providers import get_rag_service

    with patch("app.main.lifespan", _noop_lifespan):
        app = create_app()

    app.dependency_overrides[get_rag_service] = lambda: rag_service
    return TestClient(app)


def test_rag_query_returns_citations_confident_and_sources():
    rag_service = MagicMock()
    rag_service.answer = AsyncMock(
        return_value={
            "answer": "Возврат оформляется в течение 30 дней [1].",
            "top_score": 0.9,
            "confident": True,
            "sources": [
                {"id": 1, "file_name": "05_refunds.md", "page": 1, "score": 0.9, "snippet": "..."}
            ],
        }
    )

    with _client(rag_service) as client:
        resp = client.post("/rag/query", json={"question": "Каков срок возврата?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["confident"] is True
    assert body["answer"] == "Возврат оформляется в течение 30 дней [1]."
    assert body["sources"] == [
        {"id": 1, "file_name": "05_refunds.md", "page": 1, "score": 0.9, "snippet": "..."}
    ]


def test_rag_query_refusal_when_not_confident():
    from app.services.rag import REFUSAL_ANSWER

    rag_service = MagicMock()
    rag_service.answer = AsyncMock(
        return_value={
            "answer": REFUSAL_ANSWER,
            "top_score": 0.1,
            "confident": False,
            "sources": [],
        }
    )

    with _client(rag_service) as client:
        resp = client.post("/rag/query", json={"question": "какой рецепт борща?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["confident"] is False
    assert body["answer"] == REFUSAL_ANSWER


def test_rag_query_returns_503_when_service_unavailable():
    with _client(None) as client:
        resp = client.post("/rag/query", json={"question": "вопрос"})

    assert resp.status_code == 503


def test_rag_query_rejects_empty_question():
    rag_service = MagicMock()
    rag_service.answer = AsyncMock()

    with _client(rag_service) as client:
        resp = client.post("/rag/query", json={"question": ""})

    assert resp.status_code == 422
    rag_service.answer.assert_not_called()
