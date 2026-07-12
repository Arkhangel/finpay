"""Unit-тесты для app/services/embeddings.py: батчинг, кеш, E5-префиксы, инвалидация по модели.

Мокается app.services.embeddings._get_model — модель не грузится и не скачивается.
Кеш — временный diskcache.Cache в tmp_path.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import diskcache
import pytest

from app.services import embeddings


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    test_cache = diskcache.Cache(str(tmp_path / "emb_cache"))
    monkeypatch.setattr(embeddings, "cache", test_cache)
    yield test_cache
    test_cache.close()


def _mock_model(vectors: list[list[float]]) -> MagicMock:
    model = MagicMock()
    model.encode.return_value = vectors
    return model


def test_embed_texts_calls_model_on_cache_miss(mocker):
    mock_model = _mock_model([[0.1, 0.2], [0.3, 0.4]])
    mocker.patch.object(embeddings, "_get_model", return_value=mock_model)

    result = embeddings.embed_texts(["привет", "пока"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    mock_model.encode.assert_called_once()
    assert mock_model.encode.call_args[1]["normalize_embeddings"] is True


def test_embed_texts_second_call_hits_cache_not_model(mocker):
    mock_model = _mock_model([[0.1, 0.2]])
    mock_get_model = mocker.patch.object(embeddings, "_get_model", return_value=mock_model)

    embeddings.embed_texts(["тот же текст"])
    embeddings.embed_texts(["тот же текст"])

    mock_model.encode.assert_called_once()
    mock_get_model.assert_called_once()


def test_embed_texts_empty_list_returns_empty_without_loading_model(mocker):
    mock_get_model = mocker.patch.object(embeddings, "_get_model")

    assert embeddings.embed_texts([]) == []
    mock_get_model.assert_not_called()


def test_embed_query_adds_query_prefix(mocker):
    mock_model = _mock_model([[0.5, 0.5]])
    mocker.patch.object(embeddings, "_get_model", return_value=mock_model)

    result = embeddings.embed_query("какая комиссия?")

    encoded_texts = mock_model.encode.call_args[0][0]
    assert encoded_texts == ["query: какая комиссия?"]
    assert result == [0.5, 0.5]


def test_embed_documents_adds_passage_prefix(mocker):
    mock_model = _mock_model([[0.1], [0.2]])
    mocker.patch.object(embeddings, "_get_model", return_value=mock_model)

    result = embeddings.embed_documents(["текст 1", "текст 2"])

    encoded_texts = mock_model.encode.call_args[0][0]
    assert encoded_texts == ["passage: текст 1", "passage: текст 2"]
    assert result == [[0.1], [0.2]]


def test_embed_texts_uses_configured_batch_size(mocker, monkeypatch):
    monkeypatch.setattr(embeddings.settings.embeddings, "batch_size", 16)
    mock_model = _mock_model([[0.0], [0.1]])
    mocker.patch.object(embeddings, "_get_model", return_value=mock_model)

    embeddings.embed_texts(["a", "b"])

    assert mock_model.encode.call_args[1]["batch_size"] == 16


def test_cache_key_changes_with_model(monkeypatch):
    monkeypatch.setattr(embeddings.settings.embeddings, "model", "intfloat/multilingual-e5-small")
    key_small = embeddings._cache_key("текст")
    monkeypatch.setattr(embeddings.settings.embeddings, "model", "intfloat/multilingual-e5-base")
    key_base = embeddings._cache_key("текст")

    assert key_small != key_base