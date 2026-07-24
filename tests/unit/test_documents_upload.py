"""Tests for POST /documents/upload (блок 5.5).

scripts.ingest.ingest_files мокается — реальный эмбеддинг/Qdrant здесь не
нужны, фокус на контракте эндпоинта: сохранение файла, 202, валидация
формата, защита от path traversal в filename/category, вызов фоновой задачи.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

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


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app.routers.documents as documents_module

    monkeypatch.setattr(documents_module, "_DATA_ROOT", tmp_path)

    from app.main import create_app

    with patch("app.main.lifespan", _noop_lifespan):
        app = create_app()

    with TestClient(app) as c:
        yield c


@pytest.fixture
def fake_ingest(monkeypatch):
    calls = []

    def _fake(files, base_dir):
        calls.append((files, base_dir))
        return {"changed": 1, "unchanged": 0, "nodes_upserted": 2, "failed": []}

    monkeypatch.setattr("scripts.ingest.ingest_files", _fake)
    return calls


def test_upload_document_saves_file_and_returns_202(client, fake_ingest, tmp_path):
    resp = client.post(
        "/documents/upload",
        data={"category": "tariffs"},
        files={"file": ("new_fees.md", b"# Fees\ncomission 2%", "text/markdown")},
    )

    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["category"] == "tariffs"

    saved_path = tmp_path / "tariffs" / "new_fees.md"
    assert saved_path.read_bytes() == b"# Fees\ncomission 2%"


def test_upload_document_triggers_background_ingestion(client, fake_ingest, tmp_path):
    client.post(
        "/documents/upload",
        data={"category": "tariffs"},
        files={"file": ("new_fees.md", b"content", "text/markdown")},
    )

    assert len(fake_ingest) == 1
    files_arg, base_dir_arg = fake_ingest[0]
    assert files_arg == [tmp_path / "tariffs" / "new_fees.md"]
    assert base_dir_arg == tmp_path


def test_upload_document_rejects_unsupported_extension(client, fake_ingest):
    resp = client.post(
        "/documents/upload",
        data={"category": "tariffs"},
        files={"file": ("archive.zip", b"whatever", "application/zip")},
    )

    assert resp.status_code == 415
    assert not fake_ingest


def test_upload_document_sanitizes_filename_path_traversal(client, fake_ingest, tmp_path):
    resp = client.post(
        "/documents/upload",
        data={"category": "tariffs"},
        files={"file": ("../../evil.md", b"payload", "text/markdown")},
    )

    assert resp.status_code == 202
    escaped_path = tmp_path.parent.parent / "evil.md"
    assert not escaped_path.exists()
    assert (tmp_path / "tariffs" / "evil.md").read_bytes() == b"payload"


def test_upload_document_sanitizes_category_path_traversal(client, fake_ingest, tmp_path):
    resp = client.post(
        "/documents/upload",
        data={"category": "../../etc"},
        files={"file": ("note.md", b"payload", "text/markdown")},
    )

    assert resp.status_code == 202
    assert resp.json()["category"] == "etc"
    assert (tmp_path / "etc" / "note.md").read_bytes() == b"payload"


def test_upload_document_defaults_category_when_omitted(client, fake_ingest, tmp_path):
    resp = client.post(
        "/documents/upload",
        files={"file": ("note.md", b"payload", "text/markdown")},
    )

    assert resp.status_code == 202
    assert resp.json()["category"] == "uploads"
    assert (tmp_path / "uploads" / "note.md").read_bytes() == b"payload"
