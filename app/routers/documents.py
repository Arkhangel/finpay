"""POST /documents/upload — инкрементальная загрузка документа (блок 5.5).

Файл сохраняется в data/<category>/, индексация (тот же scripts/ingest.py,
DocstoreStrategy.UPSERTS) запускается в фоне через BackgroundTasks — запрос
не блокируется на время эмбеддинга/апсерта в Qdrant. Коллекция общая с
RAGService, поэтому документ становится доступен в /rag/query и /chats/{id}/
messages сразу после завершения фоновой задачи, без перезапуска сервиса.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

_DATA_ROOT = Path("data")


def _run_ingest_background(file_path: Path) -> None:
    from scripts.ingest import ingest_files

    try:
        result = ingest_files([file_path], _DATA_ROOT)
        logger.info(
            "document_upload_indexed",
            extra={
                "path": str(file_path),
                "changed": result["changed"],
                "unchanged": result["unchanged"],
                "failed": len(result["failed"]),
            },
        )
    except Exception:
        logger.exception("document_upload_indexing_failed", extra={"path": str(file_path)})


@router.post("/upload", status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form("uploads"),
) -> dict:
    from scripts.ingest import SUPPORTED_EXTENSIONS

    # Path(...).name отбрасывает директории из имени — без этого "../../"
    # в filename/category позволил бы записать файл вне data/ (path traversal).
    safe_name = Path(file.filename or "").name
    suffix = Path(safe_name).suffix.lower()
    if not safe_name or suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Неподдерживаемый формат {suffix!r}. Ожидается один из {SUPPORTED_EXTENSIONS}.",
        )

    safe_category = Path(category).name or "uploads"
    target_dir = _DATA_ROOT / safe_category
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / safe_name

    content = await file.read()
    target_path.write_bytes(content)

    background_tasks.add_task(_run_ingest_background, target_path)

    return {"status": "accepted", "path": str(target_path), "category": safe_category}
