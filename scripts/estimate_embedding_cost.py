"""Оценка стоимости индексации базы знаний — блок 5.1.

Self-hosted intfloat/multilingual-e5-base не тарифицируется по токенам:
$-стоимость — 0, реальная "цена" индексации — время CPU-инференса. Скрипт
прогоняет модель на реальном корпусе проекта и экстраполирует время на 50/100
документов сравнимого размера.

    uv run scripts/estimate_embedding_cost.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.embeddings import _get_model
from app.settings import settings

ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILES = [
    ROOT / "app/prompts/service_facts.j2",
    ROOT / "app/prompts/tools/check_transaction_status.md",
    ROOT / "app/prompts/tools/get_payment_system_status.md",
    ROOT / "docs/chat.md",
    ROOT / "docs/architecture.md",
    ROOT / "README.md",
]


def main() -> None:
    texts = [p.read_text(encoding="utf-8") for p in SOURCE_FILES if p.exists()]
    passages = [f"passage: {t}" for t in texts]

    model = _get_model()  # разовая загрузка модели — не входит в замер индексации

    start = time.perf_counter()
    model.encode(passages, batch_size=settings.embeddings.batch_size, normalize_embeddings=True)
    elapsed = time.perf_counter() - start

    per_doc = elapsed / len(texts)

    print(f"Модель: {settings.embeddings.model} (self-hosted, device={settings.embeddings.device})")
    print(f"Документов в выборке: {len(texts)}")
    print(f"Время индексации выборки: {elapsed:.2f} с ({per_doc:.3f} с/документ)\n")

    for n_docs in (50, 100):
        est_seconds = per_doc * n_docs
        print(f"{n_docs} документов ~ {est_seconds:.1f} с CPU-инференса, $0 стоимости API")


if __name__ == "__main__":
    main()