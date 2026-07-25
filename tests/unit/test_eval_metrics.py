"""Unit-тест для app/eval/metrics.py::make_has_citation — единственная
чистая функция в модуле (остальное требует живого judge LLM)."""
from __future__ import annotations

from app.eval.metrics import make_has_citation


def test_has_citation_true_with_marker():
    assert make_has_citation("Комиссия 1% от суммы [1].") is True


def test_has_citation_true_multiple_markers():
    assert make_has_citation("Лимит 100000 [1], срок 3 дня [2].") is True


def test_has_citation_false_without_marker():
    assert make_has_citation("Комиссия 1% от суммы.") is False


def test_has_citation_false_on_refusal():
    assert make_has_citation("По базе не нашёл, могу эскалировать.") is False
