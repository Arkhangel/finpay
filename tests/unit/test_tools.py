"""Unit-тесты для app/tools/handlers.py и app/tools/schemas.py.

check_transaction_status использует SQLite — патчим DB_PATH на tmp_path
чтобы тесты были изолированы и не трогали data/transactions.db.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.tools.handlers import check_transaction_status, get_payment_system_status
from app.tools.schemas import CheckTransactionStatusParams, GetPaymentSystemStatusParams


# ── get_payment_system_status ────────────────────────────────────────────────

def test_get_payment_system_status_known_component():
    result = get_payment_system_status(component="api")
    assert result["component"] == "api"
    assert "status" in result
    assert result["status"] in ("operational", "degraded_performance", "partial_outage",
                                "major_outage", "under_maintenance")


def test_get_payment_system_status_all_returns_list():
    result = get_payment_system_status(component="all")
    assert "components" in result
    assert isinstance(result["components"], list)
    assert len(result["components"]) > 0


def test_get_payment_system_status_unknown_returns_unknown():
    # GetPaymentSystemStatusParams не допускает произвольные строки,
    # поэтому проверяем что Literal-валидация отсекает невалидный компонент
    with pytest.raises(ValidationError):
        GetPaymentSystemStatusParams(component="nonexistent")


# ── GetPaymentSystemStatusParams validation ──────────────────────────────────

def test_params_valid_component_accepted():
    for comp in ("api", "checkout", "dashboard", "payments", "payouts", "all"):
        p = GetPaymentSystemStatusParams(component=comp)
        assert p.component == comp


def test_params_invalid_component_raises():
    with pytest.raises(ValidationError):
        GetPaymentSystemStatusParams(component="billing")


# ── check_transaction_status ─────────────────────────────────────────────────

def test_check_transaction_status_found(mocker, tmp_path):
    """Транзакция из seed-данных возвращает полный словарь."""
    mocker.patch("app.tools.handlers.DB_PATH", tmp_path / "test.db")

    result = check_transaction_status(transaction_id="TXN-1001")

    assert result["transaction_id"] == "TXN-1001"
    assert result["status"] == "success"
    assert "amount" in result
    assert "currency" in result


def test_check_transaction_status_not_found(mocker, tmp_path):
    """Несуществующая транзакция возвращает словарь с ключом error."""
    mocker.patch("app.tools.handlers.DB_PATH", tmp_path / "test.db")

    result = check_transaction_status(transaction_id="TXN-XXXX")

    assert "error" in result


def test_check_transaction_status_failed_has_failure_reason(mocker, tmp_path):
    """Транзакция TXN-1002 (failed) содержит failure_reason."""
    mocker.patch("app.tools.handlers.DB_PATH", tmp_path / "test.db")

    result = check_transaction_status(transaction_id="TXN-1002")

    assert result["status"] == "failed"
    assert result["failure_reason"] is not None
