import sqlite3
from pathlib import Path
from app.tools.schemas import GetPaymentSystemStatusParams, CheckTransactionStatusParams

# Путь к тестовой БД
DB_PATH = Path(__file__).parent.parent.parent / "data" / "transactions.db"


# --- get_payment_system_status ---

_MOCK_STATUS = {
    "api": "operational",
    "checkout": "operational",
    "dashboard": "operational",
    "payments": "operational",
    "payouts": "degraded_performance",
}

def get_payment_system_status(component: str) -> dict:
    params = GetPaymentSystemStatusParams(component=component)
    if params.component == "all":
        return {
            "overall": "Mock-данные",
            "components": [{"component": k, "status": v} for k, v in _MOCK_STATUS.items()],
        }
    status = _MOCK_STATUS.get(params.component, "unknown")
    return {
        "component": params.component,
        "status": status,
        "description": _status_label(status),
        "source": "mock",
    }


def _status_label(status: str) -> str:
    labels = {
        "operational": "Работает в штатном режиме",
        "degraded_performance": "Снижение производительности",
        "partial_outage": "Частичный сбой",
        "major_outage": "Серьёзный сбой",
        "under_maintenance": "Техническое обслуживание",
    }
    return labels.get(status, status)


# --- check_transaction_status ---
# Реальный источник: локальная SQLite с тестовыми данными

def check_transaction_status(transaction_id: str) -> dict:
    params = CheckTransactionStatusParams(transaction_id=transaction_id)
    if not DB_PATH.exists():
        _init_db()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM transactions WHERE transaction_id = ?",
            (params.transaction_id,)
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return {"error": f"Транзакция {params.transaction_id} не найдена"}

    return {
        "transaction_id": row["transaction_id"],
        "status": row["status"],
        "amount": row["amount"],
        "currency": row["currency"],
        "created_at": row["created_at"],
        "failure_reason": row["failure_reason"],
    }


def _init_db():
    """Создаёт тестовую БД при первом запуске."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            status TEXT,
            amount REAL,
            currency TEXT,
            created_at TEXT,
            failure_reason TEXT
        )
    """)
    test_data = [
        ("TXN-1001", "success",    1500.00, "RUB", "2025-06-01 10:23:00", None),
        ("TXN-1002", "failed",      899.00, "RUB", "2025-06-01 11:05:00", "Недостаточно средств"),
        ("TXN-1003", "pending",    3200.00, "RUB", "2025-06-01 12:44:00", None),
        ("TXN-1004", "failed",      450.00, "RUB", "2025-06-01 13:10:00", "Карта заблокирована"),
        ("TXN-1005", "success",   10000.00, "RUB", "2025-06-01 14:00:00", None),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO transactions VALUES (?,?,?,?,?,?)",
        test_data
    )
    conn.commit()
    conn.close()
