from typing import Literal
from pydantic import BaseModel
from app.prompts.loader import load_tool_description

DESC_GET_PAYMENT_SYSTEM_STATUS = load_tool_description("get_payment_system_status")
DESC_CHECK_TRANSACTION_STATUS = load_tool_description("check_transaction_status")


class GetPaymentSystemStatusParams(BaseModel):
    component: Literal["api", "checkout", "dashboard", "payments", "payouts", "all"]


class CheckTransactionStatusParams(BaseModel):
    transaction_id: str

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_payment_system_status",
            "description": DESC_GET_PAYMENT_SYSTEM_STATUS,
            "parameters": {
                "type": "object",
                "properties": {
                    "component": {
                        "type": "string",
                        "enum": ["api", "checkout", "dashboard", "payments", "payouts", "all"],
                        "description": "Компонент системы для проверки статуса",
                    }
                },
                "required": ["component"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_transaction_status",
            "description": DESC_CHECK_TRANSACTION_STATUS,
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "Идентификатор транзакции, например TXN-1001",
                    }
                },
                "required": ["transaction_id"],
            },
        },
    },
]
