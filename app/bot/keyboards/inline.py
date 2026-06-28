from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import InlineKeyboardMarkup

# FinPay topics — payment gateway support domain
_TOPICS = [
    ("Биллинг", "billing"),
    ("Транзакции", "transactions"),
    ("Ошибки API", "api_errors"),
    ("Интеграция", "integration"),
    ("Возвраты", "refunds"),
]


def topics_kb() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for label, slug in _TOPICS:
        builder.button(text=label, callback_data=f"topic:{slug}")
    builder.button(text="Отмена", callback_data="topic:cancel")
    builder.adjust(2)
    return builder.as_markup()
