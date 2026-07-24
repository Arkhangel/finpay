"""In-memory кэш sources по message_id (блок 5.5).

Заполняется после стриминга ответа (app/bot/handlers/text.py), читается при
нажатии кнопки фидбека (app/bot/handlers/feedback.py) и отправляется вместе
с оценкой на POST /chats/{id}/messages/{mid}/feedback. Список источников не
помещается в Telegram callback_data (лимит ~64 байта на кнопку), поэтому
передаётся боту отдельно, а не через callback-кнопку.
"""

from __future__ import annotations

from uuid import UUID

# Корпус — сотни пользователей одновременно максимум (учебный проект), FIFO-
# ограничение на размер защищает от неограниченного роста, если фидбек так и
# не был нажат ни разу для части сообщений.
_MAX_ENTRIES = 1000
_sources: dict[UUID, list[dict]] = {}


def remember(message_id: UUID, sources: list[dict] | None) -> None:
    if sources is None:
        return
    if len(_sources) >= _MAX_ENTRIES:
        _sources.pop(next(iter(_sources)))
    _sources[message_id] = sources


def pop(message_id: UUID) -> list[dict] | None:
    return _sources.pop(message_id, None)
