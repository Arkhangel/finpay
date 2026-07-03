"""Обратный канал backend -> bot: проактивные уведомления через /notify."""

from __future__ import annotations

import httpx

from app.settings import settings


async def notify_user(chat_id_tg: int, text: str) -> None:
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.post(
            f"{settings.bot.bot_url}/notify",
            json={"chat_id": chat_id_tg, "text": text},
            headers={"X-Internal-Token": settings.bot.internal_token.get_secret_value()},
        )
        r.raise_for_status()
