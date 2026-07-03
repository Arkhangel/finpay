"""Фоновый воркер: разбирает broadcast_queue на бэкенде и рассылает сообщения сам."""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot

from app.bot.services.backend_client import BackendClient

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 10.0


async def run_broadcast_worker(bot: Bot, backend: BackendClient) -> None:
    while True:
        try:
            pending = await backend.get_pending_broadcasts()
        except Exception:
            logger.exception("broadcast_poll_failed")
            await asyncio.sleep(_POLL_INTERVAL)
            continue

        for item in pending:
            sent = 0
            for chat_id in item["targets"]:
                try:
                    await bot.send_message(chat_id=chat_id, text=item["message"])
                    sent += 1
                except Exception:
                    logger.warning(
                        "broadcast_send_failed broadcast_id=%s chat_id=%s", item["id"], chat_id
                    )

            status = "sent" if sent or not item["targets"] else "failed"
            try:
                await backend.ack_broadcast(item["id"], status)
            except Exception:
                logger.exception("broadcast_ack_failed broadcast_id=%s", item["id"])

        await asyncio.sleep(_POLL_INTERVAL)
