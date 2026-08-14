from __future__ import annotations

import logging

import httpx
from aiogram import Router
from aiogram.filters import BaseFilter, Command, CommandObject
from aiogram.types import Message

from app.bot.services.backend_client import BackendClient
from app.bot.services.streaming import friendly_error
from app.settings import settings

router = Router()
logger = logging.getLogger(__name__)

_BACKEND_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.HTTPStatusError)


class IsAdmin(BaseFilter):
    async def __call__(self, message: Message) -> bool:
        return message.from_user is not None and message.from_user.id in settings.bot.admin_ids


router.message.filter(IsAdmin())


def _fmt_number(value: float | None, suffix: str = "") -> str:
    return f"{value:.0f}{suffix}" if value is not None else "—"


def _fmt_ratio(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "—"


@router.message(Command("stats"))
async def cmd_stats(message: Message, backend: BackendClient) -> None:
    try:
        stats = await backend.get_admin_stats()
    except _BACKEND_ERRORS as exc:
        await message.answer(friendly_error(exc))
        return

    text = (
        "<b>Статистика за 24ч</b>\n"
        f"Сообщений: {stats['total_messages_24h']}\n"
        f"Активных пользователей: {stats['active_users_24h']}\n"
        f"Средняя латентность: {_fmt_number(stats['avg_latency_ms'], ' мс')}\n"
        f"Блокировок модерацией: {stats['moderation_block_rate']:.1%}\n"
        f"Доля 👍: {_fmt_ratio(stats['feedback_up_ratio'])}"
    )
    await message.answer(text)


@router.message(Command("users"))
async def cmd_users(message: Message, backend: BackendClient) -> None:
    try:
        users = await backend.get_admin_users(limit=50)
    except _BACKEND_ERRORS as exc:
        await message.answer(friendly_error(exc))
        return

    if not users:
        await message.answer("Пользователей пока нет.")
        return

    lines = ["<b>Последние пользователи</b>"]
    for user in users[:10]:
        lines.append(
            f"<code>{user['owner_external_id']}</code> — "
            f"{user['chats_count']} чат(ов), последний визит {user['last_seen_at']}"
        )
    await message.answer("\n".join(lines))


@router.message(Command("broadcast"))
async def cmd_broadcast(message: Message, command: CommandObject, backend: BackendClient) -> None:
    text = (command.args or "").strip()
    if not text:
        await message.answer("Использование: /broadcast &lt;текст сообщения&gt;")
        return

    try:
        result = await backend.post_admin_broadcast(text, interface_filter="telegram")
    except _BACKEND_ERRORS as exc:
        await message.answer(friendly_error(exc))
        return

    await message.answer(f"Рассылка поставлена в очередь (id={result['id']}).")
