from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.schemas import BroadcastPendingItem, StatsOut, UserOut
from app.chat.repositories.pg_models import (
    BroadcastQueueRow,
    ChatMessageRow,
    ChatRow,
    MessageFeedbackRow,
    ModerationIncidentRow,
)

# Интервал считается в Postgres (func.now() - ...), а не в Python: created_at в
# ORM-моделях объявлен как naive DateTime, а в alembic-миграции — как
# TIMESTAMPTZ; сравнение с tz-aware datetime из Python упадёт на одной из схем.
_LAST_24H = func.now() - text("interval '24 hours'")


class AdminRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_stats(self) -> StatsOut:
        total_messages = await self._session.scalar(
            select(func.count())
            .select_from(ChatMessageRow)
            .where(ChatMessageRow.created_at >= _LAST_24H, ChatMessageRow.deleted_at.is_(None))
        ) or 0

        active_users = await self._session.scalar(
            select(func.count(func.distinct(ChatRow.owner_external_id)))
            .select_from(ChatMessageRow)
            .join(ChatRow, ChatRow.id == ChatMessageRow.chat_id)
            .where(ChatMessageRow.created_at >= _LAST_24H)
        ) or 0

        avg_latency = await self._session.scalar(
            select(func.avg(ChatMessageRow.latency_ms)).where(
                ChatMessageRow.role == "assistant",
                ChatMessageRow.latency_ms.is_not(None),
                ChatMessageRow.created_at >= _LAST_24H,
            )
        )

        incidents_24h = await self._session.scalar(
            select(func.count())
            .select_from(ModerationIncidentRow)
            .where(ModerationIncidentRow.created_at >= _LAST_24H)
        ) or 0

        feedback_row = (
            await self._session.execute(
                select(
                    func.count().filter(MessageFeedbackRow.value == "up").label("up"),
                    func.count().label("total"),
                )
            )
        ).one()

        total_attempts = total_messages + incidents_24h

        return StatsOut(
            total_messages_24h=total_messages,
            active_users_24h=active_users,
            avg_latency_ms=float(avg_latency) if avg_latency is not None else None,
            moderation_block_rate=round(incidents_24h / total_attempts, 4) if total_attempts else 0.0,
            feedback_up_ratio=round(feedback_row.up / feedback_row.total, 4) if feedback_row.total else None,
        )

    async def list_users(self, limit: int = 50) -> list[UserOut]:
        last_seen_at = func.max(func.coalesce(ChatMessageRow.created_at, ChatRow.created_at))
        stmt = (
            select(
                ChatRow.owner_external_id,
                ChatRow.interface,
                func.count(func.distinct(ChatRow.id)).label("chats_count"),
                last_seen_at.label("last_seen_at"),
            )
            .select_from(ChatRow)
            .outerjoin(ChatMessageRow, ChatMessageRow.chat_id == ChatRow.id)
            .group_by(ChatRow.owner_external_id, ChatRow.interface)
            .order_by(last_seen_at.desc())
            .limit(limit)
        )
        rows = (await self._session.execute(stmt)).all()

        return [
            UserOut(
                owner_external_id=row.owner_external_id,
                interface=row.interface,
                chats_count=row.chats_count,
                last_seen_at=row.last_seen_at,
            )
            for row in rows
        ]

    async def create_broadcast(self, message: str, interface: str) -> UUID:
        broadcast_id = uuid4()
        self._session.add(
            BroadcastQueueRow(id=broadcast_id, message=message, interface=interface, status="pending")
        )
        await self._session.commit()
        return broadcast_id

    async def list_pending_broadcasts(self, limit: int = 20) -> list[BroadcastPendingItem]:
        stmt = (
            select(BroadcastQueueRow)
            .where(BroadcastQueueRow.status == "pending")
            .order_by(BroadcastQueueRow.created_at)
            .limit(limit)
        )
        pending = (await self._session.execute(stmt)).scalars().all()

        items: list[BroadcastPendingItem] = []
        for row in pending:
            owner_ids = (
                await self._session.execute(
                    select(ChatRow.owner_external_id).distinct().where(ChatRow.interface == row.interface)
                )
            ).scalars().all()
            targets = [int(owner_id) for owner_id in owner_ids if owner_id.lstrip("-").isdigit()]
            items.append(BroadcastPendingItem(id=row.id, message=row.message, targets=targets))

        return items

    async def ack_broadcast(self, broadcast_id: UUID, status: str) -> None:
        await self._session.execute(
            update(BroadcastQueueRow)
            .where(BroadcastQueueRow.id == broadcast_id)
            .values(status=status, processed_at=func.now())
        )
        await self._session.commit()
