from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    # Все столбцы datetime -> TIMESTAMPTZ: domain-модели (Chat, ChatMessage) всегда
    # передают tz-aware datetime.now(timezone.utc); без этого Base.metadata.create_all()
    # (тесты) создаёт naive TIMESTAMP, что не совпадает со схемой из alembic-миграций
    # и падает на INSERT с "can't subtract offset-naive and offset-aware datetimes".
    type_annotation_map = {datetime: DateTime(timezone=True)}


class ChatRow(Base):
    __tablename__ = "chats"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    owner_external_id: Mapped[str] = mapped_column(Text, nullable=False)
    interface: Mapped[str] = mapped_column(Text, nullable=False)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)


class ChatMessageRow(Base):
    __tablename__ = "chat_messages"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    chat_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    media_refs: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(nullable=True)

    __table_args__ = (
        Index(
            "ix_chat_messages_chat_created",
            "chat_id",
            text("created_at DESC"),
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )


class MessageFeedbackRow(Base):
    __tablename__ = "message_feedback"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    message_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    owner_external_id: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("owner_external_id", "message_id", name="uq_message_feedback_owner_message"),
    )


class ModerationIncidentRow(Base):
    __tablename__ = "moderation_incidents"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    chat_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
    )
    direction: Mapped[str] = mapped_column(Text, nullable=False)
    blocked_by: Mapped[str] = mapped_column(Text, nullable=False)
    categories: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)


class BroadcastQueueRow(Base):
    __tablename__ = "broadcast_queue"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    interface: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
    processed_at: Mapped[datetime | None] = mapped_column(nullable=True)
