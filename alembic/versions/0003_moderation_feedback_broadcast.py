"""moderation incidents, message feedback, broadcast queue, latency_ms

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-03

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_messages", sa.Column("latency_ms", sa.Integer(), nullable=True))

    op.create_table(
        "message_feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("owner_external_id", sa.Text(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["message_id"], ["chat_messages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "owner_external_id", "message_id", name="uq_message_feedback_owner_message"
        ),
    )

    op.create_table(
        "moderation_incidents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chat_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("direction", sa.Text(), nullable=False),
        sa.Column("blocked_by", sa.Text(), nullable=False),
        sa.Column("categories", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["chat_id"], ["chats.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "broadcast_queue",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("interface", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("processed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("broadcast_queue")
    op.drop_table("moderation_incidents")
    op.drop_table("message_feedback")
    op.drop_column("chat_messages", "latency_ms")
