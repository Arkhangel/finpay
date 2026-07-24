"""message_feedback.sources — источники, показанные вместе с оценённым ответом

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-24

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "message_feedback",
        sa.Column("sources", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("message_feedback", "sources")
