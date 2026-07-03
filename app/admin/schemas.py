from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class StatsOut(BaseModel):
    total_messages_24h: int
    active_users_24h: int
    avg_latency_ms: float | None
    moderation_block_rate: float
    feedback_up_ratio: float | None


class UserOut(BaseModel):
    owner_external_id: str
    interface: str
    chats_count: int
    last_seen_at: datetime


class BroadcastIn(BaseModel):
    message: str
    interface_filter: str = "telegram"


class BroadcastOut(BaseModel):
    id: UUID
    status: str


class BroadcastPendingItem(BaseModel):
    id: UUID
    message: str
    targets: list[int]


class BroadcastAckIn(BaseModel):
    status: Literal["sent", "failed"]
