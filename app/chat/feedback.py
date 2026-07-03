from __future__ import annotations

from uuid import UUID

from app.chat.repository import ChatRepository


async def save_feedback(
    repo: ChatRepository, message_id: UUID, owner_external_id: str, value: str
) -> bool:
    """Returns True if newly recorded, False if this (owner, message) pair already voted."""
    return await repo.save_feedback(message_id, owner_external_id, value)
