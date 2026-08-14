from __future__ import annotations

from typing import Protocol
from uuid import UUID

from app.chat.domain import Chat, ChatMessage


class ChatRepository(Protocol):
    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat: ...

    async def get_chat(self, chat_id: UUID) -> Chat | None: ...

    async def append_message(self, chat_id: UUID, message: ChatMessage) -> ChatMessage: ...

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]: ...

    async def message_exists(self, chat_id: UUID, message_id: UUID) -> bool:
        """True if message_id belongs to chat_id — used to validate feedback
        before it hits save_feedback (см. app/chat/routes.py::submit_feedback)."""
        ...

    async def soft_delete_messages(self, chat_id: UUID) -> None: ...

    async def save_feedback(
        self,
        message_id: UUID,
        owner_external_id: str,
        value: str,
        sources: list[dict] | None = None,
    ) -> bool:
        """Returns True if newly recorded, False if this (owner, message) pair already voted.

        `sources` — список источников, показанных пользователю вместе с этим
        ответом (Б5.5), для аудита "какой ответ с какими source получил дизлайк".
        """
        ...

    async def record_moderation_incident(
        self, chat_id: UUID, direction: str, blocked_by: str, categories: list[str]
    ) -> None: ...
