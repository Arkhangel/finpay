from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.domain import Chat, ChatMessage
from app.chat.repositories.pg_models import ChatMessageRow, ChatRow


class PostgresChatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        chat = Chat(
            owner_external_id=owner_external_id,
            interface=interface,
            system_prompt=system_prompt,
        )
        row = ChatRow(
            id=chat.id,
            owner_external_id=chat.owner_external_id,
            interface=chat.interface,
            system_prompt=chat.system_prompt,
            created_at=chat.created_at,
        )
        self._session.add(row)
        await self._session.commit()
        return chat

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        result = await self._session.execute(
            select(ChatRow).where(ChatRow.id == chat_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return Chat.model_validate(row, from_attributes=True)

    async def append_message(self, chat_id: UUID, message: ChatMessage) -> ChatMessage:
        row = ChatMessageRow(
            id=message.id,
            chat_id=chat_id,
            role=message.role,
            content=message.content,
            tokens=message.tokens,
            created_at=message.created_at,
        )
        self._session.add(row)
        await self._session.commit()
        return message

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        result = await self._session.execute(
            select(ChatMessageRow)
            .where(
                ChatMessageRow.chat_id == chat_id,
                ChatMessageRow.deleted_at.is_(None),
            )
            .order_by(ChatMessageRow.created_at.desc())
            .limit(limit)
        )
        rows = result.scalars().all()
        # Return in chronological order (oldest first)
        messages = [ChatMessage.model_validate(row, from_attributes=True) for row in reversed(rows)]
        return messages

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        now = datetime.now(timezone.utc)
        await self._session.execute(
            update(ChatMessageRow)
            .where(
                ChatMessageRow.chat_id == chat_id,
                ChatMessageRow.deleted_at.is_(None),
            )
            .values(deleted_at=now)
        )
        await self._session.commit()
