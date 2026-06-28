from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import aiofiles
import aiofiles.os

from app.chat.domain import Chat, ChatMessage

_SOFT_DELETE_TYPE = "soft_delete"


class JsonChatRepository:
    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir

    def _chat_dir(self, chat_id: UUID) -> Path:
        return self._base / "chats" / str(chat_id)

    def _chat_meta_path(self, chat_id: UUID) -> Path:
        return self._chat_dir(chat_id) / "chat.json"

    def _messages_path(self, chat_id: UUID) -> Path:
        return self._chat_dir(chat_id) / "messages.jsonl"

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
        chat_dir = self._chat_dir(chat.id)
        await aiofiles.os.makedirs(chat_dir, exist_ok=True)

        async with aiofiles.open(self._chat_meta_path(chat.id), "w") as f:
            await f.write(chat.model_dump_json())

        return chat

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        path = self._chat_meta_path(chat_id)
        try:
            async with aiofiles.open(path) as f:
                data = await f.read()
            return Chat.model_validate_json(data)
        except (FileNotFoundError, OSError):
            return None

    async def append_message(self, chat_id: UUID, message: ChatMessage) -> ChatMessage:
        path = self._messages_path(chat_id)
        await aiofiles.os.makedirs(path.parent, exist_ok=True)
        async with aiofiles.open(path, "a") as f:
            await f.write(message.model_dump_json() + "\n")
        return message

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        path = self._messages_path(chat_id)
        try:
            async with aiofiles.open(path) as f:
                lines = await f.readlines()
        except (FileNotFoundError, OSError):
            return []

        # Find the last soft-delete marker and skip everything before it
        last_delete_idx = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
                if obj.get("type") == _SOFT_DELETE_TYPE:
                    last_delete_idx = i
            except (json.JSONDecodeError, AttributeError):
                continue

        messages: list[ChatMessage] = []
        for line in lines[last_delete_idx + 1 :]:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
                if obj.get("type") == _SOFT_DELETE_TYPE:
                    continue
                messages.append(ChatMessage.model_validate_json(stripped))
            except (json.JSONDecodeError, Exception):
                continue

        # Return last N in chronological order
        return messages[-limit:]

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        path = self._messages_path(chat_id)
        await aiofiles.os.makedirs(path.parent, exist_ok=True)
        marker = json.dumps({"type": _SOFT_DELETE_TYPE, "at": datetime.now(timezone.utc).isoformat()})
        async with aiofiles.open(path, "a") as f:
            await f.write(marker + "\n")
