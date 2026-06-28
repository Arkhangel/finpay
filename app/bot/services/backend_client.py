from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)


class BackendClient:
    def __init__(self, base_url: str) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=30)
        # Local cache: (owner_external_id, interface) -> chat_id
        self._chat_cache: dict[tuple[str, str], UUID] = {}

    async def get_or_create_chat(self, owner_external_id: str, interface: str) -> UUID:
        key = (owner_external_id, interface)
        if key in self._chat_cache:
            return self._chat_cache[key]
        resp = await self._client.post(
            "/chats",
            json={"owner_external_id": owner_external_id, "interface": interface},
        )
        resp.raise_for_status()
        chat_id = UUID(resp.json()["chat_id"])
        self._chat_cache[key] = chat_id
        logger.info("created_chat owner=%s chat_id=%s", owner_external_id, chat_id)
        return chat_id

    async def send_message(
        self, chat_id: UUID, content: str
    ) -> AsyncGenerator[str, None]:
        async with self._client.stream(
            "POST",
            f"/chats/{chat_id}/messages",
            json={"content": content},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                if data:
                    yield data

    async def clear_messages(self, chat_id: UUID) -> None:
        resp = await self._client.delete(f"/chats/{chat_id}/messages")
        resp.raise_for_status()
        logger.info("cleared_messages chat_id=%s", chat_id)

    async def close(self) -> None:
        await self._client.aclose()
