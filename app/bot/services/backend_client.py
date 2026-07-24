from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from uuid import UUID

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.settings import settings

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = httpx.Timeout(connect=3.0, read=60.0, write=10.0, pool=5.0)
_STREAM_TIMEOUT = httpx.Timeout(connect=3.0, read=120.0, write=10.0, pool=5.0)

# Ретраится только соединение — 5xx может значить, что LLM-вызов уже оплачен на бэкенде.
_retry_on_connect_error = retry(
    retry=retry_if_exception_type((httpx.ConnectError, httpx.ConnectTimeout)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=4),
    reraise=True,
)


class BackendClient:
    def __init__(self, base_url: str) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=_DEFAULT_TIMEOUT)
        self._chat_cache: dict[tuple[str, str], UUID] = {}

    @_retry_on_connect_error
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
        self,
        chat_id: UUID,
        content: str,
        media: bytes | None = None,
        mime: str | None = None,
        result: dict | None = None,
    ) -> AsyncIterator[str]:
        """`result`, if given, is filled with {"message_id": <UUID | None>, "sources": [...] | None}
        once the stream ends — "sources" — только если бэкенд подключён к RAG (Б5.5)."""
        # Без ретрая: частично отданный стрим нельзя повторить, не задвоив LLM-вызов.
        files = {"media": ("file.bin", media, mime)} if media else None
        data = {"content": content}
        if result is not None:
            result["sources"] = None
        current_event: str | None = None
        async with self._client.stream(
            "POST",
            f"/chats/{chat_id}/messages",
            data=data,
            files=files,
            timeout=_STREAM_TIMEOUT,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    current_event = line.removeprefix("event: ")
                    continue
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line.removeprefix("data: "))
                if current_event == "sources":
                    if result is not None:
                        result["sources"] = payload.get("sources")
                    current_event = None
                    continue
                current_event = None
                if payload.get("type") == "token":
                    yield payload["delta"]
                elif payload.get("type") == "done":
                    if result is not None:
                        message_id = payload.get("message_id")
                        result["message_id"] = UUID(message_id) if message_id else None
                    return

    @_retry_on_connect_error
    async def clear_messages(self, chat_id: UUID) -> None:
        resp = await self._client.delete(f"/chats/{chat_id}/messages")
        resp.raise_for_status()
        logger.info("cleared_messages chat_id=%s", chat_id)

    @_retry_on_connect_error
    async def submit_feedback(
        self, chat_id: UUID, message_id: UUID, value: str, sources: list[dict] | None = None
    ) -> None:
        resp = await self._client.post(
            f"/chats/{chat_id}/messages/{message_id}/feedback",
            json={"value": value, "sources": sources},
        )
        resp.raise_for_status()

    # ── admin ────────────────────────────────────────────────────────────────

    @staticmethod
    def _admin_headers() -> dict[str, str]:
        return {"X-Admin-Token": settings.admin_token.get_secret_value()}

    @_retry_on_connect_error
    async def get_admin_stats(self) -> dict:
        resp = await self._client.get("/chats/admin/stats", headers=self._admin_headers())
        resp.raise_for_status()
        return resp.json()

    @_retry_on_connect_error
    async def get_admin_users(self, limit: int = 50) -> list[dict]:
        resp = await self._client.get(
            "/chats/admin/users", params={"limit": limit}, headers=self._admin_headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def post_admin_broadcast(self, message: str, interface_filter: str = "telegram") -> dict:
        # Без ретрая: повтор POST продублировал бы рассылку.
        resp = await self._client.post(
            "/chats/admin/broadcast",
            json={"message": message, "interface_filter": interface_filter},
            headers=self._admin_headers(),
        )
        resp.raise_for_status()
        return resp.json()

    @_retry_on_connect_error
    async def get_pending_broadcasts(self, limit: int = 20) -> list[dict]:
        resp = await self._client.get(
            "/chats/admin/broadcast/pending", params={"limit": limit}, headers=self._admin_headers()
        )
        resp.raise_for_status()
        return resp.json()

    async def ack_broadcast(self, broadcast_id: str, status: str) -> None:
        resp = await self._client.post(
            f"/chats/admin/broadcast/{broadcast_id}/ack",
            json={"status": status},
            headers=self._admin_headers(),
        )
        resp.raise_for_status()

    async def close(self) -> None:
        await self._client.aclose()
