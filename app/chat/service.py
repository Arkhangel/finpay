from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from uuid import UUID

from openai import AsyncOpenAI

from app.chat.context import build_sliding_window_context, fit_to_budget
from app.chat.domain import Chat, ChatMessage
from app.chat.repository import ChatRepository
from app.settings import settings

logger = logging.getLogger(__name__)

# Token budget constants
_CONTEXT_WINDOW = 16384
_RESPONSE_TOKENS = 1024
_SAFETY_MARGIN = 256
_MAX_HISTORY_TOKENS = _CONTEXT_WINDOW - _RESPONSE_TOKENS - _SAFETY_MARGIN


class ChatService:
    def __init__(self, repository: ChatRepository, llm_client: AsyncOpenAI) -> None:
        self._repo = repository
        self._llm = llm_client

    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        return await self._repo.create_chat(owner_external_id, interface, system_prompt)

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        return await self._repo.get_chat(chat_id)

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        return await self._repo.list_messages(chat_id, limit=limit)

    async def send_message(
        self, chat_id: UUID, user_content: str
    ) -> AsyncGenerator[str, None]:
        user_msg = ChatMessage(chat_id=chat_id, role="user", content=user_content)
        await self._repo.append_message(chat_id, user_msg)

        chat = await self._repo.get_chat(chat_id)
        history = await self._repo.list_messages(chat_id, limit=settings.chat.context_window * 2)

        raw_history = [{"role": m.role, "content": m.content} for m in history]

        messages = build_sliding_window_context(
            raw_history,
            chat.system_prompt if chat else None,
            settings.chat.context_window,
        )
        messages = fit_to_budget(messages, _MAX_HISTORY_TOKENS)

        model = settings.openai.model
        full_response = ""
        stream_broken = False

        try:
            async with await self._llm.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            ) as stream:
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_response += delta
                        yield delta
        except Exception as exc:
            stream_broken = True
            logger.warning("stream_interrupted content_so_far=%d chars error=%s", len(full_response), exc)
            raise
        finally:
            if full_response:
                if stream_broken:
                    logger.warning("saving_partial_response chars=%d", len(full_response))
                assistant_msg = ChatMessage(
                    chat_id=chat_id,
                    role="assistant",
                    content=full_response,
                )
                await self._repo.append_message(chat_id, assistant_msg)

    async def clear_history(self, chat_id: UUID) -> None:
        await self._repo.soft_delete_messages(chat_id)
