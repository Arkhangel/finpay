from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from uuid import UUID

from openai import AsyncOpenAI

from app.chat.context import build_sliding_window_context, fit_to_budget
from app.chat.domain import Chat, ChatMessage
from app.chat.repository import ChatRepository
from app.moderation import ModerationResult, ModerationService
from app.settings import settings

logger = logging.getLogger(__name__)

# Token budget constants
_CONTEXT_WINDOW = 16384
_RESPONSE_TOKENS = 1024
_SAFETY_MARGIN = 256
_MAX_HISTORY_TOKENS = _CONTEXT_WINDOW - _RESPONSE_TOKENS - _SAFETY_MARGIN

_BLOCKED_OUTPUT_MESSAGE = "Не могу показать ответ — он мог нарушить правила"


class ChatService:
    def __init__(
        self,
        repository: ChatRepository,
        llm_client: AsyncOpenAI,
        moderation: ModerationService | None = None,
    ) -> None:
        self._repo = repository
        self._llm = llm_client
        self._moderation = moderation

    async def check_input(self, chat_id: UUID, content: str) -> ModerationResult:
        if self._moderation is None:
            return ModerationResult(allowed=True)
        result = await self._moderation.check_input(content)
        if not result.allowed:
            await self._repo.record_moderation_incident(
                chat_id, "input", result.blocked_by, result.categories
            )
        return result

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
        self,
        chat_id: UUID,
        user_content: str,
        media_part: dict | None = None,
        media_meta: dict | None = None,
        result: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        """`result`, if given, is filled with {"message_id": <assistant ChatMessage.id>} once persisted."""
        media_refs = {**media_meta, "part": media_part} if media_part is not None else None
        user_msg = ChatMessage(
            chat_id=chat_id,
            role="user",
            content=user_content or "[медиа]",
            media_refs=media_refs,
        )
        await self._repo.append_message(chat_id, user_msg)

        chat = await self._repo.get_chat(chat_id)
        history = await self._repo.list_messages(chat_id, limit=settings.chat.context_window * 2)

        raw_history = [_to_openai_message(m) for m in history]

        messages = build_sliding_window_context(
            raw_history,
            chat.system_prompt if chat else None,
            settings.chat.context_window,
        )
        messages = fit_to_budget(messages, _MAX_HISTORY_TOKENS)

        model = settings.openai.model
        full_response = ""
        stream_broken = False
        started_at = time.monotonic()

        try:
            async with await self._llm.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
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
                persist_content = full_response
                # Проверяем только чистое завершение стрима: при обрыве текст неполный,
                # а токены уже отданы клиенту — переретроактивно скрыть их нельзя.
                if not stream_broken and self._moderation is not None:
                    mod_result = await self._moderation.check_output(full_response)
                    if not mod_result.allowed:
                        await self._repo.record_moderation_incident(
                            chat_id, "output", mod_result.blocked_by, mod_result.categories
                        )
                        persist_content = _BLOCKED_OUTPUT_MESSAGE
                        yield _BLOCKED_OUTPUT_MESSAGE
                if stream_broken:
                    logger.warning("saving_partial_response chars=%d", len(full_response))
                assistant_msg = ChatMessage(
                    chat_id=chat_id,
                    role="assistant",
                    content=persist_content,
                    latency_ms=int((time.monotonic() - started_at) * 1000),
                )
                await self._repo.append_message(chat_id, assistant_msg)
                if result is not None:
                    result["message_id"] = assistant_msg.id

    async def clear_history(self, chat_id: UUID) -> None:
        await self._repo.soft_delete_messages(chat_id)

    async def append_system_message(self, chat_id: UUID, text: str) -> ChatMessage:
        message = ChatMessage(chat_id=chat_id, role="assistant", content=text)
        return await self._repo.append_message(chat_id, message)


def _to_openai_message(message: ChatMessage) -> dict:
    part = (message.media_refs or {}).get("part")
    if part is None:
        return {"role": message.role, "content": message.content}
    return {
        "role": message.role,
        "content": [{"type": "text", "text": message.content}, part],
    }
