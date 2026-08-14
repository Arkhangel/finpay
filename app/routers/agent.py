"""SSE-стриминг персистентного графа (блок 6.4).

Требует X-Admin-Token (тот же секрет, что и /admin/*) — эндпоинт даёт прямой
доступ к HIL-подтверждению реальной отправки в Telegram, анонимный доступ
недопустим.

    curl -N -X POST http://localhost:8000/agent/stream \
      -H 'Content-Type: application/json' -H 'X-Admin-Token: ...' \
      -d '{"thread_id":"demo-1","input":{"messages":[{"role":"user","content":"..."}]}}'

Резюме после интеррапта — тот же endpoint, тот же thread_id, другой input:

    curl -N -X POST http://localhost:8000/agent/stream \
      -H 'X-Admin-Token: ...' -d '{"thread_id":"demo-1","input":{"resume":true}}'
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from langgraph.types import Command, Interrupt
from pydantic import BaseModel

from app.admin.deps import require_admin_token
from app.chat.deps import get_moderation_service
from app.deps.providers import AgentGraphDep
from app.moderation import ModerationService
from app.services.agent_persistent import SYSTEM_PROMPT

# Эндпоинт демонстрирует HIL-механизм (Б6.4) напрямую по HTTP (см. docstring
# ниже) — в отличие от /chat, у него нет собственного клиента внутри проекта
# (бот работает через /chats/{id}/messages), поэтому не было и своего слоя
# аутентификации/модерации. Защищаем тем же X-Admin-Token, что и /admin/*
# (найдено global-аудитом: без этого любой анонимный клиент мог послать
# user_role="full" и обойти HIL-подтверждение на реальную отправку в Telegram).
router = APIRouter(prefix="/agent", tags=["agent"], dependencies=[Depends(require_admin_token)])


class AgentStreamRequest(BaseModel):
    thread_id: str
    input: dict[str, Any]
    # read-only / write-with-approve / full — см. docs/agent-persistent-report.md
    user_role: str = "write-with-approve"


def _json_safe(obj: Any) -> Any:
    # LangGraph эмитит __interrupt__ прямо в потоке updates как
    # {"__interrupt__": (Interrupt(...),)} — Interrupt не dict/pydantic-модель
    # и без явной обработки роняет json.dumps (найдено вживую curl-проверкой:
    # TypeError молча обрывал SSE-стрим сразу на моменте паузы).
    if isinstance(obj, Interrupt):
        return {"value": _json_safe(obj.value), "id": obj.id}
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


@router.post("/stream", summary="SSE-стрим ReAct-графа с HIL (блок 6.4)")
async def agent_stream(
    req: AgentStreamRequest,
    graph: AgentGraphDep,
    moderation: Annotated[ModerationService | None, Depends(get_moderation_service)],
) -> StreamingResponse:
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent graph unavailable")

    config = {"configurable": {"thread_id": req.thread_id, "user_role": req.user_role}}
    # thread_id стабильный и приходит от вызывающего — НЕ генерируем uuid4()
    # здесь, иначе checkpointer не найдёт сохранённый state следующего запроса.
    if "resume" in req.input:
        graph_input = Command(resume=req.input["resume"])
    else:
        # Вызывающий (curl/бот) знает только про messages — остальные поля
        # AgentState (iteration_count/tool_results/pending_action) у нового
        # треда дефолтятся здесь, а не требуются от клиента: без этого
        # первый ainvoke на новом thread_id падал с KeyError('iteration_count')
        # (найдено вживую при curl-проверке этого самого эндпоинта).
        messages = req.input.get("messages", [])
        if not any(isinstance(m, dict) and m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]

        # Модерация входа (global-аудит, находка №4) — только для новых
        # сообщений: у resume нет нового пользовательского текста для проверки.
        if moderation is not None:
            last_user = next(
                (m.get("content", "") for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"),
                "",
            )
            if last_user:
                mod_result = await moderation.check_input(last_user)
                if not mod_result.allowed:
                    raise HTTPException(
                        status_code=403,
                        detail={"code": "moderation_blocked", "categories": mod_result.categories},
                    )

        graph_input = {
            "iteration_count": 0, "tool_results": [], "pending_action": None,
            **req.input, "messages": messages,
        }

    async def event_generator():
        # Модерация выхода — тем же ModerationService, что /chat и /chats/*
        # (global-аудит, находка №4: раньше не было вообще никакой). Полный
        # ответ известен только после конца стрима (граф может звать
        # несколько узлов/tool call'ов подряд), поэтому итоговая проверка —
        # как и в исходном /chats/{id}/messages до его отдельного фикса —
        # постфактум: инцидент фиксируется, финальный чанк размечается
        # заблокированным, но уже отданные клиенту SSE-чанки не отозвать.
        assistant_text_parts: list[str] = []

        async for mode, chunk in graph.astream(graph_input, config=config, stream_mode=["updates", "messages"]):
            if mode == "messages":
                message_chunk, metadata = chunk
                if message_chunk.content:
                    assistant_text_parts.append(str(message_chunk.content))
                payload = {"content": message_chunk.content, "node": metadata.get("langgraph_node")}
            else:  # "updates"
                payload = _json_safe(chunk)
            yield f"data: {json.dumps({'type': mode, 'payload': payload}, ensure_ascii=False)}\n\n"

        if moderation is not None and assistant_text_parts:
            mod_result = await moderation.check_output("".join(assistant_text_parts))
            if not mod_result.allowed:
                event = {"type": "moderation_blocked", "payload": {"categories": mod_result.categories}}
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        snapshot = await graph.aget_state(config)
        if snapshot.interrupts:
            interrupt_payload = [i.value for i in snapshot.interrupts]
            event = {"type": "__interrupt__", "payload": _json_safe(interrupt_payload)}
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
