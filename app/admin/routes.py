from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from app.admin.deps import get_admin_repository
from app.admin.repository import AdminRepository
from app.admin.schemas import (
    BroadcastAckIn,
    BroadcastIn,
    BroadcastOut,
    BroadcastPendingItem,
    StatsOut,
    UserOut,
)

router = APIRouter(prefix="/chats/admin", tags=["admin"])

AdminRepositoryDep = Annotated[AdminRepository, Depends(get_admin_repository)]


@router.get("/stats", response_model=StatsOut)
async def get_stats(repo: AdminRepositoryDep) -> StatsOut:
    return await repo.get_stats()


@router.get("/users", response_model=list[UserOut])
async def list_users(repo: AdminRepositoryDep, limit: int = 50) -> list[UserOut]:
    return await repo.list_users(limit=limit)


@router.post("/broadcast", response_model=BroadcastOut, status_code=201)
async def create_broadcast(body: BroadcastIn, repo: AdminRepositoryDep) -> BroadcastOut:
    broadcast_id = await repo.create_broadcast(body.message, body.interface_filter)
    return BroadcastOut(id=broadcast_id, status="pending")


@router.get("/broadcast/pending", response_model=list[BroadcastPendingItem])
async def list_pending_broadcasts(repo: AdminRepositoryDep, limit: int = 20) -> list[BroadcastPendingItem]:
    """Internal: бот забирает отсюда сообщения и рассылает пользователям сам."""
    return await repo.list_pending_broadcasts(limit=limit)


@router.post("/broadcast/{broadcast_id}/ack")
async def ack_broadcast(broadcast_id: UUID, body: BroadcastAckIn, repo: AdminRepositoryDep) -> dict:
    await repo.ack_broadcast(broadcast_id, body.status)
    return {"status": "ok"}
