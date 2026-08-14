from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.repository import AdminRepository
from app.chat.deps import get_pg_session
from app.settings import settings


def require_admin_token(x_admin_token: Annotated[str, Header()]) -> None:
    configured = settings.admin_token.get_secret_value()
    if not configured or not secrets.compare_digest(x_admin_token, configured):
        raise HTTPException(status_code=401, detail="Invalid admin token")


async def get_admin_session(
    _: Annotated[None, Depends(require_admin_token)],
    session: Annotated[AsyncSession | None, Depends(get_pg_session)],
) -> AsyncSession:
    if session is None:
        raise HTTPException(status_code=503, detail="Admin endpoints require CHAT__REPOSITORY=postgres")
    return session


def get_admin_repository(
    session: Annotated[AsyncSession, Depends(get_admin_session)],
) -> AdminRepository:
    return AdminRepository(session)
