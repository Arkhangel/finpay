import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness check")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/ready", summary="Readiness check")
async def ready(request: Request):
    cache = request.app.state.cache
    if cache is not None:
        try:
            await asyncio.wait_for(cache.ping(), timeout=2.0)
            return {"status": "ok", "redis": "up"}
        except Exception:
            pass
    return JSONResponse(
        status_code=503,
        content={"status": "degraded", "redis": "down"},
    )
