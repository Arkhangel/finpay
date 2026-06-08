from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", summary="Health check")
async def health() -> dict:
    """Всегда 200, даже если Redis или OpenAI недоступны."""
    return {"status": "ok"}
