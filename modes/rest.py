import uvicorn

from app.settings import settings


def run_server() -> None:
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.reload,
    )
