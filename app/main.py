"""FastAPI-приложение FinPay LLM Service.

Запуск:
    uvicorn app.main:app --reload

Проверки:
    curl localhost:8000/health
    curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
         -d '{"messages":[{"role":"user","content":"hi"}]}'
    curl -N -X POST localhost:8000/chat/stream -H 'Content-Type: application/json' \
         -d '{"messages":[{"role":"user","content":"считай до пяти"}]}'
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware

from app.exceptions.handlers import setup_exception_handlers
from app.lifespan import lifespan
from app.routers import chat, health, models
from app.settings import settings
from app.settings.logging import request_id_var

logger = logging.getLogger("llm-service")


def create_app() -> FastAPI:
    app = FastAPI(
        title="FinPay LLM Service",
        version="1.0.0",
        lifespan=lifespan,
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=settings.cors_origins,
                allow_methods=["*"],
                allow_headers=["*"],
                expose_headers=["X-Request-ID"],
            ),
        ],
    )

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            logger.info(
                "http_request",
                extra={"status": response.status_code, "duration_ms": duration_ms},
            )
            response.headers["x-request-id"] = request_id
            return response
        finally:
            request_id_var.reset(token)

    setup_exception_handlers(app)

    app.include_router(chat.router)
    app.include_router(health.router)
    app.include_router(models.router)

    return app


app = create_app()
