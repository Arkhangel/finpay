from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError

logger = logging.getLogger("llm-service")


def setup_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError):
        if isinstance(exc, LLMRateLimitError):
            status = 429
        elif isinstance(exc, LLMTimeoutError):
            status = 504
        else:
            status = 502
        return JSONResponse(
            status_code=status,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        errors = [
            {"field": ".".join(str(loc) for loc in e["loc"]), "message": e["msg"]}
            for e in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"errors": errors})
