"""Внутренний HTTP-API бота — обратный канал backend -> bot для /notify."""

from __future__ import annotations

import secrets

from aiogram import Bot
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel


class NotifyRequest(BaseModel):
    chat_id: int
    text: str


def build_api(bot: Bot, internal_token: str) -> FastAPI:
    api = FastAPI(title="FinPay Bot Internal API")

    @api.post("/notify")
    async def notify(
        req: NotifyRequest,
        x_internal_token: str = Header(...),
    ) -> dict:
        if not secrets.compare_digest(x_internal_token, internal_token):
            raise HTTPException(status_code=401)
        await bot.send_message(chat_id=req.chat_id, text=req.text)
        return {"ok": True}

    return api
