from aiogram import Router

from app.bot.handlers import commands, fsm, media, text


def build_router() -> Router:
    router = Router()
    router.include_router(commands.router)
    router.include_router(fsm.router)
    router.include_router(media.router)
    router.include_router(text.router)
    return router
