from aiogram import Router

from app.bot.handlers import admin, commands, feedback, fsm, media, text


def build_router() -> Router:
    router = Router()
    router.include_router(admin.router)
    router.include_router(commands.router)
    router.include_router(feedback.router)
    router.include_router(fsm.router)
    router.include_router(media.router)
    router.include_router(text.router)
    return router
