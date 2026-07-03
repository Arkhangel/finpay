import asyncio
import logging

import uvicorn
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

from app.settings import settings
from app.bot.handlers import build_router
from app.bot.services.backend_client import BackendClient
from app.bot.services.broadcast import run_broadcast_worker
from app.bot.web import build_api

logger = logging.getLogger(__name__)


async def _run() -> None:
    bot = Bot(
        token=settings.bot.token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)

    backend = BackendClient(base_url=settings.bot.backend_url)
    dp["backend"] = backend

    dp.include_router(build_router())

    api = build_api(bot, settings.bot.internal_token.get_secret_value())
    server = uvicorn.Server(
        uvicorn.Config(
            api,
            host="0.0.0.0",
            port=settings.bot.bot_api_port,
            log_level="info",
        )
    )

    logger.info(
        "Starting bot, backend=%s, internal_api_port=%d",
        settings.bot.backend_url,
        settings.bot.bot_api_port,
    )
    try:
        await asyncio.gather(
            dp.start_polling(bot),
            server.serve(),
            run_broadcast_worker(bot, backend),
        )
    finally:
        await backend.close()
        await bot.session.close()


def run_bot() -> None:
    asyncio.run(_run())
