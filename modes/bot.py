import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

from app.settings import settings
from app.bot.handlers import build_router
from app.bot.services.backend_client import BackendClient

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

    logger.info("Starting bot, backend=%s", settings.bot.backend_url)
    try:
        await dp.start_polling(bot)
    finally:
        await backend.close()
        await bot.session.close()


def run_bot() -> None:
    asyncio.run(_run())
