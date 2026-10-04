"""Точка входа: python -m app.main"""
import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand

from app.bot import router
from app.config import settings
from app.retriever import Retriever


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if not settings.bot_token:
        raise SystemExit("BOT_TOKEN не задан в .env")

    logging.info("Загрузка индекса и модели эмбеддингов...")
    retriever = Retriever()  # грузим один раз при старте, а не на каждый вопрос

    bot = Bot(settings.bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher(storage=MemoryStorage())  # ключи и история — только в памяти
    dp["retriever"] = retriever               # внедряется в хендлеры как аргумент
    dp.include_router(router)

    await bot.set_my_commands([
        BotCommand(command="start", description="Начать заново"),
        BotCommand(command="key", description="Сменить API-ключ DeepSeek"),
        BotCommand(command="reset", description="Очистить историю диалога"),
        BotCommand(command="help", description="Справка"),
    ])
    logging.info("Бот запущен")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
