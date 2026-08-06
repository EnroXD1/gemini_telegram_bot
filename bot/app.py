from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import (
    BotCommand,
    BotCommandScopeAllGroupChats,
    BotCommandScopeAllPrivateChats,
)

from .album import AlbumBuffer
from .config import Settings
from .gemini import GeminiService
from .handlers import create_router
from .media import MediaExtractor
from .processor import MessageProcessor
from .storage import Storage

logger = logging.getLogger(__name__)


async def run_bot(settings: Settings) -> None:
    storage = Storage(settings.database_path)
    gemini = GeminiService(settings)
    albums = AlbumBuffer(settings.album_debounce_seconds)
    bot = Bot(token=settings.telegram_bot_token)
    dispatcher = Dispatcher()

    try:
        await storage.open()
        bot_user = await bot.get_me()
        processor = MessageProcessor(
            bot=bot,
            bot_user=bot_user,
            settings=settings,
            storage=storage,
            gemini=gemini,
            media=MediaExtractor(settings),
        )
        dispatcher.include_router(create_router(processor, albums))
        await _set_commands(bot)
        await bot.delete_webhook(drop_pending_updates=settings.drop_pending_updates)

        logger.info(
            "Starting @%s with model=%s", bot_user.username, settings.gemini_model
        )
        await dispatcher.start_polling(
            bot,
            allowed_updates=dispatcher.resolve_used_update_types(),
            close_bot_session=False,
        )
    finally:
        results = await asyncio.gather(
            albums.close(),
            storage.close(),
            gemini.close(),
            bot.session.close(),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Shutdown cleanup failed: %s", type(result).__name__)


async def _set_commands(bot: Bot) -> None:
    private_commands = [
        BotCommand(command="start", description="Запустить бота"),
        BotCommand(command="help", description="Справка и возможности"),
        BotCommand(command="ask", description="Задать вопрос Gemini"),
        BotCommand(command="reset", description="Сбросить контекст"),
        BotCommand(command="cancel", description="Остановить запрос"),
        BotCommand(command="status", description="Модель и состояние"),
    ]
    group_commands = private_commands + [
        BotCommand(command="mode", description="Режим ответов группы"),
    ]
    try:
        await bot.set_my_commands(
            private_commands, scope=BotCommandScopeAllPrivateChats()
        )
        await bot.set_my_commands(group_commands, scope=BotCommandScopeAllGroupChats())
    except Exception as exc:
        logger.warning("Could not update Telegram command menu: %s", type(exc).__name__)
