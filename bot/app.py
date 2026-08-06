from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import (
    BotCommand,
    BotCommandScopeAllGroupChats,
    BotCommandScopeAllPrivateChats,
)

from . import __version__
from .album import AlbumBuffer
from .business import BusinessMonitor
from .config import Settings
from .gemini import GeminiService, ModelSelectionError
from .handlers import create_router
from .media import MediaExtractor
from .processor import MessageProcessor
from .storage import Storage
from .usage import UsageTracker

logger = logging.getLogger(__name__)


async def run_bot(settings: Settings) -> None:
    storage = Storage(settings.database_path)
    gemini = GeminiService(settings)
    albums = AlbumBuffer(settings.album_debounce_seconds)
    bot = Bot(token=settings.telegram_bot_token)
    dispatcher = Dispatcher()

    try:
        await storage.open()
        saved_selection = await storage.get_ai_selection()
        if saved_selection is not None:
            try:
                gemini.select_model(*saved_selection)
            except ModelSelectionError as exc:
                logger.warning("Ignoring unavailable saved AI selection: %s", exc)
        owners_removed = await storage.delete_bot_users(settings.owner_ids)
        if owners_removed:
            logger.info("Removed %s configured owners from usage audit", owners_removed)
        bot_user = await bot.get_me()
        usage = UsageTracker(bot=bot, settings=settings, storage=storage)
        processor = MessageProcessor(
            bot=bot,
            bot_user=bot_user,
            settings=settings,
            storage=storage,
            gemini=gemini,
            media=MediaExtractor(settings),
            usage=usage,
        )
        monitor = BusinessMonitor(bot=bot, settings=settings, storage=storage)
        dispatcher.include_router(create_router(processor, albums, monitor, usage))
        removed = await storage.prune_business_messages(
            settings.business_message_retention_days
        )
        if removed:
            logger.info("Pruned %s expired business message snapshots", removed)
        history_removed = await storage.prune_conversation_history(
            settings.openrouter_history_retention_days
        )
        if history_removed:
            logger.info(
                "Pruned %s expired local conversation exchanges", history_removed
            )
        await _set_commands(bot)
        await bot.delete_webhook(drop_pending_updates=settings.drop_pending_updates)

        logger.info(
            "Starting @%s v%s with provider=%s model=%s fallback=%s",
            bot_user.username,
            __version__,
            gemini.current_provider,
            gemini.current_model,
            settings.groq_model if settings.groq_fallback_ready else "disabled",
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
        BotCommand(command="autoreply", description="Business-автоответы on/off"),
        BotCommand(command="chats", description="Настроить отдельные Business-чаты"),
        BotCommand(command="users", description="Кто пользуется ботом"),
        BotCommand(command="stats", description="Статистика использования"),
        BotCommand(command="model", description="Выбрать AI-модель (владелец)"),
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
