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
from .activity import ActivityTracker, CommandActivityMiddleware
from .album import AlbumBuffer
from .business import BusinessMonitor
from .config import Settings
from .control_panel import create_control_router, restore_ai_settings
from .emoji_theme import EmojiTheme
from .gemini import GeminiService
from .handlers import create_router
from .inline import InlineService, create_inline_router
from .media import MediaExtractor
from .moderation import BanMiddleware, create_moderation_router
from .outgoing_emoji import OutgoingEmojiMiddleware
from .processor import MessageProcessor
from .readme import create_readme_router
from .storage import Storage
from .usage import UsageTracker
from .wall import WallService

logger = logging.getLogger(__name__)


async def run_bot(settings: Settings) -> None:
    storage = Storage(settings.database_path)
    gemini = GeminiService(settings)
    albums = AlbumBuffer(settings.album_debounce_seconds)
    bot = Bot(token=settings.telegram_bot_token)
    dispatcher = Dispatcher()
    inline: InlineService | None = None

    try:
        await storage.open()
        gemini.health.attach_storage(storage)
        await restore_ai_settings(storage, gemini)
        owners_removed = await storage.delete_bot_users(settings.owner_ids)
        if owners_removed:
            logger.info("Removed %s configured owners from usage audit", owners_removed)
        emoji_theme = EmojiTheme(storage)
        bot.session.middleware(OutgoingEmojiMiddleware(emoji_theme))
        bot_user = await bot.get_me()
        usage = UsageTracker(bot=bot, settings=settings, storage=storage)
        activity = ActivityTracker(
            storage, settings.owner_ids, bot_user.username or "",
            bot_id=bot_user.id,
            group_default_mode=settings.group_default_mode,
        )
        processor = MessageProcessor(
            bot=bot,
            bot_user=bot_user,
            settings=settings,
            storage=storage,
            gemini=gemini,
            media=MediaExtractor(settings),
            usage=usage,
            emoji_theme=emoji_theme,
            activity=activity,
        )
        monitor = BusinessMonitor(bot=bot, settings=settings, storage=storage)
        inline = InlineService(processor)
        wall = WallService(
            max_source_bytes=settings.wall_max_source_bytes,
            max_pixels=settings.wall_max_pixels,
            tile_size=settings.wall_tile_size,
            download_timeout_seconds=settings.media_download_timeout_seconds,
        )
        ban_middleware = BanMiddleware(storage, settings.owner_ids, activity)
        dispatcher.message.outer_middleware(ban_middleware)
        dispatcher.edited_message.outer_middleware(ban_middleware)
        dispatcher.callback_query.outer_middleware(ban_middleware)
        dispatcher.message.outer_middleware(CommandActivityMiddleware(activity))
        dispatcher.include_router(create_moderation_router(processor))
        dispatcher.include_router(create_inline_router(inline))
        dispatcher.include_router(create_readme_router())
        dispatcher.include_router(create_control_router(processor))
        dispatcher.include_router(
            create_router(processor, albums, monitor, usage, emoji_theme, wall)
        )
        media_removed = await monitor.prune_archived_media(
            settings.business_message_retention_days
        )
        if media_removed:
            logger.info("Pruned %s expired business media archives", media_removed)
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
        await storage.prune_user_events()
        await _set_commands(bot)
        await bot.delete_webhook(drop_pending_updates=settings.drop_pending_updates)

        logger.info(
            "Starting @%s v%s with provider=%s model=%s fallback=%s free_only=%s",
            bot_user.username,
            __version__,
            gemini.current_provider,
            gemini.current_model,
            settings.groq_model if settings.groq_fallback_ready else "disabled",
            gemini.free_only,
        )
        await dispatcher.start_polling(
            bot,
            allowed_updates=dispatcher.resolve_used_update_types(),
            close_bot_session=False,
        )
    finally:
        if inline is not None:
            await inline.close()
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
        BotCommand(command="readme", description="Актуальная документация с GitHub"),
        BotCommand(command="menu", description="Показать клавиатуру"),
        BotCommand(command="inline", description="Перевод, текст и калькулятор в любом чате"),
        BotCommand(command="hidekeyboard", description="Скрыть клавиатуру"),
        BotCommand(command="settings", description="Настройки бота (владелец)"),
        BotCommand(command="diagnostics", description="Диагностика моделей (владелец)"),
        BotCommand(command="guide", description="Видеоинструкция"),
        BotCommand(command="wall", description="Нарезать фото на стенку"),
        BotCommand(command="emoji", description="Настроить эмодзи (владелец)"),
        BotCommand(command="ask", description="Задать вопрос Gemini"),
        BotCommand(command="reset", description="Сбросить контекст"),
        BotCommand(command="cancel", description="Остановить запрос"),
        BotCommand(command="autoreply", description="Business-автоответы on/off"),
        BotCommand(command="chats", description="Настроить отдельные Business-чаты"),
        BotCommand(command="users", description="Кто пользуется ботом"),
        BotCommand(command="ban", description="Заблокировать пользователя (владелец)"),
        BotCommand(command="unban", description="Разблокировать пользователя (владелец)"),
        BotCommand(command="banlist", description="Заблокированные пользователи (владелец)"),
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
