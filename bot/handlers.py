from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.filters.command import CommandObject
from aiogram.types import (
    BusinessConnection,
    BusinessMessagesDeleted,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from . import __version__
from .album import AlbumBuffer
from .business import BusinessMessageCaptureMiddleware, BusinessMonitor
from .processor import MessageProcessor
from .storage import BotUserRecord
from .usage import BotUsageMiddleware, UsageTracker


def create_router(
    processor: MessageProcessor,
    albums: AlbumBuffer,
    monitor: BusinessMonitor,
    usage: UsageTracker,
) -> Router:
    router = Router(name="gemini-assistant")
    router.message.outer_middleware(BotUsageMiddleware(usage))
    router.business_message.outer_middleware(
        BusinessMessageCaptureMiddleware(monitor)
    )

    @router.business_connection()
    async def business_connection_handler(connection: BusinessConnection) -> None:
        await monitor.handle_connection(connection)

    @router.edited_business_message()
    async def edited_business_message_handler(message: Message) -> None:
        await monitor.handle_edited_message(message)

    @router.deleted_business_messages()
    async def deleted_business_messages_handler(
        event: BusinessMessagesDeleted,
    ) -> None:
        await monitor.handle_deleted_messages(event)

    @router.message(Command("start"))
    @router.business_message(Command("start"))
    async def start_handler(message: Message) -> None:
        if not await processor.can_respond(message):
            return
        await message.reply(
            "Привет! Я передаю сообщения Gemini и отвечаю с учётом контекста.\n\n"
            "Если подключить меня к Telegram Business с правом чтения сообщений, "
            "я уведомлю вас об изменённых и удалённых сообщениях собеседников, а "
            "также сохраню входящие фото, видео, голосовые и кружки.\n\n"
            "Мне можно отправлять текст, фото, PDF, текстовые документы, аудио, "
            "голосовые, видео, стикеры, геолокацию, контакты и опросы. Я также "
            "понимаю подписи, альбомы и сообщения, на которые вы отвечаете.\n\n"
            "Команда /help покажет режимы работы."
        )

    @router.message(Command("help"))
    @router.business_message(Command("help"))
    async def help_handler(message: Message) -> None:
        if not await processor.can_respond(message):
            return
        await message.reply(
            "Команды:\n"
            "/ask запрос — явно обратиться к боту в группе; команду можно добавить "
            "к подписи файла\n"
            "/reset — начать диалог с чистым контекстом\n"
            "/cancel — остановить текущий запрос\n"
            "/status — показать модель и режим этого чата\n"
            "/mode mentions|all|off — режим группы (только администратор)\n"
            "/autoreply on|off — общие Business-автоответы (только владелец)\n"
            "/chats — режимы отдельных Business-собеседников\n"
            "/users — последние пользователи самого бота (только владелец)\n"
            "/stats — статистика использования (только владелец)\n\n"
            "Telegram Business: сохраняю исходный текст входящих сообщений на 30 "
            "дней, уведомляю об изменениях и удалениях и сразу отправляю вам копию "
            "входящих медиа. Секретные чаты обычным ботам недоступны.\n\n"
            "Ответы оформляются с помощью Markdown: поддерживаются заголовки, "
            "списки, выделение, цитаты, ссылки и блоки кода. Пока модель думает, "
            "показывается временный анимированный статус.\n\n"
            "В личном чате я отвечаю на все поддерживаемые сообщения. В группе по "
            "умолчанию отвечаю на /ask, упоминание моего @username или ответ на моё "
            "сообщение."
        )

    @router.message(Command("ask"))
    @router.business_message(Command("ask"))
    async def ask_handler(message: Message) -> None:
        if not await processor.can_respond(message):
            return
        if message.media_group_id:
            await albums.add(message, processor.process)
        else:
            await processor.process([message], force=True)

    @router.message(Command("reset"))
    @router.business_message(Command("reset"))
    async def reset_handler(message: Message) -> None:
        if not await processor.can_respond(message):
            return
        if not await _can_manage_shared_context(processor, message):
            await message.reply(
                "В группе сбрасывать общий контекст может только администратор."
            )
            return
        processor.cancel(message)
        await processor.reset(message)
        await message.reply(
            "Контекст этого диалога сброшен. Следующее сообщение начнёт новую беседу."
        )

    @router.message(Command("cancel"))
    @router.business_message(Command("cancel"))
    async def cancel_handler(message: Message) -> None:
        if not await processor.can_respond(message):
            return
        if not await _can_manage_shared_context(processor, message):
            await message.reply(
                "В группе останавливать общий запрос может только администратор."
            )
            return
        if processor.cancel(message):
            await message.reply("Текущий запрос остановлен.")
        else:
            await message.reply("Сейчас нет активного запроса для этого диалога.")

    @router.message(Command("mode"))
    async def mode_handler(message: Message, command: CommandObject) -> None:
        if message.chat.type == "private":
            await message.reply(
                "В личном чате бот всегда отвечает на поддерживаемые сообщения."
            )
            return
        if not await _is_admin(processor, message):
            await message.reply("Менять режим группы может только администратор.")
            return

        aliases = {
            "mentions": "mentions",
            "mention": "mentions",
            "упоминания": "mentions",
            "all": "all",
            "все": "all",
            "off": "off",
            "выкл": "off",
        }
        argument = (command.args or "").strip().lower()
        mode = aliases.get(argument)
        if mode is None:
            current = await processor.storage.get_group_mode(
                message.chat.id, processor.settings.group_default_mode
            )
            await message.reply(
                f"Текущий режим: {current}.\n"
                "Использование: /mode mentions, /mode all или /mode off"
            )
            return

        await processor.storage.set_group_mode(message.chat.id, mode)
        descriptions = {
            "mentions": "отвечаю на /ask, упоминания и ответы на мои сообщения",
            "all": "отвечаю на все доступные сообщения",
            "off": "автоматические ответы отключены; команда /ask продолжает работать",
        }
        extra = (
            "\nДля режима all у бота должен быть отключён Group Privacy в BotFather."
            if mode == "all"
            else ""
        )
        await message.reply(f"Режим изменён: {mode} — {descriptions[mode]}.{extra}")

    @router.message(Command("autoreply", "auto"))
    async def autoreply_handler(message: Message, command: CommandObject) -> None:
        if message.chat.type != "private":
            await message.reply(
                "Настройка Business-автоответов доступна в личном чате с ботом."
            )
            return
        user = message.from_user
        if user is None or not await _is_business_owner(processor, user.id):
            await message.reply(
                "Эта настройка доступна владельцу подключённого Telegram Business."
            )
            return

        aliases = {
            "on": True,
            "вкл": True,
            "enable": True,
            "off": False,
            "выкл": False,
            "disable": False,
        }
        argument = (command.args or "status").strip().lower()
        if argument in {"", "status", "статус"}:
            enabled = await processor.storage.get_business_auto_reply_enabled(
                user.id, processor.settings.business_auto_reply_enabled
            )
            state = "включены" if enabled else "выключены"
            await message.reply(
                f"Автоответы в ваших Business-чатах: {state}.\n"
                "Мониторинг удалений, изменений и сохранение медиа управляются "
                "отдельно и продолжают работать.\n\n"
                "Использование: /autoreply on или /autoreply off\n"
                "/chats — индивидуальные настройки собеседников"
            )
            return

        enabled = aliases.get(argument)
        if enabled is None:
            await message.reply(
                "Неизвестный режим. Используйте /autoreply on, "
                "/autoreply off или /autoreply status."
            )
            return

        await processor.storage.set_business_auto_reply_enabled(user.id, enabled)
        if enabled:
            await message.reply(
                "Общие Business-автоответы включены. Чаты, которые вы отдельно "
                "перевели в режим мониторинга через /chats, останутся без ответов."
            )
        else:
            await message.reply(
                "Включён режим «только мониторинг». Я не буду отвечать "
                "собеседникам и отправлять приветствие, но продолжу сохранять "
                "медиа и уведомлять вас об изменённых и удалённых сообщениях."
            )

    @router.message(Command("chats"))
    async def business_chats_handler(message: Message) -> None:
        if message.chat.type != "private":
            await message.reply(
                "Список Business-чатов доступен в личном чате с ботом."
            )
            return
        user = message.from_user
        if user is None or not await _is_business_owner(processor, user.id):
            await message.reply(
                "Эта настройка доступна владельцу подключённого Telegram Business."
            )
            return
        text, keyboard = await _business_chats_menu(processor, user.id)
        await message.reply(text, reply_markup=keyboard)

    @router.callback_query(F.data.startswith("biz_ar:"))
    async def business_chat_toggle_handler(callback: CallbackQuery) -> None:
        user = callback.from_user
        if not await _is_business_owner(processor, user.id):
            await callback.answer("Недостаточно прав.", show_alert=True)
            return
        try:
            chat_id = int((callback.data or "").removeprefix("biz_ar:"))
        except ValueError:
            await callback.answer("Некорректная настройка.", show_alert=True)
            return
        if not await processor.storage.is_known_business_chat(user.id, chat_id):
            await callback.answer("Этот чат больше недоступен.", show_alert=True)
            return

        global_enabled = await processor.storage.get_business_auto_reply_enabled(
            user.id, processor.settings.business_auto_reply_enabled
        )
        if not global_enabled:
            await callback.answer(
                "Сначала включите общие ответы командой /autoreply on.",
                show_alert=True,
            )
            return

        current = await processor.storage.get_business_chat_auto_reply_enabled(
            user.id, chat_id
        )
        enabled = not current
        await processor.storage.set_business_chat_auto_reply_enabled(
            user.id, chat_id, enabled
        )
        text, keyboard = await _business_chats_menu(processor, user.id)
        if isinstance(callback.message, Message):
            await callback.message.edit_text(text, reply_markup=keyboard)
        state = "автоответы включены" if enabled else "только мониторинг"
        await callback.answer(state)

    @router.message(Command("status"))
    @router.business_message(Command("status"))
    async def status_handler(message: Message) -> None:
        if not await processor.can_respond(message):
            return
        mode = "private"
        if message.chat.type != "private":
            mode = await processor.storage.get_group_mode(
                message.chat.id, processor.settings.group_default_mode
            )
        scope_key = processor.scope_key(message)
        if processor.settings.ai_provider == "openrouter":
            has_context = await processor.storage.has_conversation_history(scope_key)
            context_storage = (
                "локально в SQLite, до "
                f"{processor.settings.openrouter_history_turns} обменов"
            )
            fallback = (
                f"Groq / {processor.settings.groq_model} — готов"
                if processor.settings.groq_fallback_ready
                else "Groq — не настроен"
            )
        else:
            interaction_id = await processor.storage.get_interaction_id(scope_key)
            has_context = bool(interaction_id)
            context_storage = (
                "на стороне Gemini"
                if processor.settings.gemini_store_interactions
                else "не используется"
            )
            fallback = "не используется"
        context = "есть" if has_context else "пуст"
        await message.reply(
            f"Версия: {__version__}\n"
            f"Провайдер: {processor.settings.ai_provider}\n"
            f"Модель: {processor.settings.active_model}\n"
            f"Резерв: {fallback}\n"
            f"Режим чата: {mode}\n"
            f"Контекст: {context}\n"
            f"Серверное хранение контекста: {context_storage}"
        )

    @router.message(Command("users"))
    async def users_handler(message: Message, command: CommandObject) -> None:
        user = message.from_user
        if message.chat.type != "private" or user is None:
            await message.reply("Отчёт доступен в личном чате с ботом.")
            return
        if not _is_bot_owner(processor, user.id):
            await message.reply("Статистика доступна только владельцу бота.")
            return
        limit = _parse_users_limit(command.args)
        records = await processor.storage.list_bot_users(limit)
        stats = await processor.storage.get_bot_usage_stats()
        if not records:
            await message.reply(
                "Другие пользователи пока не обращались к самому боту.\n"
                "Business-собеседники доступны отдельно: /chats"
            )
            return
        lines = [
            f"Пользователи самого бота: {stats.total_users}",
            f"Последние {len(records)} (Business-чаты не включены):",
        ]
        for index, record in enumerate(records, start=1):
            lines.append(_format_bot_user(index, record))
        await message.reply("\n\n".join(lines))

    @router.message(Command("stats"))
    async def stats_handler(message: Message) -> None:
        user = message.from_user
        if message.chat.type != "private" or user is None:
            await message.reply("Отчёт доступен в личном чате с ботом.")
            return
        if not _is_bot_owner(processor, user.id):
            await message.reply("Статистика доступна только владельцу бота.")
            return
        stats = await processor.storage.get_bot_usage_stats()
        await message.reply(
            f"Статистика бота — версия {__version__}\n\n"
            f"Уникальных пользователей: {stats.total_users}\n"
            f"Активны за 24 часа: {stats.active_24h}\n"
            f"Активны за 7 дней: {stats.active_7d}\n"
            f"Активны за 30 дней: {stats.active_30d}\n"
            f"Последний источник — личный чат: {stats.private_users}\n"
            f"Последний источник — группа: {stats.group_users}\n\n"
            f"Учтённых обращений: {stats.interaction_count}\n"
            f"AI-запросов: {stats.ai_request_count}\n"
            f"Через OpenRouter: {stats.openrouter_request_count}\n"
            f"Через резерв Groq: {stats.groq_request_count}\n"
            f"Напрямую Google: {stats.google_request_count}\n\n"
            "Business-собеседники считаются отдельно и доступны через /chats."
        )

    @router.message()
    @router.business_message()
    async def message_handler(message: Message) -> None:
        if message.media_group_id:
            await albums.add(message, processor.process)
        else:
            await processor.process([message])

    return router


async def _can_manage_shared_context(
    processor: MessageProcessor, message: Message
) -> bool:
    if (
        message.chat.type == "private"
        or processor.settings.conversation_scope == "chat_thread_user"
    ):
        return True
    return await _is_admin(processor, message)


async def _is_admin(processor: MessageProcessor, message: Message) -> bool:
    if message.sender_chat is not None and message.sender_chat.id == message.chat.id:
        return True
    user = message.from_user
    if user is None:
        return False
    if user.id in processor.settings.owner_ids:
        return True
    try:
        member = await processor.bot.get_chat_member(message.chat.id, user.id)
    except Exception:
        return False
    status = getattr(member.status, "value", member.status)
    return status in {"creator", "administrator"}


async def _is_business_owner(processor: MessageProcessor, user_id: int) -> bool:
    if user_id in processor.settings.owner_ids:
        return True
    return await processor.storage.is_business_owner(user_id)


def _is_bot_owner(processor: MessageProcessor, user_id: int) -> bool:
    """Authorize global bot reports only through the explicit OWNER_IDS list."""
    return user_id in processor.settings.owner_ids


async def _business_chats_menu(
    processor: MessageProcessor, owner_user_id: int
) -> tuple[str, InlineKeyboardMarkup | None]:
    global_enabled = await processor.storage.get_business_auto_reply_enabled(
        owner_user_id, processor.settings.business_auto_reply_enabled
    )
    chats = await processor.storage.list_business_chats(owner_user_id, limit=20)
    global_state = "включены" if global_enabled else "выключены для всех чатов"
    if not chats:
        return (
            f"Общие Business-автоответы: {global_state}.\n\n"
            "Список пока пуст. Он появится после первого входящего сообщения "
            "в подключённом Business-чате.",
            None,
        )

    rows: list[list[InlineKeyboardButton]] = []
    for chat in chats:
        if not global_enabled:
            icon = "⏸"
            state = "глобально выкл."
        elif chat.auto_reply_enabled:
            icon = "✅"
            state = "отвечает"
        else:
            icon = "🔕"
            state = "мониторинг"
        name = _truncate_button_text(chat.sender_name, 32)
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{icon} {name} — {state}",
                    callback_data=f"biz_ar:{chat.chat_id}",
                )
            ]
        )

    text = (
        f"Общие Business-автоответы: {global_state}.\n\n"
        "Нажмите на собеседника, чтобы переключить его чат.\n"
        "✅ — автоответы; 🔕 — только мониторинг.\n"
        "Показаны последние 20 диалогов."
    )
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


def _truncate_button_text(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def _parse_users_limit(argument: str | None) -> int:
    if not argument:
        return 10
    try:
        return max(1, min(15, int(argument.strip())))
    except ValueError:
        return 10


def _format_bot_user(index: int, record: BotUserRecord) -> str:
    username = f"@{record.username}" if record.username else "без username"
    if record.last_chat_type == "private":
        source = "личный чат"
    else:
        title = _truncate_button_text(record.last_chat_title or "группа", 40)
        source = f"группа «{title}»"
    first_seen = datetime.fromtimestamp(
        record.first_seen_at, timezone(timedelta(hours=3))
    ).strftime("%d.%m.%Y %H:%M")
    last_seen = datetime.fromtimestamp(
        record.last_seen_at, timezone(timedelta(hours=3))
    ).strftime("%d.%m.%Y %H:%M")
    provider_parts: list[str] = []
    if record.openrouter_request_count:
        provider_parts.append(f"OpenRouter: {record.openrouter_request_count}")
    if record.groq_request_count:
        provider_parts.append(f"Groq: {record.groq_request_count}")
    if record.google_request_count:
        provider_parts.append(f"Google: {record.google_request_count}")
    providers = ", ".join(provider_parts) if provider_parts else "AI ещё не вызывался"
    name = _truncate_button_text(record.display_name, 60)
    return (
        f"{index}. {name} ({username})\n"
        f"ID: {record.user_id}; источник: {source}\n"
        f"Первое обращение: {first_seen}; последнее: {last_seen} МСК\n"
        f"Обращений: {record.interaction_count}; AI: {record.ai_request_count} "
        f"({providers})"
    )
