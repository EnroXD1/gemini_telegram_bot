from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.filters.command import CommandObject
from aiogram.types import BusinessConnection, BusinessMessagesDeleted, Message

from .album import AlbumBuffer
from .business import BusinessMessageCaptureMiddleware, BusinessMonitor
from .processor import MessageProcessor


def create_router(
    processor: MessageProcessor, albums: AlbumBuffer, monitor: BusinessMonitor
) -> Router:
    router = Router(name="gemini-assistant")
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
        await message.reply(
            "Команды:\n"
            "/ask запрос — явно обратиться к боту в группе; команду можно добавить "
            "к подписи файла\n"
            "/reset — начать диалог с чистым контекстом\n"
            "/cancel — остановить текущий запрос\n"
            "/status — показать модель и режим этого чата\n"
            "/mode mentions|all|off — режим группы (только администратор)\n\n"
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
        if message.media_group_id:
            await albums.add(message, processor.process)
        else:
            await processor.process([message], force=True)

    @router.message(Command("reset"))
    @router.business_message(Command("reset"))
    async def reset_handler(message: Message) -> None:
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

    @router.message(Command("status"))
    @router.business_message(Command("status"))
    async def status_handler(message: Message) -> None:
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
        else:
            interaction_id = await processor.storage.get_interaction_id(scope_key)
            has_context = bool(interaction_id)
            context_storage = (
                "на стороне Gemini"
                if processor.settings.gemini_store_interactions
                else "не используется"
            )
        context = "есть" if has_context else "пуст"
        await message.reply(
            f"Провайдер: {processor.settings.ai_provider}\n"
            f"Модель: {processor.settings.active_model}\n"
            f"Режим чата: {mode}\n"
            f"Контекст: {context}\n"
            f"Серверное хранение контекста: {context_storage}"
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
