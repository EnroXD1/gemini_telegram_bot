from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import Bot
from aiogram.dispatcher.middlewares.base import BaseMiddleware
from aiogram.types import Message, TelegramObject

from .config import Settings
from .storage import Storage

logger = logging.getLogger(__name__)


class BotUsageMiddleware(BaseMiddleware):
    """Track non-AI commands; AI requests are recorded by MessageProcessor."""

    def __init__(self, tracker: UsageTracker) -> None:
        self._tracker = tracker

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if isinstance(event, Message) and _is_non_ai_command(event):
            await self._tracker.record(event)
        return await handler(event, data)


class UsageTracker:
    def __init__(self, *, bot: Bot, settings: Settings, storage: Storage) -> None:
        self.bot = bot
        self.settings = settings
        self.storage = storage

    async def record(self, message: Message, *, ai_provider: str | None = None) -> None:
        user = message.from_user
        if (
            user is None
            or user.is_bot
            or message.business_connection_id is not None
        ):
            return
        if user.id in self.settings.owner_ids:
            return
        try:
            if await self.storage.is_business_owner(user.id):
                return
            chat_type = getattr(message.chat.type, "value", str(message.chat.type))
            is_new = await self.storage.record_bot_user_activity(
                user_id=user.id,
                username=user.username,
                display_name=user.full_name,
                chat_id=message.chat.id,
                chat_type=chat_type,
                chat_title=message.chat.title,
                ai_provider=ai_provider,
            )
            if is_new:
                await self._notify_owners(message)
        except Exception:
            logger.exception("Could not record bot usage user_id=%s", user.id)

    async def _notify_owners(self, message: Message) -> None:
        user = message.from_user
        if user is None:
            return
        recipients = set(self.settings.owner_ids)
        recipients.update(await self.storage.list_business_owner_chat_ids())
        recipients.discard(user.id)
        if not recipients:
            return

        username = f"@{user.username}" if user.username else "не указан"
        chat_type = getattr(message.chat.type, "value", str(message.chat.type))
        if chat_type == "private":
            source = "личный чат с ботом"
        else:
            title = message.chat.title or str(message.chat.id)
            source = f"группа «{title}»"
        text = (
            "👤 Новый пользователь бота\n\n"
            f"Имя: {user.full_name}\n"
            f"Username: {username}\n"
            f"Telegram ID: {user.id}\n"
            f"Источник: {source}\n\n"
            "Это первое зафиксированное обращение к самому боту. "
            "Business-собеседники показываются отдельно командой /chats."
        )
        for chat_id in recipients:
            try:
                await self.bot.send_message(chat_id=chat_id, text=text)
            except Exception as exc:
                logger.warning(
                    "Could not notify owner about new user chat_id=%s: %s",
                    chat_id,
                    type(exc).__name__,
                )


def _is_non_ai_command(message: Message) -> bool:
    text = (message.text or message.caption or "").strip()
    if not text.startswith("/"):
        return False
    token = text.split(maxsplit=1)[0]
    command = token[1:].split("@", maxsplit=1)[0].casefold()
    return bool(command) and command != "ask"
