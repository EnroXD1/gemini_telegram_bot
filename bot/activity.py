from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from aiogram.dispatcher.middlewares.base import BaseMiddleware
from aiogram.types import CallbackQuery, InlineQuery, Message, TelegramObject

from .storage import Storage, UserActivityRecord, UserActivityStats

logger = logging.getLogger(__name__)
_MOSCOW = timezone(timedelta(hours=3))
_HIDDEN_COMMANDS = {"history", "activity"}
_MEDIA_LABELS = {
    "photo": "фото",
    "document": "документ",
    "video": "видео",
    "voice": "голосовое",
    "audio": "аудио",
    "video_note": "видеосообщение",
    "animation": "анимация",
    "sticker": "стикер",
    "contact": "контакт",
    "location": "геопозиция",
    "poll": "опрос",
    "live_photo": "живое фото",
    "venue": "место",
    "dice": "кубик",
    "game": "игра",
    "checklist": "список задач",
    "paid_media": "платное медиа",
    "story": "история",
}


class ActivityTracker:
    """A bounded moderation journal, separate from AI context and Business snapshots."""

    def __init__(
        self,
        storage: Storage,
        owner_ids: frozenset[int],
        bot_username: str,
        *,
        bot_id: int = 0,
        group_default_mode: str = "mentions",
    ) -> None:
        self.storage = storage
        self.owner_ids = owner_ids
        self.bot_username = bot_username.casefold()
        self.bot_id = bot_id
        self.group_default_mode = group_default_mode
        self._next_prune = 0.0

    async def is_addressed(self, message: Message) -> bool:
        if message.chat.type == "private":
            return True
        text = (message.text or message.caption or "").casefold()
        if text.startswith("/"):
            _, _, mention = text.split(maxsplit=1)[0].partition("@")
            return not mention or mention == self.bot_username
        if self.bot_username and f"@{self.bot_username}" in text:
            return True
        replied = message.reply_to_message
        if replied and replied.from_user and replied.from_user.id == self.bot_id:
            return True
        return (
            await self.storage.get_group_mode(
                message.chat.id,
                self.group_default_mode,
            )
            == "all"
        )

    async def record(
        self,
        messages: Sequence[Message],
        *,
        kind: str = "ai",
        outcome: str = "pending",
        reason: str | None = None,
    ) -> int | None:
        message = messages[0]
        user = message.from_user
        if (
            user is None
            or user.is_bot
            or user.id in self.owner_ids
            or message.business_connection_id
            or message.sender_chat
        ):
            return None
        parts = []
        for item in messages:
            text = item.text or item.caption
            if text:
                parts.append(text)
            labels = [
                label for attr, label in _MEDIA_LABELS.items() if getattr(item, attr, None)
            ]
            if labels:
                parts.append("[" + ", ".join(labels) + "]")
        content = "\n".join(parts) or "[Сообщение без текста]"
        if len(content) > 1000:
            content = content[:999] + "…"
        try:
            await self._maybe_prune()
            return await self.storage.record_user_event(
                user_id=user.id,
                chat_id=message.chat.id,
                message_id=message.message_id,
                kind=kind,
                content=content,
                outcome=outcome,
                reason=reason,
            )
        except Exception:
            logger.exception("Could not record moderation event user_id=%s", user.id)
            return None

    async def finish(
        self,
        event_id: int | None,
        outcome: str,
        reason: str | None = None,
    ) -> None:
        if event_id is None:
            return
        try:
            await self.storage.finish_user_event(event_id, outcome, reason)
        except Exception:
            logger.exception("Could not finish moderation event id=%s", event_id)

    async def record_inline(
        self,
        query: InlineQuery,
        *,
        kind: str = "ai",
        outcome: str = "pending",
        reason: str | None = None,
    ) -> int | None:
        user = query.from_user
        if user.is_bot or user.id in self.owner_ids:
            return None
        try:
            await self._maybe_prune()
            # Telegram supplies neither a destination chat nor a message ID in inline mode.
            return await self.storage.record_user_event(
                user_id=user.id,
                chat_id=0,
                message_id=0,
                kind=kind,
                content="[Inline] " + query.query[:990],
                outcome=outcome,
                reason=reason,
            )
        except Exception:
            logger.exception("Could not record inline event user_id=%s", user.id)
            return None

    async def denied_callback(self, callback: CallbackQuery) -> None:
        if not isinstance(callback.message, Message):
            return
        try:
            await self._maybe_prune()
            await self.storage.record_user_event(
                user_id=callback.from_user.id,
                chat_id=callback.message.chat.id,
                message_id=callback.message.message_id,
                kind="callback",
                content="[Нажатие кнопки]",
                outcome="rejected",
                reason="banned",
            )
        except Exception:
            logger.exception(
                "Could not record rejected callback user=%s", callback.from_user.id
            )

    async def _maybe_prune(self) -> None:
        if time.monotonic() >= self._next_prune:
            await self.storage.prune_user_events()
            self._next_prune = time.monotonic() + 3600


class CommandActivityMiddleware(BaseMiddleware):
    def __init__(self, tracker: ActivityTracker) -> None:
        self.tracker = tracker

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        event_id = None
        if isinstance(event, Message):
            text = (event.text or "").strip()
            token = text.split(maxsplit=1)[0].casefold() if text else ""
            command, _, mention = token.removeprefix("/").partition("@")
            if (
                token.startswith("/")
                and command not in _HIDDEN_COMMANDS | {"ask"}
                and (not mention or mention == self.tracker.bot_username)
            ):
                event_id = await self.tracker.record([event], kind="command")
        try:
            result = await handler(event, data)
        except asyncio.CancelledError:
            await self.tracker.finish(event_id, "cancelled")
            raise
        except Exception:
            await self.tracker.finish(event_id, "error")
            raise
        else:
            await self.tracker.finish(event_id, "completed")
            return result


def activity_text(user_id: int, day: UserActivityStats, week: UserActivityStats) -> str:
    lines = [f"📊 Активность пользователя {user_id}"]
    for title, stats in (("Последние 24 часа", day), ("Последние 7 дней", week)):
        lines.append(
            f"{title}:\n"
            f"Обращений: {stats.total}; AI-запросов: {stats.ai_requests}\n"
            f"Принято: {stats.total - stats.rejected}; отклонено: {stats.rejected}\n"
            f"Антиспам: {stats.busy + stats.rate_limit}\n"
            f"  Повтор во время обработки: {stats.busy}\n"
            f"  Лимит частоты / временная пауза: {stats.rate_limit}\n"
            f"Отклонено из-за бана: {stats.banned}\n"
            f"Ошибки обработки: {stats.errors}; отменено: {stats.cancelled}"
        )
    lines.append(
        "AI-запросы включают отклонённые попытки. Команды учитываются отдельно от AI. "
        "Отказ антиспама — сигнал для проверки, а не доказательство злоупотребления.\n"
        "Учитываются только события после обновления; владельцы и Business не включены."
    )
    return "\n\n".join(lines)


def history_pages(user_id: int, records: list[UserActivityRecord]) -> list[str]:
    title = f"🗂 Последние обращения пользователя {user_id}\nВремя — МСК. Хранение — 30 дней."
    if not records:
        return [title + "\n\nЗаписей нет. Сбор начинается после обновления бота."]
    statuses = {
        "pending": "⏳ в обработке",
        "completed": "✅ обработано",
        "error": "⚠️ ошибка обработки",
        "cancelled": "⏹ отменено / прервано",
    }
    reasons = {
        "busy": "повтор во время обработки",
        "rate_limit": "лимит частоты / пауза",
        "banned": "пользователь заблокирован",
    }
    kinds = {"ai": "AI-запрос", "command": "команда", "callback": "кнопка"}
    pages = [title]
    for record in records:
        stamp = datetime.fromtimestamp(record.created_at, _MOSCOW).strftime("%d.%m.%Y %H:%M:%S")
        status = statuses.get(record.outcome, "🚫 отклонено")
        if record.reason:
            status += ": " + reasons.get(record.reason, record.reason)
        entry = (
            f"\n\n{stamp} · {kinds[record.kind]} · {status}\n"
            + (
                "Источник: inline, чат назначения неизвестен\n"
                if record.chat_id == 0
                else f"Чат: {record.chat_id}; сообщение: {record.message_id}\n"
            )
            + record.content
        )
        if len((pages[-1] + entry).encode("utf-16-le")) // 2 > 3800:
            pages.append(title + " (продолжение)")
        pages[-1] += entry
    return pages
