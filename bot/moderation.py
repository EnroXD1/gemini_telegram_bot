from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from aiogram import F, Router
from aiogram.dispatcher.middlewares.base import BaseMiddleware
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.filters.command import CommandObject
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    MessageOriginUser,
    TelegramObject,
    User,
)

from .activity import ActivityTracker, activity_text, history_pages
from .storage import BotUserRecord, Storage

if TYPE_CHECKING:
    from .processor import MessageProcessor

_PAGE_SIZE = 5
_MOSCOW = timezone(timedelta(hours=3))


class BanMiddleware(BaseMiddleware):
    """Stop access before command filters, usage tracking or downloading media."""

    def __init__(
        self,
        storage: Storage,
        owner_ids: frozenset[int],
        activity: ActivityTracker | None = None,
    ) -> None:
        self.storage = storage
        self.owner_ids = owner_ids
        self.activity = activity

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user = getattr(event, "from_user", None)
        message = event.message if isinstance(event, CallbackQuery) else event
        business = isinstance(message, Message) and message.business_connection_id
        if (
            user is not None
            and not business
            and user.id not in self.owner_ids
            and await self.storage.is_user_banned(user.id)
        ):
            if self.activity:
                if isinstance(event, Message) and await self.activity.is_addressed(event):
                    token = (event.text or "").split(maxsplit=1)[0] if event.text else ""
                    command = token.split("@", maxsplit=1)[0].casefold()
                    kind = "command" if command.startswith("/") and command != "/ask" else "ai"
                    await self.activity.record(
                        [event],
                        kind=kind,
                        outcome="rejected",
                        reason="banned",
                    )
                elif isinstance(event, CallbackQuery):
                    await self.activity.denied_callback(event)
            if isinstance(event, CallbackQuery):
                await event.answer("Доступ к боту ограничен владельцем.", show_alert=True)
            return None
        return await handler(event, data)


def _short(value: str, limit: int = 60) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _user_id(value: str) -> int:
    if not re.fullmatch(r"[1-9][0-9]{0,18}", value) or int(value) >= 2**63:
        raise ValueError("Укажите положительный Telegram ID пользователя.")
    return int(value)


def _target(message: Message, argument: str | None) -> tuple[int, User | None]:
    if argument and argument.strip():
        return _user_id(argument.strip()), None
    replied = message.reply_to_message
    if replied is None:
        raise ValueError(
            "Укажите ID: /ban 123456789 или /unban 123456789. "
            "Также можно отправить команду ответом на сообщение пользователя."
        )
    if replied.forward_origin is not None:
        if not isinstance(replied.forward_origin, MessageOriginUser):
            raise ValueError("Telegram скрыл ID автора пересылки. Укажите его числовой ID.")
        user = replied.forward_origin.sender_user
    else:
        if replied.sender_chat is not None:
            raise ValueError("Это сообщение от имени канала или анонимного администратора.")
        user = replied.from_user
    if user is None or user.is_bot:
        raise ValueError("Ответьте на сообщение человека или укажите его числовой ID.")
    return _user_id(str(user.id)), user


def _users_limit(argument: str | None) -> int:
    try:
        return max(1, min(15, int((argument or "10").strip())))
    except ValueError:
        return 10


def _nav(
    action: str, target: int = 0, section: str = "u", limit: int = 10, page: int = 0
) -> str:
    return f"mod:{action}:{target}:{section}:{limit}:{page}"


def _button(text: str, data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=data)


def _format_user(index: int, record: BotUserRecord) -> str:
    username = f"@{_short(record.username, 32)}" if record.username else "без username"
    source = (
        "личный чат"
        if record.last_chat_type == "private"
        else "inline (чат назначения неизвестен)"
        if record.last_chat_type == "inline"
        else (f"группа «{_short(record.last_chat_title or 'группа', 40)}»")
    )
    first = datetime.fromtimestamp(record.first_seen_at, _MOSCOW).strftime("%d.%m.%Y %H:%M")
    last = datetime.fromtimestamp(record.last_seen_at, _MOSCOW).strftime("%d.%m.%Y %H:%M")
    providers = (
        ", ".join(
            f"{name}: {count}"
            for name, count in (
                ("OpenRouter", record.openrouter_request_count),
                ("Groq", record.groq_request_count),
                ("Google", record.google_request_count),
            )
            if count
        )
        or "AI ещё не вызывался"
    )
    state = "🚫 заблокирован" if record.is_banned else "✅ доступ разрешён"
    return (
        f"{index}. {_short(record.display_name)} ({username}) — {state}\n"
        f"ID: {record.user_id}; источник: {source}\n"
        f"Первое обращение: {first}; последнее: {last} МСК\n"
        f"Обращений: {record.interaction_count}; AI: {record.ai_request_count} ({providers})"
    )


async def users_view(
    storage: Storage,
    *,
    section: str = "u",
    limit: int = 10,
    page: int = 0,
) -> tuple[str, InlineKeyboardMarkup]:
    rows: list[list[InlineKeyboardButton]] = []
    if section == "b":
        count = await storage.count_user_bans()
        pages = max(1, (count + _PAGE_SIZE - 1) // _PAGE_SIZE)
        page = min(page, pages - 1)
        bans = await storage.list_user_bans(_PAGE_SIZE, page * _PAGE_SIZE)
        lines = [f"Заблокированных пользователей: {count}"]
        if not bans:
            lines.append("Список блокировок пуст.")
        for index, ban in enumerate(bans, page * _PAGE_SIZE + 1):
            name = _short(ban.display_name or str(ban.user_id))
            date = datetime.fromtimestamp(ban.banned_at, _MOSCOW).strftime("%d.%m.%Y %H:%M")
            lines.append(
                f"{index}. {name}\nID: {ban.user_id}; заблокирован {date} МСК\n"
                f"Кем: {ban.banned_by}"
            )
            rows.append(
                [
                    _button(
                        f"🚫 {_short(name, 30)} · {ban.user_id}",
                        _nav("view", ban.user_id, section, limit, page),
                    )
                ]
            )
    else:
        records = await storage.list_bot_users(limit)
        stats = await storage.get_bot_usage_stats()
        pages = max(1, (len(records) + _PAGE_SIZE - 1) // _PAGE_SIZE)
        page = min(page, pages - 1)
        start = page * _PAGE_SIZE
        lines = [
            f"Пользователи самого бота: {stats.total_users}\n"
            f"Последние {len(records)} (Business-чаты не включены):"
        ]
        if not records:
            lines.append(
                "Другие пользователи пока не обращались к самому боту.\n"
                "Business-собеседники доступны отдельно: /chats"
            )
        for index, record in enumerate(records[start : start + _PAGE_SIZE], start + 1):
            lines.append(_format_user(index, record))
            icon = "🚫" if record.is_banned else "👤"
            rows.append(
                [
                    _button(
                        f"{icon} {_short(record.display_name, 30)} · {record.user_id}",
                        _nav("view", record.user_id, section, limit, page),
                    )
                ]
            )
    navigation = []
    if page:
        navigation.append(_button("⬅️ Предыдущие", _nav("list", 0, section, limit, page - 1)))
    if page + 1 < pages:
        navigation.append(_button("Следующие ➡️", _nav("list", 0, section, limit, page + 1)))
    if navigation:
        rows.append(navigation)
    other = "u" if section == "b" else "b"
    rows.append(
        [_button("👥 Пользователи" if other == "u" else "🚫 Бан-лист", _nav("list", 0, other))]
    )
    lines.append(f"Страница {page + 1}/{pages}. Выберите пользователя для управления доступом.")
    return "\n\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


async def _card(
    processor: MessageProcessor,
    target: int,
    section: str,
    limit: int,
    page: int,
) -> tuple[str, InlineKeyboardMarkup]:
    record = await processor.storage.get_bot_user(target)
    ban = await processor.storage.get_user_ban(target)
    name = record.display_name if record else (ban.display_name if ban else None)
    lines = [f"👤 {_short(name or str(target))}", f"Telegram ID: {target}"]
    if record:
        lines.append(
            f"Обращений: {record.interaction_count}; AI-запросов: {record.ai_request_count}"
        )
    protected = target in processor.settings.owner_ids or target == processor.bot_user.id
    state = "🚫 заблокирован" if ban else "✅ доступ разрешён"
    lines.append(f"Состояние: {state}")
    if protected:
        lines.append("Владелец и сам бот защищены от блокировки.")
    rows = []
    if ban is not None:
        rows.append([_button("✅ Разблокировать", _nav("unban", target, section, limit, page))])
    elif not protected and record is not None:
        rows.append([_button("🚫 Заблокировать", _nav("ban", target, section, limit, page))])
    rows.append([_button("⬅️ К списку", _nav("list", 0, section, limit, page))])
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


def create_moderation_router(processor: MessageProcessor) -> Router:
    router = Router(name="user-moderation")

    def is_owner(message: Message) -> bool:
        return bool(
            message.from_user
            and not message.from_user.is_bot
            and message.from_user.id in processor.settings.owner_ids
            and not message.business_connection_id
            and not message.sender_chat
        )

    @router.message(Command("history", "activity"))
    async def private_audit_handler(message: Message, command: CommandObject) -> None:
        # Deliberately absent from command registration, public help and keyboards.
        if (
            message.chat.type != "private"
            or not is_owner(message)
            or message.chat.id != message.from_user.id
        ):
            return
        try:
            target = _user_id((command.args or "").strip())
        except ValueError:
            await message.reply(f"Формат: /{command.command} 123456789", parse_mode=None)
            return
        if command.command.casefold() == "history":
            records = await processor.storage.list_user_events(target)
            for text in history_pages(target, records):
                await message.answer(text, parse_mode=None)
        else:
            now = int(time.time())
            day = await processor.storage.get_user_activity_stats(target, since=now - 86_400)
            week = await processor.storage.get_user_activity_stats(
                target,
                since=now - 7 * 86_400,
            )
            await message.reply(activity_text(target, day, week), parse_mode=None)

    @router.message(Command("ban", "unban"))
    async def ban_handler(message: Message, command: CommandObject) -> None:
        if not is_owner(message):
            await message.reply("Управлять блокировками может только владелец бота.")
            return
        try:
            target, user = _target(message, command.args)
        except ValueError as exc:
            await message.reply(str(exc))
            return
        banning = command.command.casefold() == "ban"
        if banning and (
            target in processor.settings.owner_ids or target == processor.bot_user.id
        ):
            await message.reply("Нельзя заблокировать владельца или самого бота.")
            return
        if banning:
            changed = await processor.storage.ban_user(
                target,
                message.from_user.id,
                display_name=user.full_name if user else None,
                username=user.username if user else None,
            )
            processor.cancel_user(target)
            state = "заблокирован" if changed else "уже заблокирован"
        else:
            changed = await processor.storage.unban_user(target)
            state = "разблокирован" if changed else "не был заблокирован"
        await message.reply(
            f"Пользователь {target} {state}.\n"
            "Блокировка относится к доступу к боту, а не к членству в группе."
        )

    @router.message(Command("users", "banlist"))
    async def users_handler(message: Message, command: CommandObject) -> None:
        if (
            message.chat.type != "private"
            or not is_owner(message)
            or message.chat.id != message.from_user.id
        ):
            await message.reply("Список доступен только владельцу в личном чате с ботом.")
            return
        section = "b" if command.command.casefold() == "banlist" else "u"
        text, markup = await users_view(
            processor.storage, section=section, limit=_users_limit(command.args)
        )
        await message.reply(text, reply_markup=markup, parse_mode=None)

    @router.callback_query(F.data.startswith("mod:"))
    async def moderation_callback(callback: CallbackQuery) -> None:
        message = callback.message
        if (
            not isinstance(message, Message)
            or message.chat.type != "private"
            or message.chat.id != callback.from_user.id
            or message.business_connection_id
            or callback.from_user.id not in processor.settings.owner_ids
        ):
            await callback.answer(
                "Управление доступно только владельцу в личном чате.", show_alert=True
            )
            return
        try:
            prefix, action, raw_target, section, raw_limit, raw_page = (
                callback.data or ""
            ).split(":")
            if (
                prefix != "mod"
                or action not in {"list", "view", "ban", "unban"}
                or section not in {"u", "b"}
            ):
                raise ValueError
            target = 0 if action == "list" and raw_target == "0" else _user_id(raw_target)
            limit, page = int(raw_limit), int(raw_page)
            if not 1 <= limit <= 15 or not 0 <= page <= 1_000_000:
                raise ValueError
        except ValueError:
            await callback.answer(
                "Некорректная кнопка. Откройте /users или /banlist.", show_alert=True
            )
            return
        if action != "list":
            known = await processor.storage.get_bot_user(target)
            ban = await processor.storage.get_user_ban(target)
            if known is None and ban is None:
                await callback.answer(
                    "Пользователь больше не найден. Обновите список.", show_alert=True
                )
                return
        if action == "ban":
            if target in processor.settings.owner_ids or target == processor.bot_user.id:
                await callback.answer(
                    "Нельзя заблокировать владельца или самого бота.", show_alert=True
                )
                return
            await processor.storage.ban_user(target, callback.from_user.id)
            processor.cancel_user(target)
        elif action == "unban":
            await processor.storage.unban_user(target)
        await callback.answer()
        if action == "list":
            text, markup = await users_view(
                processor.storage, section=section, limit=limit, page=page
            )
        else:
            text, markup = await _card(processor, target, section, limit, page)
        try:
            await message.edit_text(text, reply_markup=markup, parse_mode=None)
        except TelegramBadRequest as exc:
            if "message is not modified" not in str(exc).lower():
                raise

    return router
