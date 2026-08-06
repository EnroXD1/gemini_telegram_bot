from __future__ import annotations

import asyncio
import io
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from aiogram import Bot
from aiogram.dispatcher.middlewares.base import BaseMiddleware
from aiogram.types import (
    BufferedInputFile,
    BusinessConnection,
    BusinessMessagesDeleted,
    Message,
    TelegramObject,
)

from .config import Settings
from .storage import (
    BusinessConnectionRecord,
    BusinessMessageRecord,
    Storage,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ArchiveAttachment:
    kind: str
    file_id: str
    file_size: int | None
    file_name: str


class BusinessMessageCaptureMiddleware(BaseMiddleware):
    def __init__(self, monitor: BusinessMonitor) -> None:
        self._monitor = monitor

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if isinstance(event, Message):
            try:
                await self._monitor.capture_message(event)
                await self._monitor.welcome_contact(event)
            except Exception:
                logger.exception(
                    "Could not capture business message chat_id=%s message_id=%s",
                    event.chat.id,
                    event.message_id,
                )
        return await handler(event, data)


class BusinessMonitor:
    def __init__(self, *, bot: Bot, settings: Settings, storage: Storage) -> None:
        self.bot = bot
        self.settings = settings
        self.storage = storage
        self._chat_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def handle_connection(self, connection: BusinessConnection) -> None:
        await self.storage.save_business_connection(
            connection_id=connection.id,
            owner_user_id=connection.user.id,
            owner_chat_id=connection.user_chat_id,
            is_enabled=connection.is_enabled,
        )
        logger.info(
            "Business connection %s owner=%s enabled=%s",
            connection.id,
            connection.user.id,
            connection.is_enabled,
        )

    async def capture_message(self, message: Message) -> None:
        if not self.settings.business_monitor_enabled:
            return
        connection_id = message.business_connection_id
        if not connection_id:
            return
        connection = await self._resolve_connection(connection_id)
        if connection is None or not connection.is_enabled:
            return

        record = _record_from_message(message, connection)
        async with self._chat_locks[_chat_key(connection_id, message.chat.id)]:
            await self.storage.upsert_business_message(record)
        if not record.is_incoming or not self.settings.business_archive_media:
            return
        attachment = archive_attachment_from_message(message)
        if attachment is not None:
            await self._archive_attachment(message, connection, attachment)

    async def welcome_contact(self, message: Message) -> bool:
        """Send the configured disclosure once to each Business interlocutor."""
        if not self.settings.business_welcome_enabled:
            return False
        connection_id = message.business_connection_id
        if not connection_id:
            return False
        connection = await self._resolve_connection(connection_id)
        if connection is None or not connection.is_enabled:
            return False
        if not _record_from_message(message, connection).is_incoming:
            return False

        claimed = await self.storage.claim_business_greeting(
            connection_id=connection_id, chat_id=message.chat.id
        )
        if not claimed:
            return False
        try:
            await self.bot.send_message(
                business_connection_id=connection_id,
                chat_id=message.chat.id,
                text=self.settings.business_welcome_text,
            )
            return True
        except Exception as exc:
            await self.storage.release_business_greeting(
                connection_id=connection_id, chat_id=message.chat.id
            )
            logger.warning(
                "Could not welcome business contact chat_id=%s: %s",
                message.chat.id,
                type(exc).__name__,
            )
            return False

    async def handle_edited_message(self, message: Message) -> None:
        if not self.settings.business_monitor_enabled:
            return
        connection_id = message.business_connection_id
        if not connection_id:
            return
        connection = await self._resolve_connection(connection_id)
        if connection is None or not connection.is_enabled:
            return

        current = _record_from_message(message, connection)
        async with self._chat_locks[_chat_key(connection_id, message.chat.id)]:
            previous = await self.storage.upsert_business_message(current)
        if not current.is_incoming:
            return
        if previous is not None and not previous.is_incoming:
            return
        if (
            previous is not None
            and previous.content == current.content
            and previous.media_kind == current.media_kind
        ):
            return

        if previous is None:
            text = (
                f"✏️ {current.sender_name} изменил(а) сообщение.\n\n"
                "Исходная версия не была сохранена (бот мог быть выключен).\n\n"
                f"Сейчас:\n{current.content}"
            )
        else:
            text = (
                f"✏️ {current.sender_name} изменил(а) сообщение.\n\n"
                f"Было:\n{previous.content}\n\n"
                f"Стало:\n{current.content}"
            )
        await self._send_notification(connection.owner_chat_id, text)

    async def handle_deleted_messages(self, event: BusinessMessagesDeleted) -> None:
        if not self.settings.business_monitor_enabled:
            return
        connection = await self._resolve_connection(event.business_connection_id)
        if connection is None or not connection.is_enabled:
            return

        lock_key = _chat_key(event.business_connection_id, event.chat.id)
        async with self._chat_locks[lock_key]:
            records = await self.storage.get_business_messages(
                connection_id=event.business_connection_id,
                chat_id=event.chat.id,
                message_ids=event.message_ids,
            )
        delivered = 0
        for record in records:
            if not record.is_incoming or record.deleted_at is not None:
                continue
            text = (
                f"🗑 {record.sender_name} удалил(а) сообщение.\n\n"
                f"Удалённое содержимое:\n{record.content}"
            )
            if await self._send_notification(connection.owner_chat_id, text):
                delivered += 1
                await self.storage.mark_business_message_deleted(
                    connection_id=record.connection_id,
                    chat_id=record.chat_id,
                    message_id=record.message_id,
                )

        known_ids = {record.message_id for record in records}
        missing_ids = [item for item in event.message_ids if item not in known_ids]
        if missing_ids and delivered == 0:
            await self._send_notification(
                connection.owner_chat_id,
                "🗑 Telegram сообщил об удалении сообщения в Business-чате, но "
                "автор и исходная версия неизвестны (бот мог быть выключен).",
            )

    async def _resolve_connection(
        self, connection_id: str
    ) -> BusinessConnectionRecord | None:
        stored = await self.storage.get_business_connection(connection_id)
        if stored is not None:
            return stored
        try:
            connection = await self.bot.get_business_connection(connection_id)
        except Exception as exc:
            logger.warning(
                "Could not resolve business connection %s: %s",
                connection_id,
                type(exc).__name__,
            )
            return None
        await self.handle_connection(connection)
        return BusinessConnectionRecord(
            connection_id=connection.id,
            owner_user_id=connection.user.id,
            owner_chat_id=connection.user_chat_id,
            is_enabled=connection.is_enabled,
        )

    async def _archive_attachment(
        self,
        message: Message,
        connection: BusinessConnectionRecord,
        attachment: ArchiveAttachment,
    ) -> None:
        limit = self.settings.business_archive_max_bytes
        if attachment.file_size is not None and attachment.file_size > limit:
            await self._send_notification(
                connection.owner_chat_id,
                f"⚠️ Не удалось сохранить {_kind_label(attachment.kind)} от "
                f"{_sender_name(message)}: размер превышает {_human_size(limit)}.",
            )
            return

        destination = io.BytesIO()
        try:
            await asyncio.wait_for(
                self.bot.download(attachment.file_id, destination=destination),
                timeout=self.settings.media_download_timeout_seconds,
            )
            data = destination.getvalue()
            if len(data) > limit:
                await self._send_notification(
                    connection.owner_chat_id,
                    f"⚠️ Не удалось сохранить {_kind_label(attachment.kind)} от "
                    f"{_sender_name(message)}: размер превышает {_human_size(limit)}.",
                )
                return
            await self._send_archived_media(
                chat_id=connection.owner_chat_id,
                message=message,
                attachment=attachment,
                data=data,
            )
        except TimeoutError:
            await self._send_notification(
                connection.owner_chat_id,
                f"⚠️ Не удалось вовремя скачать {_kind_label(attachment.kind)} от "
                f"{_sender_name(message)}.",
            )
        except Exception as exc:
            logger.warning(
                "Could not archive business media chat_id=%s message_id=%s: %s",
                message.chat.id,
                message.message_id,
                type(exc).__name__,
            )
            await self._send_notification(
                connection.owner_chat_id,
                f"⚠️ Telegram не позволил сохранить {_kind_label(attachment.kind)} "
                f"от {_sender_name(message)}.",
            )

    async def _send_archived_media(
        self,
        *,
        chat_id: int,
        message: Message,
        attachment: ArchiveAttachment,
        data: bytes,
    ) -> None:
        caption = _archive_caption(message, attachment.kind)
        upload = BufferedInputFile(data, filename=_safe_file_name(attachment.file_name))
        if attachment.kind == "photo":
            await self.bot.send_photo(chat_id=chat_id, photo=upload, caption=caption)
        elif attachment.kind == "video":
            await self.bot.send_video(chat_id=chat_id, video=upload, caption=caption)
        elif attachment.kind == "voice":
            await self.bot.send_voice(chat_id=chat_id, voice=upload, caption=caption)
        elif attachment.kind == "video_note":
            await self.bot.send_message(chat_id=chat_id, text=caption)
            await self.bot.send_video_note(chat_id=chat_id, video_note=upload)
        elif attachment.kind == "animation":
            await self.bot.send_animation(
                chat_id=chat_id, animation=upload, caption=caption
            )
        elif attachment.kind == "audio":
            await self.bot.send_audio(chat_id=chat_id, audio=upload, caption=caption)
        else:
            await self.bot.send_document(chat_id=chat_id, document=upload, caption=caption)

    async def _send_notification(self, chat_id: int, text: str) -> bool:
        try:
            await self.bot.send_message(chat_id=chat_id, text=_truncate(text, 4096))
            return True
        except Exception as exc:
            logger.warning(
                "Could not notify business owner chat_id=%s: %s",
                chat_id,
                type(exc).__name__,
            )
            return False


def archive_attachment_from_message(message: Message) -> ArchiveAttachment | None:
    if message.photo:
        item = message.photo[-1]
        return ArchiveAttachment(
            kind="photo",
            file_id=item.file_id,
            file_size=item.file_size,
            file_name="photo.jpg",
        )
    if message.animation is not None:
        item = message.animation
        return ArchiveAttachment(
            kind="animation",
            file_id=item.file_id,
            file_size=item.file_size,
            file_name=item.file_name or "animation.gif",
        )
    if message.video is not None:
        item = message.video
        return ArchiveAttachment(
            kind="video",
            file_id=item.file_id,
            file_size=item.file_size,
            file_name=item.file_name or "video.mp4",
        )
    if message.voice is not None:
        item = message.voice
        return ArchiveAttachment(
            kind="voice",
            file_id=item.file_id,
            file_size=item.file_size,
            file_name="voice-message.ogg",
        )
    if message.video_note is not None:
        item = message.video_note
        return ArchiveAttachment(
            kind="video_note",
            file_id=item.file_id,
            file_size=item.file_size,
            file_name="video-note.mp4",
        )
    if message.audio is not None:
        item = message.audio
        return ArchiveAttachment(
            kind="audio",
            file_id=item.file_id,
            file_size=item.file_size,
            file_name=item.file_name or "audio.mp3",
        )
    if message.document is not None:
        item = message.document
        return ArchiveAttachment(
            kind="document",
            file_id=item.file_id,
            file_size=item.file_size,
            file_name=item.file_name or "document",
        )
    return None


def _record_from_message(
    message: Message, connection: BusinessConnectionRecord
) -> BusinessMessageRecord:
    sender_id = message.from_user.id if message.from_user is not None else 0
    is_incoming = (
        message.sender_business_bot is None and sender_id != connection.owner_user_id
    )
    attachment = archive_attachment_from_message(message)
    return BusinessMessageRecord(
        connection_id=connection.connection_id,
        chat_id=message.chat.id,
        message_id=message.message_id,
        sender_user_id=sender_id,
        sender_name=_sender_name(message),
        is_incoming=is_incoming,
        content=_message_content(message, attachment),
        media_kind=None if attachment is None else attachment.kind,
    )


def _message_content(
    message: Message, attachment: ArchiveAttachment | None = None
) -> str:
    parts: list[str] = []
    if attachment is not None:
        parts.append(f"[{_kind_label(attachment.kind)}]")
    text = (message.text or message.caption or "").strip()
    if text:
        parts.append(text)
    if not parts:
        parts.append("[сообщение без доступного текстового содержимого]")
    return "\n".join(parts)


def _archive_caption(message: Message, kind: str) -> str:
    chat_label = message.chat.title or getattr(message.chat, "full_name", None)
    if not chat_label:
        chat_label = str(message.chat.id)
    caption = (
        f"⏳ Сохранено: {_kind_label(kind)}\n"
        f"От: {_sender_name(message)}\n"
        f"Чат: {chat_label}"
    )
    original_caption = (message.caption or "").strip()
    if original_caption:
        caption += f"\n\nПодпись:\n{original_caption}"
    return _truncate(caption, 1024)


def _sender_name(message: Message) -> str:
    if message.from_user is not None:
        user = message.from_user
        if user.username:
            return f"{user.full_name} (@{user.username})"
        return user.full_name
    if message.sender_chat is not None:
        return message.sender_chat.title or str(message.sender_chat.id)
    return "неизвестный собеседник"


def _kind_label(kind: str) -> str:
    return {
        "photo": "фото",
        "video": "видео",
        "voice": "голосовое сообщение",
        "video_note": "кружок",
        "animation": "GIF/анимация",
        "audio": "аудиофайл",
        "document": "документ",
    }.get(kind, "медиафайл")


def _safe_file_name(value: str) -> str:
    normalized = value.replace("\\", "_").replace("/", "_").strip(" .")
    return (normalized or "telegram-file")[:128]


def _chat_key(connection_id: str, chat_id: int) -> str:
    return f"{connection_id}:{chat_id}"


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    suffix = "\n…"
    return value[: limit - len(suffix)] + suffix


def _human_size(value: int) -> str:
    size = float(value)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if size < 1024 or unit == "ГБ":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} Б"
