from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
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
    BusinessMediaArchiveRecord,
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
        self._archive_root = settings.database_path.parent / "business_media"

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
        attachment = archive_attachment_from_message(message)
        async with self._chat_locks[_chat_key(connection_id, message.chat.id)]:
            await self.storage.upsert_business_message(record)
            if (
                record.is_incoming
                and self.settings.business_archive_media
                and attachment is not None
            ):
                await self._archive_attachment(message, connection, attachment)
            elif self.settings.business_archive_media:
                await self._save_media_from_owner_reply(message, connection)

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
        auto_reply_enabled = (
            await self.storage.get_effective_business_auto_reply_enabled(
                owner_user_id=connection.owner_user_id,
                chat_id=message.chat.id,
                global_default=self.settings.business_auto_reply_enabled,
            )
        )
        if not auto_reply_enabled:
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
            archives = await self.storage.get_business_media_archives(
                connection_id=event.business_connection_id,
                chat_id=event.chat.id,
                message_ids=event.message_ids,
            )
            archive_by_message_id = {archive.message_id: archive for archive in archives}
            for record in records:
                if not record.is_incoming or record.deleted_at is not None:
                    continue
                archive = archive_by_message_id.get(record.message_id)
                media_delivered = False
                if archive is not None:
                    media_delivered = await self._send_deleted_media(
                        chat_id=connection.owner_chat_id,
                        record=record,
                        archive=archive,
                    )
                    if media_delivered:
                        await self._remove_archived_media(archive)

                notification_delivered = False
                if not media_delivered:
                    notification_delivered = await self._send_notification(
                        connection.owner_chat_id,
                        _deleted_message_text(record),
                    )
                if media_delivered or notification_delivered:
                    await self.storage.mark_business_message_deleted(
                        connection_id=record.connection_id,
                        chat_id=record.chat_id,
                        message_id=record.message_id,
                    )

            known_ids = {record.message_id for record in records}
            missing_ids = [item for item in event.message_ids if item not in known_ids]
            if missing_ids:
                logger.info(
                    "Ignoring %s unknown Business deletion(s) connection=%s chat_id=%s",
                    len(missing_ids),
                    event.business_connection_id,
                    event.chat.id,
                )

    async def prune_archived_media(self, retention_days: int) -> int:
        archives = await self.storage.pop_expired_business_media_archives(retention_days)
        for archive in archives:
            await self._unlink_archive_file(archive)
        return len(archives)

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
            logger.info(
                "Skipping oversized business media chat_id=%s message_id=%s "
                "size=%s limit=%s",
                message.chat.id,
                message.message_id,
                attachment.file_size,
                limit,
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
                logger.info(
                    "Skipping oversized downloaded business media chat_id=%s "
                    "message_id=%s size=%s limit=%s",
                    message.chat.id,
                    message.message_id,
                    len(data),
                    limit,
                )
                return
            relative_path = _archive_relative_path(
                connection.connection_id,
                message.chat.id,
                message.message_id,
                attachment.file_name,
            )
            archive_path = self._resolve_archive_path(relative_path)
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = archive_path.with_name(f"{archive_path.name}.part")
            try:
                await asyncio.to_thread(temporary_path.write_bytes, data)
                await asyncio.to_thread(temporary_path.replace, archive_path)
            finally:
                if temporary_path.exists():
                    await asyncio.to_thread(temporary_path.unlink)

            previous = await self.storage.upsert_business_media_archive(
                BusinessMediaArchiveRecord(
                    connection_id=connection.connection_id,
                    chat_id=message.chat.id,
                    message_id=message.message_id,
                    kind=attachment.kind,
                    file_name=_safe_file_name(attachment.file_name),
                    file_path=relative_path,
                    file_size=len(data),
                    created_at=int(time.time()),
                )
            )
            if previous is not None and previous.file_path != relative_path:
                await self._unlink_archive_file(previous)
            logger.info(
                "Archived business media silently chat_id=%s message_id=%s kind=%s",
                message.chat.id,
                message.message_id,
                attachment.kind,
            )
        except TimeoutError:
            logger.warning(
                "Timed out archiving business media chat_id=%s message_id=%s",
                message.chat.id,
                message.message_id,
            )
        except Exception as exc:
            logger.warning(
                "Could not archive business media chat_id=%s message_id=%s: %s",
                message.chat.id,
                message.message_id,
                type(exc).__name__,
            )

    async def _save_media_from_owner_reply(
        self,
        message: Message,
        connection: BusinessConnectionRecord,
    ) -> None:
        """Save media explicitly requested by the owner through a reply.

        Bot API doesn't expose the self-destruct timer itself. It does expose
        ``has_protected_content`` for messages Telegram doesn't allow to be saved,
        which is the only reliable Bot API signal available for view-once media.
        Fail closed when the flag is absent so an ordinary photo, video or video
        note never creates a duplicate just because the owner replied to it.
        """
        if message.sender_business_bot is not None:
            return
        if message.from_user is None or message.from_user.id != connection.owner_user_id:
            return
        replied = message.reply_to_message
        if replied is None:
            return
        attachment = archive_attachment_from_message(replied)
        if attachment is None:
            return
        if not _is_explicitly_protected_media(replied):
            logger.info(
                "Ignoring owner reply to ordinary Business media chat_id=%s "
                "message_id=%s kind=%s media_spoiler=%s",
                replied.chat.id,
                replied.message_id,
                attachment.kind,
                bool(replied.has_media_spoiler),
            )
            return

        original = _record_from_message(replied, connection)
        if not original.is_incoming:
            return
        previous = await self.storage.upsert_business_message(original)
        if previous is not None and previous.deleted_at is not None:
            logger.info(
                "Skipping already delivered replied media chat_id=%s message_id=%s",
                original.chat_id,
                original.message_id,
            )
            return

        archive = await self.storage.get_business_media_archive(
            connection_id=connection.connection_id,
            chat_id=original.chat_id,
            message_id=original.message_id,
        )
        if archive is None:
            await self._archive_attachment(replied, connection, attachment)
            archive = await self.storage.get_business_media_archive(
                connection_id=connection.connection_id,
                chat_id=original.chat_id,
                message_id=original.message_id,
            )
        if archive is None:
            await self._send_notification(
                connection.owner_chat_id,
                "⚠️ Не удалось сохранить медиа из сообщения, на которое вы ответили. "
                "Возможно, Telegram уже закрыл доступ к файлу или превышен лимит размера.",
            )
            return

        delivered = await self._send_archived_media(
            chat_id=connection.owner_chat_id,
            record=original,
            archive=archive,
            caption=_saved_reply_media_caption(original, archive.kind),
        )
        if not delivered:
            await self._send_notification(
                connection.owner_chat_id,
                "⚠️ Медиа скачано, но Telegram не позволил отправить сохранённую копию. "
                "Архив останется до следующей попытки или автоматической очистки.",
            )
            return

        await self._remove_archived_media(archive)
        await self.storage.mark_business_message_deleted(
            connection_id=original.connection_id,
            chat_id=original.chat_id,
            message_id=original.message_id,
        )
        logger.info(
            "Delivered media saved through owner reply chat_id=%s message_id=%s kind=%s",
            original.chat_id,
            original.message_id,
            archive.kind,
        )

    async def _send_deleted_media(
        self,
        *,
        chat_id: int,
        record: BusinessMessageRecord,
        archive: BusinessMediaArchiveRecord,
    ) -> bool:
        return await self._send_archived_media(
            chat_id=chat_id,
            record=record,
            archive=archive,
            caption=_deleted_media_caption(record, archive.kind),
        )

    async def _send_archived_media(
        self,
        *,
        chat_id: int,
        record: BusinessMessageRecord,
        archive: BusinessMediaArchiveRecord,
        caption: str,
    ) -> bool:
        try:
            archive_path = self._resolve_archive_path(archive.file_path)
            data = await asyncio.to_thread(archive_path.read_bytes)
            upload = BufferedInputFile(data, filename=archive.file_name)
            if archive.kind == "photo":
                await self.bot.send_photo(chat_id=chat_id, photo=upload, caption=caption)
            elif archive.kind == "video":
                await self.bot.send_video(chat_id=chat_id, video=upload, caption=caption)
            elif archive.kind == "voice":
                await self.bot.send_voice(chat_id=chat_id, voice=upload, caption=caption)
            elif archive.kind == "video_note":
                await self.bot.send_video_note(chat_id=chat_id, video_note=upload)
                await self._send_notification(chat_id, caption)
            elif archive.kind == "animation":
                await self.bot.send_animation(
                    chat_id=chat_id, animation=upload, caption=caption
                )
            elif archive.kind == "audio":
                await self.bot.send_audio(chat_id=chat_id, audio=upload, caption=caption)
            else:
                await self.bot.send_document(chat_id=chat_id, document=upload, caption=caption)
            return True
        except Exception as exc:
            logger.warning(
                "Could not deliver deleted business media chat_id=%s message_id=%s: %s",
                record.chat_id,
                record.message_id,
                type(exc).__name__,
            )
            return False

    async def _remove_archived_media(self, archive: BusinessMediaArchiveRecord) -> None:
        await self._unlink_archive_file(archive)
        await self.storage.delete_business_media_archive(
            connection_id=archive.connection_id,
            chat_id=archive.chat_id,
            message_id=archive.message_id,
        )

    async def _unlink_archive_file(self, archive: BusinessMediaArchiveRecord) -> None:
        try:
            archive_path = self._resolve_archive_path(archive.file_path)
            await asyncio.to_thread(archive_path.unlink, missing_ok=True)
        except Exception as exc:
            logger.warning(
                "Could not remove archived media chat_id=%s message_id=%s: %s",
                archive.chat_id,
                archive.message_id,
                type(exc).__name__,
            )

    def _resolve_archive_path(self, relative_path: str) -> Path:
        archive_root = self._archive_root.resolve()
        archive_path = (archive_root / relative_path).resolve()
        if archive_path != archive_root and archive_root not in archive_path.parents:
            raise ValueError("Unsafe business archive path")
        return archive_path

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


def _is_explicitly_protected_media(message: Message) -> bool:
    """Return true only for media Telegram explicitly marks as non-saveable.

    ``has_media_spoiler`` is intentionally not considered: a spoiler is ordinary
    media hidden behind a visual cover, not a self-destruct/view-once attachment.
    """
    return message.has_protected_content is True


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


def _deleted_message_text(record: BusinessMessageRecord) -> str:
    return (
        f"🗑 {record.sender_name} удалил(а) сообщение.\n\n"
        f"Удалённое содержимое:\n{record.content}"
    )


def _deleted_media_caption(record: BusinessMessageRecord, kind: str) -> str:
    return _truncate(
        f"🗑 {record.sender_name} удалил(а) {_kind_label(kind)}.\n"
        f"Чат: {record.chat_id}\n\n"
        f"Удалённое содержимое:\n{record.content}",
        1024,
    )


def _saved_reply_media_caption(record: BusinessMessageRecord, kind: str) -> str:
    return _truncate(
        f"⏳ {_kind_label(kind).capitalize()} сохранено по вашему ответу до открытия.\n"
        f"Отправитель: {record.sender_name}\n"
        f"Чат: {record.chat_id}",
        1024,
    )


def _archive_relative_path(
    connection_id: str,
    chat_id: int,
    message_id: int,
    file_name: str,
) -> str:
    connection_key = hashlib.sha256(connection_id.encode("utf-8")).hexdigest()[:16]
    suffix = Path(_safe_file_name(file_name)).suffix[:16] or ".bin"
    return str(Path(connection_key) / str(chat_id) / f"{message_id}{suffix}")


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
