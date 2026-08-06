from __future__ import annotations

import asyncio
import io
import json
import logging
import mimetypes
from dataclasses import dataclass
from typing import Any

from aiogram import Bot
from aiogram.types import Message

from .config import Settings
from .models import MediaPayload, PromptBundle

logger = logging.getLogger(__name__)

TEXT_DOCUMENT_MIME_TYPES = {
    "application/json",
    "application/ld+json",
    "application/xml",
    "application/x-yaml",
    "application/yaml",
    "application/csv",
    "application/sql",
}


def classify_mime(mime_type: str) -> str | None:
    mime = (mime_type or "application/octet-stream").lower().split(";", 1)[0]
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("video/"):
        return "video"
    if mime == "application/pdf":
        return "document"
    if mime.startswith("text/") or mime in TEXT_DOCUMENT_MIME_TYPES:
        return "text"
    return None


@dataclass(frozen=True, slots=True)
class _Attachment:
    file_id: str
    file_size: int | None
    file_name: str
    mime_type: str
    label: str


class MediaExtractor:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def prepare(
        self,
        *,
        bot: Bot,
        messages: list[Message],
        user_text: str,
    ) -> PromptBundle:
        ordered = sorted(messages, key=lambda item: item.message_id)
        lead = ordered[0]
        media: list[MediaPayload] = []
        notes: list[str] = []
        total_bytes = 0

        reply = next((item.reply_to_message for item in ordered if item.reply_to_message), None)
        if reply is not None:
            reply_text = _message_text(reply)
            reply_description = [
                "Сообщение, на которое отвечает пользователь:",
                f"Автор исходного сообщения: {_author_name(reply)}.",
            ]
            if reply_text:
                reply_description.append(f"Текст исходного сообщения:\n{reply_text}")
            quote = getattr(lead, "quote", None)
            if quote is not None and getattr(quote, "text", None):
                reply_description.append(f"Процитированный фрагмент: {quote.text}")
            notes.append("\n".join(reply_description))
            total_bytes = await self._append_message_content(
                bot=bot,
                message=reply,
                context="в исходном сообщении",
                media=media,
                notes=notes,
                total_bytes=total_bytes,
            )
        else:
            external_reply = getattr(lead, "external_reply", None)
            if external_reply is not None:
                origin = getattr(external_reply, "origin", None)
                notes.append(
                    "Пользователь отвечает на сообщение из другого чата. "
                    f"Доступные сведения об источнике: {_safe_model_json(origin, 4000)}"
                )
                quote = getattr(lead, "quote", None)
                if quote is not None and getattr(quote, "text", None):
                    notes.append(f"Процитированный фрагмент: {quote.text}")
                total_bytes = await self._append_message_content(
                    bot=bot,
                    message=external_reply,
                    context="во внешнем исходном сообщении",
                    media=media,
                    notes=notes,
                    total_bytes=total_bytes,
                )

        if getattr(lead, "reply_to_story", None) is not None:
            notes.append(
                "Пользователь отвечает на Telegram Story; содержимое Story боту "
                "не предоставлено."
            )
        if getattr(lead, "forward_origin", None) is not None:
            notes.append(
                "Сообщение переслано. Доступные сведения об источнике: "
                + _safe_model_json(lead.forward_origin, 4000)
            )

        for index, message in enumerate(ordered, start=1):
            context = (
                f"в текущем вложении {index} из {len(ordered)}"
                if len(ordered) > 1
                else "в текущем сообщении"
            )
            total_bytes = await self._append_message_content(
                bot=bot,
                message=message,
                context=context,
                media=media,
                notes=notes,
                total_bytes=total_bytes,
            )

        chat_title = getattr(lead.chat, "title", None)
        chat_label = (
            f"{lead.chat.type} «{chat_title}»" if chat_title else str(lead.chat.type)
        )
        prompt_parts = [
            "Контекст Telegram-сообщения:",
            f"Чат: {chat_label}.",
            f"Автор текущего сообщения: {_author_name(lead)}.",
        ]
        if len(ordered) > 1:
            prompt_parts.append(f"Сообщение содержит альбом из {len(ordered)} элементов.")
        if notes:
            prompt_parts.extend(notes)

        normalized_text = user_text.strip()
        if normalized_text:
            prompt_parts.append(f"Текст пользователя:\n{normalized_text}")
        elif media:
            prompt_parts.append(
                "Отдельного текста нет. Определи намерение пользователя по вложению и "
                "дай уместный полезный ответ; если задача неясна, задай короткий вопрос."
            )
        else:
            prompt_parts.append(
                "Доступного текста или поддерживаемого вложения нет. Кратко объясни, "
                "что именно пользователь может прислать или уточнить."
            )

        return PromptBundle(prompt="\n\n".join(prompt_parts), media=tuple(media))

    async def _append_message_content(
        self,
        *,
        bot: Bot,
        message: Message,
        context: str,
        media: list[MediaPayload],
        notes: list[str],
        total_bytes: int,
    ) -> int:
        structured = _structured_message_description(message)
        if structured:
            notes.append(f"Дополнительные данные {context}: {structured}")

        attachment, unsupported_note = _attachment_from_message(message, context)
        if unsupported_note:
            notes.append(unsupported_note)
        if attachment is None:
            return total_bytes

        kind = classify_mime(attachment.mime_type)
        if kind is None:
            notes.append(
                f"Файл «{attachment.file_name}» имеет неподдерживаемый формат "
                f"{attachment.mime_type}; его содержимое модели не передано."
            )
            return total_bytes

        if len(media) >= self._settings.max_media_items:
            notes.append(
                f"Вложение «{attachment.file_name}» не передано модели: достигнут "
                f"лимит {self._settings.max_media_items} файлов на один запрос."
            )
            return total_bytes

        declared_size = attachment.file_size
        if declared_size and declared_size > self._settings.max_media_bytes:
            notes.append(
                f"Вложение «{attachment.file_name}» не передано модели: его размер "
                f"{_human_size(declared_size)} превышает лимит "
                f"{_human_size(self._settings.max_media_bytes)}."
            )
            return total_bytes

        remaining = self._settings.max_media_bytes - total_bytes
        if declared_size and declared_size > remaining:
            notes.append(
                f"Вложение «{attachment.file_name}» не передано модели: превышен "
                "общий лимит размера файлов для одного запроса."
            )
            return total_bytes

        try:
            data = await self._download(bot, attachment.file_id)
        except Exception as exc:  # Telegram/network errors differ between versions.
            logger.warning(
                "Could not download Telegram attachment: %s",
                type(exc).__name__,
            )
            notes.append(
                f"Вложение «{attachment.file_name}» временно не удалось скачать."
            )
            return total_bytes

        if len(data) > remaining or len(data) > self._settings.max_media_bytes:
            notes.append(
                f"Вложение «{attachment.file_name}» не передано модели: фактический "
                "размер превышает установленный лимит."
            )
            return total_bytes

        if kind == "text":
            decoded = _decode_text_file(data)
            truncated = len(decoded) > self._settings.max_text_file_chars
            decoded = decoded[: self._settings.max_text_file_chars]
            suffix = "\n[Файл сокращён из-за лимита.]" if truncated else ""
            notes.append(
                f"Содержимое текстового файла «{attachment.file_name}» {context}:\n"
                f"{decoded}{suffix}"
            )
            return total_bytes + len(data)

        media.append(
            MediaPayload(
                label=attachment.label,
                kind=kind,
                mime_type=attachment.mime_type,
                data=data,
            )
        )
        return total_bytes + len(data)

    async def _download(self, bot: Bot, file_id: str) -> bytes:
        buffer = io.BytesIO()
        async with asyncio.timeout(self._settings.media_download_timeout_seconds):
            await bot.download(file_id, destination=buffer)
        return buffer.getvalue()


def _attachment_from_message(
    message: Message, context: str
) -> tuple[_Attachment | None, str | None]:
    if message.animation is not None:
        item = message.animation
        mime = item.mime_type or "image/gif"
        return _make_attachment(item, item.file_name or "animation", mime, context), None

    live_photo = getattr(message, "live_photo", None)
    if live_photo is not None:
        return _make_attachment(
            live_photo,
            "live-photo.mp4",
            live_photo.mime_type or "video/mp4",
            context,
        ), None

    if message.audio is not None:
        item = message.audio
        return _make_attachment(
            item, item.file_name or "audio", item.mime_type or "audio/mpeg", context
        ), None

    if message.voice is not None:
        item = message.voice
        return _make_attachment(
            item, "voice-message.ogg", item.mime_type or "audio/ogg", context
        ), None

    if message.video_note is not None:
        return _make_attachment(
            message.video_note, "video-note.mp4", "video/mp4", context
        ), None

    if message.video is not None:
        item = message.video
        return _make_attachment(
            item, item.file_name or "video.mp4", item.mime_type or "video/mp4", context
        ), None

    if message.sticker is not None:
        item = message.sticker
        emoji = item.emoji or ""
        if item.is_animated:
            return None, (
                f"Пользователь прислал анимированный стикер {emoji!r} {context}; "
                "формат TGS не передан модели."
            )
        mime = "video/webm" if item.is_video else "image/webp"
        name = "sticker.webm" if item.is_video else "sticker.webp"
        return _make_attachment(item, name, mime, context, extra=f" (эмодзи {emoji})"), None

    if message.photo:
        item = message.photo[-1]
        return _make_attachment(item, "photo.jpg", "image/jpeg", context), None

    if message.document is not None:
        item = message.document
        name = item.file_name or "document"
        guessed_mime = mimetypes.guess_type(name)[0]
        return _make_attachment(
            item,
            name,
            item.mime_type or guessed_mime or "application/octet-stream",
            context,
        ), None

    return None, None


def _make_attachment(
    item: Any,
    name: str,
    mime: str,
    context: str,
    *,
    extra: str = "",
) -> _Attachment:
    return _Attachment(
        file_id=str(item.file_id),
        file_size=getattr(item, "file_size", None),
        file_name=name,
        mime_type=mime,
        label=f"Файл «{name}» {context}{extra}",
    )


def _message_text(message: Message) -> str:
    return (message.text or message.caption or "").strip()


def _author_name(message: Message) -> str:
    if (
        message.sender_chat is not None
        and (message.from_user is None or message.from_user.is_bot)
    ):
        return message.sender_chat.title or str(message.sender_chat.id)
    if message.from_user is not None:
        user = message.from_user
        name = user.full_name
        if user.username:
            return f"{name} (@{user.username})"
        return name
    if message.sender_chat is not None:
        return message.sender_chat.title or str(message.sender_chat.id)
    return "неизвестен"


def _structured_message_description(message: Message) -> str:
    if getattr(message, "paid_media", None) is not None:
        return "платное медиа; его закрытое содержимое боту недоступно"
    if getattr(message, "story", None) is not None:
        return "Telegram Story; содержимое Story боту недоступно"
    if message.venue is not None:
        venue = message.venue
        return (
            f"место «{venue.title}», адрес: {venue.address}, координаты: "
            f"{venue.location.latitude}, {venue.location.longitude}"
        )
    if message.location is not None:
        location = message.location
        return f"геолокация: {location.latitude}, {location.longitude}"
    if message.contact is not None:
        contact = message.contact
        full_name = " ".join(
            part for part in (contact.first_name, contact.last_name) if part
        )
        return f"контакт: {full_name}, телефон: {contact.phone_number}"
    if message.poll is not None:
        poll = message.poll
        options = "; ".join(
            f"{option.text} — {option.voter_count} голосов" for option in poll.options
        )
        return f"опрос «{poll.question}»; варианты: {options}"
    if message.dice is not None:
        return f"бросок {message.dice.emoji}: выпало {message.dice.value}"
    if message.game is not None:
        return f"игра «{message.game.title}»: {message.game.description}"
    checklist = getattr(message, "checklist", None)
    if checklist is not None:
        return "чек-лист: " + _safe_model_json(checklist)
    rich_message = getattr(message, "rich_message", None)
    if rich_message is not None:
        return "структурированное сообщение: " + _safe_model_json(rich_message)
    return ""


def _safe_model_json(value: Any, limit: int = 20_000) -> str:
    try:
        if hasattr(value, "model_dump"):
            raw = json.dumps(
                value.model_dump(mode="json", exclude_none=True), ensure_ascii=False
            )
        else:
            raw = str(value)
    except Exception:
        raw = str(value)
    return raw[:limit]


def _decode_text_file(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig", errors="replace")
    for encoding in ("utf-8", "utf-16", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _human_size(value: int) -> str:
    size = float(value)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if size < 1024 or unit == "ГБ":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} Б"
