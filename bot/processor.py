from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import Message, User

from .config import Settings
from .gemini import GeminiRequestError, GeminiService
from .markdown import FormattedChunk, render_markdown_chunks
from .media import MediaExtractor
from .models import ConversationMessage, PromptBundle
from .rate_limit import SlidingWindowRateLimiter
from .scope import build_scope_key
from .storage import Storage
from .text_utils import remove_command, strip_bot_mention
from .typing_action import show_progress

logger = logging.getLogger(__name__)


class MessageProcessor:
    def __init__(
        self,
        *,
        bot: Bot,
        bot_user: User,
        settings: Settings,
        storage: Storage,
        gemini: GeminiService,
        media: MediaExtractor,
    ) -> None:
        self.bot = bot
        self.bot_user = bot_user
        self.settings = settings
        self.storage = storage
        self.gemini = gemini
        self.media = media
        self._rate_limiter = SlidingWindowRateLimiter(
            settings.rate_limit_requests, settings.rate_limit_window_seconds
        )
        self._scope_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._active_tasks: dict[str, asyncio.Task[object]] = {}

    async def process(self, messages: list[Message], *, force: bool = False) -> None:
        if not messages:
            return
        ordered = sorted(messages, key=lambda item: item.message_id)
        lead = ordered[0]
        sender = lead.from_user

        if sender is not None and sender.is_bot and lead.sender_chat is None:
            return
        if await _is_outgoing_business_message(self.storage, lead):
            return
        if not self._is_chat_allowed(lead):
            return
        if not any(_has_processable_content(item) for item in ordered):
            return

        raw_text = _collect_text(ordered)
        explicit_ask = _is_ask_command(raw_text, self.bot_user.username)
        if raw_text.startswith("/") and not explicit_ask and not force:
            return

        if not await self._should_answer(ordered, force=force or explicit_ask):
            return

        user_id = _effective_sender_id(lead)
        if user_id not in self.settings.owner_ids:
            retry_after = self._rate_limiter.check(f"{lead.chat.id}:{user_id}")
            if retry_after is not None:
                await self.reply(
                    lead,
                    f"Слишком много запросов. Повторите примерно через "
                    f"{max(1, math.ceil(retry_after))} сек.",
                )
                return

        scope_key = self.scope_key(lead)
        clean_text = raw_text
        if explicit_ask:
            clean_text = remove_command(clean_text, "ask", self.bot_user.username)
        if lead.chat.type != "private":
            clean_text = strip_bot_mention(clean_text, self.bot_user.username)

        lock = self._scope_locks[scope_key]
        async with lock:
            current_task = asyncio.current_task()
            if current_task is not None:
                self._active_tasks[scope_key] = current_task
            try:
                previous_id = await self.storage.get_interaction_id(scope_key)
                history: tuple[ConversationMessage, ...] = ()
                if self.settings.ai_provider == "openrouter":
                    try:
                        history = await self.storage.get_conversation_history(
                            scope_key=scope_key,
                            max_exchanges=self.settings.openrouter_history_turns,
                            max_chars=self.settings.openrouter_history_max_chars,
                        )
                    except Exception:
                        logger.exception(
                            "Could not load local conversation history scope=%s",
                            scope_key,
                        )
                async with show_progress(lead):
                    bundle = await self.media.prepare(
                        bot=self.bot, messages=ordered, user_text=clean_text
                    )
                    result = await self.gemini.generate(
                        bundle, previous_id, history=history
                    )

                try:
                    if self.settings.ai_provider == "openrouter":
                        await self.storage.append_conversation_exchange(
                            scope_key=scope_key,
                            user_content=_history_user_content(
                                bundle,
                                self.settings.openrouter_history_item_chars,
                            ),
                            assistant_content=_trim_history_item(
                                result.text,
                                self.settings.openrouter_history_item_chars,
                            ),
                            max_exchanges=self.settings.openrouter_history_turns,
                        )
                    else:
                        await self.storage.set_interaction_id(
                            scope_key, result.interaction_id
                        )
                except Exception:
                    logger.exception(
                        "Could not persist conversation context chat_id=%s",
                        lead.chat.id,
                    )
                await self.reply(lead, result.text)
            except GeminiRequestError as exc:
                await self.reply(lead, exc.user_message)
            except asyncio.CancelledError:
                logger.info("Gemini request cancelled scope=%s", scope_key)
            except Exception:
                logger.exception(
                    "Unhandled message processing error chat_id=%s message_id=%s",
                    lead.chat.id,
                    lead.message_id,
                )
                await self.reply(
                    lead,
                    "Не удалось обработать сообщение из-за внутренней ошибки. "
                    "Попробуйте ещё раз позднее.",
                )
            finally:
                if self._active_tasks.get(scope_key) is current_task:
                    self._active_tasks.pop(scope_key, None)

    async def _should_answer(self, messages: list[Message], *, force: bool) -> bool:
        lead = messages[0]
        if lead.chat.type == "private":
            return True
        if force:
            return True

        mode = await self.storage.get_group_mode(
            lead.chat.id, self.settings.group_default_mode
        )
        if mode == "off":
            return False
        if mode == "all":
            return True

        username = (self.bot_user.username or "").casefold()
        for message in messages:
            text = (message.text or message.caption or "").casefold()
            if username and f"@{username}" in text:
                return True
            replied = message.reply_to_message
            if (
                replied is not None
                and replied.from_user is not None
                and replied.from_user.id == self.bot_user.id
            ):
                return True
        return False

    def _is_chat_allowed(self, message: Message) -> bool:
        if not self.settings.allowed_chat_ids:
            return True
        sender_id = _effective_sender_id(message)
        return (
            message.chat.id in self.settings.allowed_chat_ids
            or sender_id in self.settings.owner_ids
        )

    def scope_key(self, message: Message) -> str:
        return build_scope_key(
            chat_id=message.chat.id,
            thread_id=message.message_thread_id,
            user_id=_effective_sender_id(message),
            chat_type=str(message.chat.type),
            scope_mode=self.settings.conversation_scope,
            business_connection_id=message.business_connection_id,
        )

    async def reset(self, message: Message) -> None:
        scope_key = self.scope_key(message)
        async with self._scope_locks[scope_key]:
            await self.storage.reset_conversation(scope_key)

    def cancel(self, message: Message) -> bool:
        task = self._active_tasks.get(self.scope_key(message))
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def reply(self, message: Message, text: str) -> None:
        chunks = render_markdown_chunks(text, self.settings.reply_chunk_size)
        if not chunks:
            chunks = [FormattedChunk("Gemini не вернул текстовый ответ.")]
        for index, chunk in enumerate(chunks):
            entities = list(chunk.entities) or None
            while True:
                try:
                    if index == 0:
                        await message.reply(
                            chunk.text,
                            entities=entities,
                            allow_sending_without_reply=True,
                        )
                    else:
                        await message.answer(chunk.text, entities=entities)
                    break
                except TelegramRetryAfter as exc:
                    await asyncio.sleep(max(0.1, float(exc.retry_after)))
                except TelegramBadRequest as exc:
                    if entities is None:
                        raise
                    logger.warning(
                        "Telegram rejected formatted entities; retrying as plain text: %s",
                        type(exc).__name__,
                    )
                    entities = None
            if index + 1 < len(chunks):
                await asyncio.sleep(0.05)


def _collect_text(messages: list[Message]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for message in messages:
        text = (message.text or message.caption or "").strip()
        if text and text not in seen:
            seen.add(text)
            parts.append(text)
    return "\n".join(parts)


def _is_ask_command(text: str, username: str | None) -> bool:
    if not text.startswith("/ask"):
        return False
    first = text.split(maxsplit=1)[0].casefold()
    valid = {"/ask"}
    if username:
        valid.add(f"/ask@{username.casefold().lstrip('@')}")
    return first in valid


def _has_processable_content(message: Message) -> bool:
    attributes = (
        "text",
        "caption",
        "photo",
        "live_photo",
        "animation",
        "audio",
        "voice",
        "video",
        "video_note",
        "sticker",
        "document",
        "location",
        "venue",
        "contact",
        "poll",
        "dice",
        "game",
        "checklist",
        "rich_message",
        "paid_media",
        "story",
    )
    return any(bool(getattr(message, name, None)) for name in attributes)


def _effective_sender_id(message: Message) -> int:
    if message.sender_chat is not None and (
        message.from_user is None or message.from_user.is_bot
    ):
        return message.sender_chat.id
    return message.from_user.id if message.from_user else 0


async def _is_outgoing_business_message(
    storage: Storage, message: Message
) -> bool:
    connection_id = message.business_connection_id
    if not connection_id:
        return False
    if message.sender_business_bot is not None:
        return True
    sender = message.from_user
    if sender is None:
        return False
    connection = await storage.get_business_connection(connection_id)
    if connection is not None:
        return sender.id == connection.owner_user_id
    return str(message.chat.type) == "private" and sender.id != message.chat.id


def _history_user_content(bundle: PromptBundle, limit: int) -> str:
    parts: list[str] = []
    if bundle.media:
        labels = "; ".join(item.label for item in bundle.media)
        parts.append(
            "В этом предыдущем сообщении были вложения, которые повторно не "
            f"передаются модели: {labels}."
        )
    parts.append(bundle.prompt)
    return _trim_history_item("\n\n".join(parts), limit)


def _trim_history_item(value: str, limit: int) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    marker = "\n[…сокращено для локальной истории…]\n"
    head = max(1, (limit - len(marker)) // 3)
    tail = max(1, limit - len(marker) - head)
    return text[:head] + marker + text[-tail:]
