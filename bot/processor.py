from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import Message, User

from .activity import ActivityTracker
from .config import Settings
from .emoji_theme import EmojiTheme
from .gemini import GeminiRequestError, GeminiService
from .keyboards import answer_keyboard
from .markdown import FormattedChunk, render_markdown_chunks
from .media import MediaExtractor
from .models import ConversationMessage, PromptBundle
from .rate_limit import SlidingWindowRateLimiter
from .scope import build_scope_key
from .storage import Storage
from .text_utils import remove_command, split_text, strip_bot_mention
from .typing_action import show_progress
from .usage import UsageTracker

logger = logging.getLogger(__name__)

_UNAVAILABLE_BUSINESS_PEER_ERRORS = (
    "BUSINESS_PEER_INVALID",
    "BUSINESS_CONNECTION_INVALID",
    "BUSINESS_CONNECTION_NOT_ALLOWED",
    "BUSINESS_PEER_USAGE_MISSING",
)


class InlineRequestRejected(ValueError):
    def __init__(self, reason: str, retry_after: float) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after


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
        usage: UsageTracker,
        emoji_theme: EmojiTheme | None = None,
        activity: ActivityTracker | None = None,
    ) -> None:
        self.bot = bot
        self.bot_user = bot_user
        self.settings = settings
        self.storage = storage
        self.gemini = gemini
        self.media = media
        self.usage = usage
        self.emoji_theme = emoji_theme
        self.activity = activity or ActivityTracker(
            storage, settings.owner_ids, bot_user.username or "",
        )
        self._rate_limiter = SlidingWindowRateLimiter(
            settings.rate_limit_requests,
            settings.rate_limit_window_seconds,
            violation_limit=settings.spam_violation_limit,
            block_seconds=settings.spam_block_seconds,
        )
        self._scope_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        # Keep the complete attachment lifecycle inside one slot. Limiting only
        # the provider call is too late: queued requests would still retain the
        # downloaded files in RAM while waiting for the AI service.
        self._attachment_semaphore = asyncio.Semaphore(1)
        self._active_tasks: dict[str, asyncio.Task[object]] = {}
        self._active_task_users: dict[str, int] = {}
        self._active_users: set[str] = set()
        self._cleanup_tasks: set[asyncio.Task[None]] = set()

    async def process(self, messages: list[Message], *, force: bool = False) -> None:
        if not messages:
            return
        ordered = sorted(messages, key=lambda item: item.message_id)
        lead = ordered[0]
        sender = lead.from_user

        if sender is not None and sender.is_bot and lead.sender_chat is None:
            return
        if not await self.can_respond(lead):
            if not lead.business_connection_id:
                await self.activity.record(ordered, outcome="rejected", reason="banned")
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

        event_id = await self.activity.record(ordered)
        user_id = _effective_sender_id(lead)
        user_request_key: str | None = None
        if user_id not in self.settings.owner_ids:
            user_request_key = str(user_id)
            if user_request_key in self._active_users:
                penalty = self._rate_limiter.violate(user_request_key)
                await self.activity.finish(event_id, "rejected", "busy")
                if penalty > self.settings.rate_limit_window_seconds:
                    text = (
                        "Из-за повторного спама во время обработки доступ к ИИ "
                        "временно ограничен. Попробуйте примерно через "
                        f"{max(1, math.ceil(penalty))} сек."
                    )
                else:
                    text = (
                        "Ваш предыдущий запрос ещё обрабатывается. Дождитесь ответа, "
                        "прежде чем отправлять следующий."
                    )
                await self.reply(
                    lead,
                    text,
                )
                return
            retry_after = self._rate_limiter.check(user_request_key)
            if retry_after is not None:
                await self.activity.finish(event_id, "rejected", "rate_limit")
                if retry_after > self.settings.rate_limit_window_seconds:
                    text = (
                        "Из-за слишком частых запросов доступ к ИИ временно "
                        "ограничен. Попробуйте примерно через "
                        f"{max(1, math.ceil(retry_after))} сек."
                    )
                else:
                    text = (
                        "Слишком много запросов. Повторите примерно через "
                        f"{max(1, math.ceil(retry_after))} сек."
                    )
                await self.reply(
                    lead,
                    text,
                )
                return
            self._active_users.add(user_request_key)

        scope_key = self.scope_key(lead)
        clean_text = raw_text
        if explicit_ask:
            clean_text = remove_command(clean_text, "ask", self.bot_user.username)
        if lead.chat.type != "private":
            clean_text = strip_bot_mention(clean_text, self.bot_user.username)

        lock = self._scope_locks[scope_key]
        outcome, reason = "cancelled", None
        try:
            await lock.acquire()
        except BaseException:
            if user_request_key is not None:
                self._active_users.discard(user_request_key)
            await self.activity.finish(event_id, "cancelled")
            raise
        try:
            current_task = asyncio.current_task()
            if current_task is not None:
                self._active_tasks[scope_key] = current_task
                if not lead.business_connection_id:
                    self._active_task_users[scope_key] = user_id
            try:
                if not await self.can_respond(lead):
                    outcome, reason = "rejected", "banned"
                    return
                previous_id = await self.storage.get_interaction_id(scope_key)
                history: tuple[ConversationMessage, ...] = ()
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
                has_attachment = _has_attachment_input(ordered)
                if has_attachment:
                    await self._attachment_semaphore.acquire()
                bundle = PromptBundle(prompt="")
                history_user_content = ""
                try:
                    if not await self.can_respond(lead):
                        outcome, reason = "rejected", "banned"
                        return
                    async with show_progress(
                        lead, emoji_theme=self.emoji_theme
                    ) as progress:
                        bundle = await self.media.prepare(
                            bot=self.bot, messages=ordered, user_text=clean_text
                        )
                        result = await self.gemini.generate(
                            bundle,
                            previous_id,
                            history=history,
                            on_fallback=progress.show_fallback,
                        )
                        await self.usage.record(
                            lead,
                            ai_provider=(
                                result.provider or self.gemini.current_provider
                            ),
                        )
                        history_user_content = _history_user_content(
                            bundle,
                            self.settings.openrouter_history_item_chars,
                        )

                finally:
                    # Drop the raw attachment bytes before another media request
                    # is allowed to start downloading.
                    bundle = PromptBundle(prompt="")
                    if has_attachment:
                        self._attachment_semaphore.release()
                if not await self.can_respond(lead):
                    outcome, reason = "rejected", "banned"
                    return
                sent = await self.reply(
                    lead,
                    result.text,
                    formatted=not result.truncated,
                    controls=True,
                )
                if not sent:
                    outcome = "error"
                    return
                outcome = "completed"
                try:
                    await self.storage.append_conversation_exchange(
                        scope_key=scope_key,
                        user_content=history_user_content,
                        assistant_content=_trim_history_item(
                            result.text,
                            self.settings.openrouter_history_item_chars,
                        ),
                        max_exchanges=self.settings.openrouter_history_turns,
                    )
                    await self.storage.set_interaction_id(
                        scope_key,
                        (
                            result.interaction_id
                            if result.provider == "google"
                            else None
                        ),
                    )
                except Exception:
                    logger.exception(
                        "Could not persist conversation context chat_id=%s",
                        lead.chat.id,
                    )
            except GeminiRequestError as exc:
                outcome = "error"
                if not await self.can_respond(lead):
                    return
                await self.usage.record(
                    lead, ai_provider=self.gemini.current_provider
                )
                sent = await self.reply(lead, exc.user_message)
                if exc.delete_after_seconds is not None:
                    self._schedule_deletion(sent, exc.delete_after_seconds)
            except asyncio.CancelledError:
                logger.info("Gemini request cancelled scope=%s", scope_key)
            except Exception:
                outcome = "error"
                logger.exception(
                    "Unhandled message processing error chat_id=%s message_id=%s",
                    lead.chat.id,
                    lead.message_id,
                )
                if not await self.can_respond(lead):
                    return
                await self.reply(
                    lead,
                    "Не удалось обработать сообщение из-за внутренней ошибки. "
                    "Попробуйте ещё раз позднее.",
                )
            finally:
                if self._active_tasks.get(scope_key) is current_task:
                    self._active_tasks.pop(scope_key, None)
                    self._active_task_users.pop(scope_key, None)
                if user_request_key is not None:
                    self._active_users.discard(user_request_key)
                await self.activity.finish(event_id, outcome, reason)
        finally:
            lock.release()

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

    async def can_respond(self, message: Message) -> bool:
        """Return whether this update may produce a reply to the interlocutor."""
        sender = message.from_user
        if (
            not message.business_connection_id
            and sender is not None
            and sender.id not in self.settings.owner_ids
            and await self.storage.is_user_banned(sender.id)
        ):
            return False
        if await _is_outgoing_business_message(self.storage, message):
            return False
        connection_id = message.business_connection_id
        if not connection_id:
            return True
        connection = await self.storage.get_business_connection(connection_id)
        if connection is None:
            return True
        return await self.storage.get_effective_business_auto_reply_enabled(
            owner_user_id=connection.owner_user_id,
            chat_id=message.chat.id,
            global_default=self.settings.business_auto_reply_enabled,
        )

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

    def cancel_user(self, user_id: int) -> int:
        """Cancel this user's active bot requests without touching Business chats."""
        cancelled = 0
        for scope, sender_id in self._active_task_users.items():
            task = self._active_tasks.get(scope)
            if sender_id == user_id and task is not None and not task.done():
                task.cancel()
                cancelled += 1
        return cancelled

    @asynccontextmanager
    async def inline_request(self, user_id: int) -> AsyncIterator[None]:
        """Share bot-wide per-user admission and owner bans with inline AI requests."""
        key = str(user_id)
        guarded = user_id not in self.settings.owner_ids
        if guarded:
            if key in self._active_users:
                raise InlineRequestRejected("busy", self._rate_limiter.violate(key))
            retry_after = self._rate_limiter.check(key)
            if retry_after is not None:
                raise InlineRequestRejected("rate_limit", retry_after)
            self._active_users.add(key)
        task = asyncio.current_task()
        scope = f"inline:{user_id}:{id(task)}"
        if task is not None:
            self._active_tasks[scope] = task
            self._active_task_users[scope] = user_id
        try:
            yield
        finally:
            self._active_tasks.pop(scope, None)
            self._active_task_users.pop(scope, None)
            if guarded:
                self._active_users.discard(key)

    async def reply(
        self, message: Message, text: str, *, formatted: bool = True, controls: bool = False
    ) -> list[Message]:
        if formatted:
            chunks = render_markdown_chunks(text, self.settings.reply_chunk_size)
        else:
            chunks = [
                FormattedChunk(chunk)
                for chunk in split_text(text, self.settings.reply_chunk_size)
            ]
        if not chunks:
            chunks = [FormattedChunk("Gemini не вернул текстовый ответ.")]
        sent_messages: list[Message] = []
        for index, chunk in enumerate(chunks):
            entities = list(chunk.entities) or None
            kwargs = {}
            if (controls and index == len(chunks) - 1 and message.chat.type == "private"
                    and not message.business_connection_id):
                kwargs["reply_markup"] = answer_keyboard()
            while True:
                try:
                    if index == 0:
                        sent = await message.reply(
                            chunk.text,
                            entities=entities,
                            allow_sending_without_reply=True,
                            **kwargs,
                        )
                    else:
                        sent = await message.answer(chunk.text, entities=entities, **kwargs)
                    sent_messages.append(sent)
                    break
                except TelegramRetryAfter as exc:
                    await asyncio.sleep(max(0.1, float(exc.retry_after)))
                except TelegramBadRequest as exc:
                    if _is_unavailable_business_peer(message, exc):
                        logger.warning(
                            "Telegram Business peer is unavailable; reply skipped "
                            "chat_id=%s connection_id=%s error=%s",
                            message.chat.id,
                            message.business_connection_id,
                            exc,
                        )
                        return sent_messages
                    if entities is None:
                        raise
                    logger.warning(
                        "Telegram rejected formatted entities; retrying as plain text: %s",
                        type(exc).__name__,
                    )
                    entities = None
            if index + 1 < len(chunks):
                await asyncio.sleep(0.05)
        return sent_messages

    def _schedule_deletion(
        self, messages: list[Message], delay_seconds: float
    ) -> None:
        if not messages:
            return
        task = asyncio.create_task(
            _delete_messages_later(messages, delay_seconds),
            name="telegram-delete-transient-error",
        )
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)


def _is_unavailable_business_peer(
    message: Message, error: TelegramBadRequest
) -> bool:
    if not message.business_connection_id:
        return False
    normalized = str(error).upper().replace(" ", "_").replace("-", "_")
    return any(
        marker in normalized for marker in _UNAVAILABLE_BUSINESS_PEER_ERRORS
    )


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


def _has_attachment_input(messages: list[Message]) -> bool:
    attachment_attributes = (
        "photo",
        "live_photo",
        "animation",
        "audio",
        "voice",
        "video",
        "video_note",
        "sticker",
        "document",
    )
    for message in messages:
        candidates = (
            message,
            getattr(message, "reply_to_message", None),
            getattr(message, "external_reply", None),
        )
        if any(
            candidate is not None
            and any(
                bool(getattr(candidate, attribute, None))
                for attribute in attachment_attributes
            )
            for candidate in candidates
        ):
            return True
    return False


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


async def _delete_messages_later(
    messages: list[Message], delay_seconds: float
) -> None:
    await asyncio.sleep(max(0.0, delay_seconds))
    for message in messages:
        try:
            await message.delete()
        except Exception as exc:
            logger.debug("Could not delete transient error: %s", type(exc).__name__)
