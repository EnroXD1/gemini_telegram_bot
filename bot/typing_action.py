from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message

from .emoji_theme import EmojiTheme, apply_service_custom_emojis
from .gemini import ModelSwitchNotice

logger = logging.getLogger(__name__)

PROGRESS_FRAMES = (
    "⏳ Обрабатываю запрос",
    "🔎 Анализирую данные",
    "✍️ Готовлю ответ",
)
COMPLETED_STATUS_TEXT = "✅ Обработка завершена"
COMPLETED_STATUS_TTL_SECONDS = 3.0


async def _typing_loop(message: Message, interval: float) -> None:
    while True:
        try:
            await message.bot.send_chat_action(
                chat_id=message.chat.id,
                action=ChatAction.TYPING,
                message_thread_id=message.message_thread_id,
                business_connection_id=message.business_connection_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Could not send typing action: %s", type(exc).__name__)
        await asyncio.sleep(interval)


@asynccontextmanager
async def keep_typing(message: Message, interval: float = 4.5) -> AsyncIterator[None]:
    task = asyncio.create_task(_typing_loop(message, interval), name="telegram-typing")
    try:
        await asyncio.sleep(0)
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def _progress_loop(
    status: Message,
    interval: float,
    emoji_theme: EmojiTheme | None = None,
) -> None:
    frame_index = 1
    custom_emoji_allowed = True
    while True:
        await asyncio.sleep(interval)
        try:
            frame = PROGRESS_FRAMES[frame_index]
            themed = (
                await emoji_theme.service_text(frame)
                if emoji_theme is not None
                else apply_service_custom_emojis(frame)
            )
            if custom_emoji_allowed:
                try:
                    await status.edit_text(frame, entities=themed.entities)
                except TelegramBadRequest:
                    custom_emoji_allowed = False
                    await status.edit_text(frame)
            else:
                await status.edit_text(frame)
            frame_index = (frame_index + 1) % len(PROGRESS_FRAMES)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Could not animate progress message: %s", type(exc).__name__)


class ProgressReporter:
    def __init__(
        self,
        *,
        business_connection_id: str | None = None,
        completed_status_ttl: float = COMPLETED_STATUS_TTL_SECONDS,
        emoji_theme: EmojiTheme | None = None,
    ) -> None:
        self._status: Message | None = None
        self._animation: asyncio.Task[None] | None = None
        self._business_connection_id = business_connection_id
        self._completed_status_ttl = max(0.0, completed_status_ttl)
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._emoji_theme = emoji_theme

    def bind(
        self, status: Message, animation: asyncio.Task[None]
    ) -> None:
        self._status = status
        self._animation = animation

    async def show_fallback(self, notice: ModelSwitchNotice) -> None:
        """Replace animation with a stable provider-switch notification."""
        if self._animation is not None:
            self._animation.cancel()
            await asyncio.gather(self._animation, return_exceptions=True)
            self._animation = None
        if self._status is None:
            return
        try:
            source = _provider_label(notice.source_provider)
            target = _provider_label(notice.target_provider)
            if notice.reason == "limit":
                headline = (
                    f"⚡ Лимит модели {source} ({notice.source_model}) исчерпан."
                )
            else:
                headline = (
                    f"⚠️ Модель {source} ({notice.source_model}) временно недоступна."
                )
            text = f"{headline}\n🔄 Переключаюсь на {target} ({notice.target_model})…"
            if self._emoji_theme is None:
                themed = apply_service_custom_emojis(text)
                try:
                    await self._status.edit_text(text, entities=themed.entities)
                except TelegramBadRequest:
                    await self._status.edit_text(text)
            else:
                themed = await self._emoji_theme.service_text(text)
                try:
                    await self._status.edit_text(
                        themed.text, entities=themed.entities
                    )
                except TelegramBadRequest:
                    await self._status.edit_text(text)
        except Exception as exc:
            logger.debug("Could not show fallback status: %s", type(exc).__name__)

    async def close(self) -> None:
        if self._animation is not None:
            self._animation.cancel()
            await asyncio.gather(self._animation, return_exceptions=True)
            self._animation = None
        if self._status is not None:
            deleted = await _delete_progress_message(
                self._status,
                business_connection_id=self._business_connection_id,
            )
            if not deleted:
                with suppress(Exception):
                    await self._status.edit_text(COMPLETED_STATUS_TEXT)
                cleanup_task = asyncio.create_task(
                    _delete_progress_message_later(
                        self._status,
                        business_connection_id=self._business_connection_id,
                        delay=self._completed_status_ttl,
                    ),
                    name="telegram-progress-cleanup",
                )
                self._cleanup_tasks.add(cleanup_task)
                cleanup_task.add_done_callback(self._cleanup_tasks.discard)
            self._status = None


async def _delete_progress_message(
    status: Message, *, business_connection_id: str | None
) -> bool:
    try:
        await status.delete()
        return True
    except Exception as direct_exc:
        connection_id = (
            getattr(status, "business_connection_id", None)
            or business_connection_id
        )
        bot = getattr(status, "bot", None)
        message_id = getattr(status, "message_id", None)
        if connection_id and bot is not None and message_id is not None:
            try:
                await bot.delete_business_messages(
                    business_connection_id=connection_id,
                    message_ids=[message_id],
                )
                return True
            except Exception as business_exc:
                logger.debug(
                    "Could not delete Business progress message: direct=%s "
                    "business=%s",
                    type(direct_exc).__name__,
                    type(business_exc).__name__,
                )
        logger.debug(
            "Could not delete progress message: %s", type(direct_exc).__name__
        )
        return False


def _provider_label(provider: str) -> str:
    return {
        "google": "Google/Gemini",
        "openrouter": "OpenRouter",
        "groq": "Groq",
    }.get(provider, provider)


async def _delete_progress_message_later(
    status: Message, *, business_connection_id: str | None, delay: float
) -> None:
    await asyncio.sleep(max(0.0, delay))
    await _delete_progress_message(
        status,
        business_connection_id=business_connection_id,
    )


@asynccontextmanager
async def show_progress(
    message: Message,
    interval: float = 3.0,
    completed_status_ttl: float = COMPLETED_STATUS_TTL_SECONDS,
    emoji_theme: EmojiTheme | None = None,
) -> AsyncIterator[ProgressReporter]:
    """Show Telegram typing and a temporary, animated progress message."""
    reporter = ProgressReporter(
        business_connection_id=message.business_connection_id,
        completed_status_ttl=completed_status_ttl,
        emoji_theme=emoji_theme,
    )

    async with keep_typing(message):
        try:
            first_frame = PROGRESS_FRAMES[0]
            themed = (
                await emoji_theme.service_text(first_frame)
                if emoji_theme is not None
                else apply_service_custom_emojis(first_frame)
            )
            try:
                status = await message.reply(
                    first_frame,
                    entities=themed.entities,
                    allow_sending_without_reply=True,
                )
            except TelegramBadRequest:
                status = await message.reply(
                    first_frame, allow_sending_without_reply=True
                )
            progress_task = asyncio.create_task(
                _progress_loop(status, interval, emoji_theme),
                name="telegram-progress",
            )
            reporter.bind(status, progress_task)
        except Exception as exc:
            logger.debug("Could not send progress message: %s", type(exc).__name__)

        try:
            yield reporter
        finally:
            await reporter.close()
