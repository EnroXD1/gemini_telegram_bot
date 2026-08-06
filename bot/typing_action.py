from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from aiogram.enums import ChatAction
from aiogram.types import Message

logger = logging.getLogger(__name__)

PROGRESS_FRAMES = (
    "⏳ Обрабатываю запрос",
    "🔎 Анализирую данные",
    "✍️ Готовлю ответ",
)


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


async def _progress_loop(status: Message, interval: float) -> None:
    frame_index = 1
    while True:
        await asyncio.sleep(interval)
        try:
            await status.edit_text(PROGRESS_FRAMES[frame_index])
            frame_index = (frame_index + 1) % len(PROGRESS_FRAMES)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Could not animate progress message: %s", type(exc).__name__)


@asynccontextmanager
async def show_progress(
    message: Message, interval: float = 3.0
) -> AsyncIterator[None]:
    """Show Telegram typing and a temporary, animated progress message."""
    status: Message | None = None
    progress_task: asyncio.Task[None] | None = None

    async with keep_typing(message):
        try:
            status = await message.reply(
                PROGRESS_FRAMES[0], allow_sending_without_reply=True
            )
            progress_task = asyncio.create_task(
                _progress_loop(status, interval), name="telegram-progress"
            )
        except Exception as exc:
            logger.debug("Could not send progress message: %s", type(exc).__name__)

        try:
            yield
        finally:
            if progress_task is not None:
                progress_task.cancel()
                await asyncio.gather(progress_task, return_exceptions=True)
            if status is not None:
                try:
                    await status.delete()
                except Exception as exc:
                    logger.debug(
                        "Could not delete progress message: %s", type(exc).__name__
                    )
                    with suppress(Exception):
                        await status.edit_text("✅ Обработка завершена")
