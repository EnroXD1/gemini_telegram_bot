from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from aiogram.enums import ChatAction
from aiogram.types import Message

logger = logging.getLogger(__name__)


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
