from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from aiogram.types import Message

logger = logging.getLogger(__name__)

AlbumCallback = Callable[[list[Message]], Awaitable[None]]


@dataclass(slots=True)
class _AlbumEntry:
    messages: list[Message] = field(default_factory=list)
    callback: AlbumCallback | None = None
    task: asyncio.Task[None] | None = None


class AlbumBuffer:
    """Debounce Telegram media-group updates into one logical request."""

    def __init__(self, delay_seconds: float) -> None:
        self._delay = delay_seconds
        self._entries: dict[str, _AlbumEntry] = {}
        self._lock = asyncio.Lock()

    async def add(self, message: Message, callback: AlbumCallback) -> None:
        if not message.media_group_id:
            await callback([message])
            return

        key = self._key(message)
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = _AlbumEntry(callback=callback)
                self._entries[key] = entry
            entry.messages.append(message)
            if entry.task is not None:
                entry.task.cancel()
            entry.task = asyncio.create_task(
                self._flush_after_delay(key), name=f"album:{message.media_group_id}"
            )

    async def _flush_after_delay(self, key: str) -> None:
        try:
            await asyncio.sleep(self._delay)
            async with self._lock:
                entry = self._entries.pop(key, None)
            if entry is None or entry.callback is None:
                return
            messages = sorted(entry.messages, key=lambda item: item.message_id)
            await entry.callback(messages)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Unhandled error while processing Telegram album")

    async def close(self) -> None:
        async with self._lock:
            tasks = [entry.task for entry in self._entries.values() if entry.task]
            self._entries.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _key(message: Message) -> str:
        business = message.business_connection_id or "regular"
        return f"{message.chat.id}:{business}:{message.media_group_id}"
