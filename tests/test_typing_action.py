from __future__ import annotations

import asyncio
from types import SimpleNamespace

from bot.typing_action import PROGRESS_FRAMES, show_progress


class FakeBot:
    def __init__(self) -> None:
        self.actions: list[dict[str, object]] = []

    async def send_chat_action(self, **kwargs: object) -> None:
        self.actions.append(kwargs)


class FakeStatusMessage:
    def __init__(self) -> None:
        self.edits: list[str] = []
        self.deleted = False

    async def edit_text(self, text: str) -> None:
        self.edits.append(text)

    async def delete(self) -> None:
        self.deleted = True


class FakeSourceMessage:
    def __init__(self) -> None:
        self.bot = FakeBot()
        self.chat = SimpleNamespace(id=123)
        self.message_thread_id = None
        self.business_connection_id = None
        self.status = FakeStatusMessage()
        self.replies: list[tuple[str, dict[str, object]]] = []

    async def reply(self, text: str, **kwargs: object) -> FakeStatusMessage:
        self.replies.append((text, kwargs))
        return self.status


async def test_show_progress_animates_and_cleans_up() -> None:
    message = FakeSourceMessage()

    async with show_progress(message, interval=0.01):
        await asyncio.sleep(0.025)

    assert message.replies[0][0] == PROGRESS_FRAMES[0]
    assert message.status.edits
    assert message.status.deleted is True
    assert message.bot.actions


async def test_show_progress_displays_fallback_and_then_deletes_it() -> None:
    message = FakeSourceMessage()

    async with show_progress(message, interval=10.0) as progress:
        await progress.show_fallback("llama-3.1-8b-instant")

    assert "Переключаюсь на резервную модель Groq" in message.status.edits[-1]
    assert message.status.deleted is True
