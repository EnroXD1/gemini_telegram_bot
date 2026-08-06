from __future__ import annotations

import asyncio
from types import SimpleNamespace

from bot.typing_action import PROGRESS_FRAMES, show_progress


class FakeBot:
    def __init__(self) -> None:
        self.actions: list[dict[str, object]] = []
        self.business_deletions: list[dict[str, object]] = []
        self.fail_business_deletion = False

    async def send_chat_action(self, **kwargs: object) -> None:
        self.actions.append(kwargs)

    async def delete_business_messages(self, **kwargs: object) -> None:
        self.business_deletions.append(kwargs)
        if self.fail_business_deletion:
            raise RuntimeError("temporary Business deletion error")


class FakeStatusMessage:
    def __init__(self, bot: FakeBot) -> None:
        self.bot = bot
        self.message_id = 456
        self.business_connection_id: str | None = None
        self.edits: list[str] = []
        self.deleted = False
        self.delete_failures = 0

    async def edit_text(self, text: str) -> None:
        self.edits.append(text)

    async def delete(self) -> None:
        if self.delete_failures > 0:
            self.delete_failures -= 1
            raise RuntimeError("temporary deletion error")
        self.deleted = True


class FakeSourceMessage:
    def __init__(self) -> None:
        self.bot = FakeBot()
        self.chat = SimpleNamespace(id=123)
        self.message_thread_id = None
        self.business_connection_id = None
        self.status = FakeStatusMessage(self.bot)
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


async def test_show_progress_uses_business_delete_method() -> None:
    message = FakeSourceMessage()
    message.business_connection_id = "business-123"
    message.status.delete_failures = 1

    async with show_progress(message, interval=10.0):
        pass

    assert message.bot.business_deletions == [
        {
            "business_connection_id": "business-123",
            "message_ids": [456],
        }
    ]
    assert "✅ Обработка завершена" not in message.status.edits


async def test_completed_status_is_deleted_after_transient_failure() -> None:
    message = FakeSourceMessage()
    message.business_connection_id = "business-123"
    message.status.delete_failures = 1
    message.bot.fail_business_deletion = True

    async with show_progress(
        message,
        interval=10.0,
        completed_status_ttl=0.01,
    ):
        pass
    await asyncio.sleep(0.03)

    assert message.status.edits[-1] == "✅ Обработка завершена"
    assert message.status.deleted is True
