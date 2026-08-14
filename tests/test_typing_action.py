from __future__ import annotations

import asyncio
from types import SimpleNamespace

from aiogram.enums import MessageEntityType
from aiogram.types import MessageEntity

from bot.emoji_theme import ThemedText
from bot.gemini import ModelSwitchNotice
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
        self.edit_kwargs: list[dict[str, object]] = []
        self.deleted = False
        self.delete_failures = 0

    async def edit_text(self, text: str, **kwargs: object) -> None:
        self.edits.append(text)
        self.edit_kwargs.append(kwargs)

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


class FakeEmojiTheme:
    async def service_text(self, text: str) -> ThemedText:
        return ThemedText(
            text=text,
            entities=[
                MessageEntity(
                    type=MessageEntityType.CUSTOM_EMOJI,
                    offset=0,
                    length=1,
                    custom_emoji_id="5000000000000000001",
                )
            ],
        )


async def test_show_progress_animates_and_cleans_up() -> None:
    message = FakeSourceMessage()

    async with show_progress(message, interval=0.01):
        await asyncio.sleep(0.025)

    assert message.replies[0][0] == PROGRESS_FRAMES[0]
    assert message.replies[0][1]["entities"][0].custom_emoji_id == (
        "5345988476716226868"
    )
    assert message.status.edits
    assert message.status.edit_kwargs[0]["entities"][0].custom_emoji_id == (
        "5231012545799666522"
    )
    assert message.status.deleted is True
    assert message.bot.actions


async def test_show_progress_displays_fallback_and_then_deletes_it() -> None:
    message = FakeSourceMessage()

    async with show_progress(message, interval=10.0) as progress:
        await progress.show_fallback(
            ModelSwitchNotice(
                source_provider="openrouter",
                source_model="google/gemini-3.5-flash",
                target_provider="groq",
                target_model="llama-3.1-8b-instant",
                reason="limit",
            )
        )

    assert "Лимит модели OpenRouter" in message.status.edits[-1]
    assert "Переключаюсь на Groq" in message.status.edits[-1]
    assert message.status.deleted is True


async def test_fallback_status_uses_saved_custom_emoji() -> None:
    message = FakeSourceMessage()

    async with show_progress(
        message,
        interval=10.0,
        emoji_theme=FakeEmojiTheme(),  # type: ignore[arg-type]
    ) as progress:
        await progress.show_fallback(
            ModelSwitchNotice(
                source_provider="google",
                source_model="gemini-test",
                target_provider="openrouter",
                target_model="openrouter/free",
                reason="limit",
            )
        )

    assert message.status.edits[-1].startswith("⚡ ")
    assert message.status.edit_kwargs[-1]["entities"][0].custom_emoji_id == (
        "5000000000000000001"
    )


async def test_show_progress_displays_blocked_fallback_reason() -> None:
    message = FakeSourceMessage()

    async with show_progress(message, interval=10.0) as progress:
        await progress.show_fallback(
            ModelSwitchNotice(
                source_provider="openrouter",
                source_model="openrouter/free",
                target_provider="groq",
                target_model="llama-3.1-8b-instant",
                reason="blocked",
            )
        )

    assert "не смогла обработать этот запрос" in message.status.edits[-1]
    assert "Переключаюсь на Groq" in message.status.edits[-1]
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
