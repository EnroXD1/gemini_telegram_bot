from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aiogram.enums import ChatType
from aiogram.types import Chat, Message, User

from bot.handlers import _is_bot_owner
from bot.storage import Storage
from bot.usage import UsageTracker, _is_non_ai_command


class FakeBot:
    def __init__(self) -> None:
        self.sent_messages: list[dict[str, Any]] = []

    async def send_message(self, **kwargs: Any) -> None:
        self.sent_messages.append(kwargs)


def make_message(
    *,
    user_id: int = 200,
    text: str = "Привет",
    chat_id: int | None = None,
    chat_type: ChatType = ChatType.PRIVATE,
    business_connection_id: str | None = None,
) -> Message:
    actual_chat_id = user_id if chat_id is None else chat_id
    return Message(
        message_id=10,
        date=datetime.now(UTC),
        chat=Chat(
            id=actual_chat_id,
            type=chat_type,
            first_name="Иван" if chat_type == ChatType.PRIVATE else None,
            title="Тестовая группа" if chat_type != ChatType.PRIVATE else None,
        ),
        from_user=User(
            id=user_id,
            is_bot=False,
            first_name="Иван",
            username="ivan",
        ),
        business_connection_id=business_connection_id,
        text=text,
    )


class UsageStorageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self.temp_dir.name) / "bot.sqlite3")
        await self.storage.open()

    async def asyncTearDown(self) -> None:
        await self.storage.close()
        self.temp_dir.cleanup()

    async def test_activity_and_provider_counters_are_aggregated(self) -> None:
        first = await self.storage.record_bot_user_activity(
            user_id=200,
            username="ivan",
            display_name="Иван",
            chat_id=200,
            chat_type="private",
            chat_title=None,
            ai_provider=None,
        )
        second = await self.storage.record_bot_user_activity(
            user_id=200,
            username="ivan_new",
            display_name="Иван Новый",
            chat_id=200,
            chat_type="private",
            chat_title=None,
            ai_provider="openrouter",
        )
        await self.storage.record_bot_user_activity(
            user_id=201,
            username=None,
            display_name="Анна",
            chat_id=-1001,
            chat_type="supergroup",
            chat_title="Группа",
            ai_provider="google",
        )

        records = {item.user_id: item for item in await self.storage.list_bot_users()}
        stats = await self.storage.get_bot_usage_stats()

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(records[200].username, "ivan_new")
        self.assertEqual(records[200].interaction_count, 2)
        self.assertEqual(records[200].ai_request_count, 1)
        self.assertEqual(records[200].openrouter_request_count, 1)
        self.assertEqual(stats.total_users, 2)
        self.assertEqual(stats.private_users, 1)
        self.assertEqual(stats.group_users, 1)
        self.assertEqual(stats.interaction_count, 3)
        self.assertEqual(stats.ai_request_count, 2)
        self.assertEqual(stats.openrouter_request_count, 1)
        self.assertEqual(stats.google_request_count, 1)

    async def test_new_user_notification_is_sent_only_once(self) -> None:
        await self.storage.save_business_connection(
            connection_id="connection-1",
            owner_user_id=100,
            owner_chat_id=100,
            is_enabled=False,
        )
        bot = FakeBot()
        tracker = UsageTracker(
            bot=bot,
            settings=SimpleNamespace(owner_ids=frozenset({100})),
            storage=self.storage,
        )

        await tracker.record(make_message(text="/start"))
        await tracker.record(make_message(text="Вопрос"), ai_provider="openrouter")
        await tracker.record(make_message(user_id=100, text="Сообщение владельца"))
        await tracker.record(
            make_message(
                user_id=202,
                text="Business-сообщение",
                business_connection_id="connection-1",
            )
        )

        records = await self.storage.list_bot_users()
        self.assertEqual(len(bot.sent_messages), 1)
        self.assertEqual(bot.sent_messages[0]["chat_id"], 100)
        self.assertIn("Новый пользователь", bot.sent_messages[0]["text"])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].user_id, 200)
        self.assertEqual(records[0].interaction_count, 2)
        self.assertEqual(records[0].openrouter_request_count, 1)

    async def test_business_owner_is_not_implicitly_a_global_bot_owner(self) -> None:
        await self.storage.record_bot_user_activity(
            user_id=100,
            username="owner",
            display_name="Владелец",
            chat_id=100,
            chat_type="private",
            chat_title=None,
            ai_provider=None,
        )

        await self.storage.save_business_connection(
            connection_id="connection-1",
            owner_user_id=100,
            owner_chat_id=100,
            is_enabled=True,
        )

        records = await self.storage.list_bot_users()
        self.assertEqual([record.user_id for record in records], [100])

        processor = SimpleNamespace(
            settings=SimpleNamespace(owner_ids=frozenset({101}))
        )
        self.assertFalse(_is_bot_owner(processor, 100))
        self.assertTrue(_is_bot_owner(processor, 101))

    async def test_only_explicit_owner_receives_global_usage_notifications(self) -> None:
        await self.storage.save_business_connection(
            connection_id="connection-1",
            owner_user_id=100,
            owner_chat_id=100,
            is_enabled=True,
        )
        bot = FakeBot()
        tracker = UsageTracker(
            bot=bot,
            settings=SimpleNamespace(owner_ids=frozenset({101})),
            storage=self.storage,
        )

        await tracker.record(make_message(user_id=200, text="/start"))
        await tracker.record(make_message(user_id=100, text="/start"))

        self.assertEqual(
            [message["chat_id"] for message in bot.sent_messages],
            [101, 101],
        )
        records = await self.storage.list_bot_users()
        self.assertEqual({record.user_id for record in records}, {100, 200})

    async def test_configured_owners_can_be_removed_from_usage_audit(self) -> None:
        for user_id in (100, 200):
            await self.storage.record_bot_user_activity(
                user_id=user_id,
                username=None,
                display_name=str(user_id),
                chat_id=user_id,
                chat_type="private",
                chat_title=None,
                ai_provider=None,
            )

        removed = await self.storage.delete_bot_users(frozenset({100, 300}))

        self.assertEqual(removed, 1)
        records = await self.storage.list_bot_users()
        self.assertEqual([record.user_id for record in records], [200])

    def test_command_middleware_leaves_ask_for_processor(self) -> None:
        self.assertTrue(_is_non_ai_command(make_message(text="/start")))
        self.assertTrue(_is_non_ai_command(make_message(text="/stats@sample_bot")))
        self.assertFalse(_is_non_ai_command(make_message(text="/ask вопрос")))
        self.assertFalse(_is_non_ai_command(make_message(text="обычный текст")))


if __name__ == "__main__":
    unittest.main()
