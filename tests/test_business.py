import io
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aiogram.enums import ChatType
from aiogram.types import BusinessMessagesDeleted, Chat, Message, PhotoSize, User

from bot.business import BusinessMonitor
from bot.storage import BusinessMessageRecord, Storage


class FakeBot:
    def __init__(self) -> None:
        self.sent_messages: list[dict[str, Any]] = []
        self.sent_photos: list[dict[str, Any]] = []
        self.downloaded_file_id: str | None = None

    async def download(self, file_id: str, destination: io.BytesIO) -> None:
        self.downloaded_file_id = file_id
        destination.write(b"saved-photo")

    async def send_message(self, **kwargs: Any) -> None:
        self.sent_messages.append(kwargs)

    async def send_photo(self, **kwargs: Any) -> None:
        self.sent_photos.append(kwargs)


def make_settings() -> SimpleNamespace:
    return SimpleNamespace(
        business_monitor_enabled=True,
        business_archive_media=True,
        business_archive_max_bytes=1024,
        media_download_timeout_seconds=1.0,
    )


def make_message(
    *,
    message_id: int = 10,
    text: str | None = None,
    photo: list[PhotoSize] | None = None,
) -> Message:
    return Message(
        message_id=message_id,
        date=datetime.now(UTC),
        chat=Chat(id=200, type=ChatType.PRIVATE, first_name="Собеседник"),
        from_user=User(
            id=200,
            is_bot=False,
            first_name="Иван",
            username="ivan",
        ),
        business_connection_id="connection-1",
        text=text,
        photo=photo,
    )


class BusinessStorageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self.temp_dir.name) / "bot.sqlite3")
        await self.storage.open()

    async def asyncTearDown(self) -> None:
        await self.storage.close()
        self.temp_dir.cleanup()

    async def test_upsert_returns_previous_message_version(self) -> None:
        original = BusinessMessageRecord(
            connection_id="connection-1",
            chat_id=200,
            message_id=10,
            sender_user_id=200,
            sender_name="Иван",
            is_incoming=True,
            content="Старый текст",
        )
        changed = BusinessMessageRecord(
            connection_id="connection-1",
            chat_id=200,
            message_id=10,
            sender_user_id=200,
            sender_name="Иван",
            is_incoming=True,
            content="Новый текст",
        )

        self.assertIsNone(await self.storage.upsert_business_message(original))
        previous = await self.storage.upsert_business_message(changed)

        self.assertIsNotNone(previous)
        assert previous is not None
        self.assertEqual(previous.content, "Старый текст")


class BusinessMonitorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self.temp_dir.name) / "bot.sqlite3")
        await self.storage.open()
        await self.storage.save_business_connection(
            connection_id="connection-1",
            owner_user_id=100,
            owner_chat_id=100,
            is_enabled=True,
        )
        self.bot = FakeBot()
        self.monitor = BusinessMonitor(
            bot=self.bot, settings=make_settings(), storage=self.storage
        )

    async def asyncTearDown(self) -> None:
        await self.storage.close()
        self.temp_dir.cleanup()

    async def test_photo_is_downloaded_and_uploaded_to_owner(self) -> None:
        message = make_message(
            photo=[
                PhotoSize(
                    file_id="photo-id",
                    file_unique_id="photo-unique",
                    width=100,
                    height=100,
                    file_size=100,
                )
            ]
        )

        await self.monitor.capture_message(message)

        self.assertEqual(self.bot.downloaded_file_id, "photo-id")
        self.assertEqual(len(self.bot.sent_photos), 1)
        self.assertEqual(self.bot.sent_photos[0]["chat_id"], 100)
        self.assertIn("Иван", self.bot.sent_photos[0]["caption"])

    async def test_edit_and_delete_notifications_include_original_text(self) -> None:
        await self.monitor.capture_message(make_message(text="Старый текст"))
        await self.monitor.handle_edited_message(make_message(text="Новый текст"))

        edit_text = self.bot.sent_messages[-1]["text"]
        self.assertIn("изменил", edit_text)
        self.assertIn("Старый текст", edit_text)
        self.assertIn("Новый текст", edit_text)

        event = BusinessMessagesDeleted(
            business_connection_id="connection-1",
            chat=Chat(id=200, type=ChatType.PRIVATE, first_name="Собеседник"),
            message_ids=[10],
        )
        await self.monitor.handle_deleted_messages(event)

        delete_text = self.bot.sent_messages[-1]["text"]
        self.assertIn("удалил", delete_text)
        self.assertIn("Новый текст", delete_text)


if __name__ == "__main__":
    unittest.main()
