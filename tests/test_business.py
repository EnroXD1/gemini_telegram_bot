import io
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aiogram.enums import ChatType
from aiogram.types import (
    BusinessMessagesDeleted,
    Chat,
    Message,
    PhotoSize,
    User,
    VideoNote,
)

from bot.business import BusinessMonitor
from bot.models import ConversationMessage
from bot.processor import MessageProcessor, _is_outgoing_business_message
from bot.storage import BusinessMediaArchiveRecord, BusinessMessageRecord, Storage


class FakeBot:
    def __init__(self) -> None:
        self.sent_messages: list[dict[str, Any]] = []
        self.sent_photos: list[dict[str, Any]] = []
        self.sent_video_notes: list[dict[str, Any]] = []
        self.downloaded_file_id: str | None = None

    async def download(self, file_id: str, destination: io.BytesIO) -> None:
        self.downloaded_file_id = file_id
        destination.write(b"saved-photo")

    async def send_message(self, **kwargs: Any) -> None:
        self.sent_messages.append(kwargs)

    async def send_photo(self, **kwargs: Any) -> None:
        self.sent_photos.append(kwargs)

    async def send_video_note(self, **kwargs: Any) -> None:
        self.sent_video_notes.append(kwargs)


def make_settings(database_path: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        database_path=database_path or Path("bot.sqlite3"),
        business_monitor_enabled=True,
        business_auto_reply_enabled=True,
        business_archive_media=True,
        business_archive_max_bytes=1024,
        media_download_timeout_seconds=1.0,
        business_welcome_enabled=True,
        business_welcome_text="Автоответчик подключён. /help — помощь",
    )


def make_message(
    *,
    message_id: int = 10,
    text: str | None = None,
    photo: list[PhotoSize] | None = None,
    video_note: VideoNote | None = None,
    sender_id: int = 200,
) -> Message:
    return Message(
        message_id=message_id,
        date=datetime.now(UTC),
        chat=Chat(id=200, type=ChatType.PRIVATE, first_name="Собеседник"),
        from_user=User(
            id=sender_id,
            is_bot=False,
            first_name="Иван",
            username="ivan",
        ),
        business_connection_id="connection-1",
        text=text,
        photo=photo,
        video_note=video_note,
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

    async def test_business_media_archive_round_trip(self) -> None:
        archive = BusinessMediaArchiveRecord(
            connection_id="connection-1",
            chat_id=200,
            message_id=10,
            kind="photo",
            file_name="photo.jpg",
            file_path="archive/10.jpg",
            file_size=42,
            created_at=123,
        )

        self.assertIsNone(await self.storage.upsert_business_media_archive(archive))
        stored = await self.storage.get_business_media_archive(
            connection_id="connection-1", chat_id=200, message_id=10
        )

        self.assertEqual(stored, archive)

        await self.storage.delete_business_media_archive(
            connection_id="connection-1", chat_id=200, message_id=10
        )
        self.assertIsNone(
            await self.storage.get_business_media_archive(
                connection_id="connection-1", chat_id=200, message_id=10
            )
        )

    async def test_local_history_is_bounded_and_reset(self) -> None:
        for index in range(5):
            await self.storage.append_conversation_exchange(
                scope_key="chat:200",
                user_content=f"вопрос {index}",
                assistant_content=f"ответ {index}",
                max_exchanges=3,
            )

        history = await self.storage.get_conversation_history(
            scope_key="chat:200", max_exchanges=3, max_chars=10_000
        )

        self.assertEqual(len(history), 6)
        self.assertEqual(
            history[0], ConversationMessage(role="user", content="вопрос 2")
        )
        self.assertEqual(history[-1].content, "ответ 4")
        self.assertTrue(await self.storage.has_conversation_history("chat:200"))

        await self.storage.reset_conversation("chat:200")

        self.assertFalse(await self.storage.has_conversation_history("chat:200"))

    async def test_outgoing_business_message_is_not_treated_as_user_request(self) -> None:
        await self.storage.save_business_connection(
            connection_id="connection-1",
            owner_user_id=100,
            owner_chat_id=100,
            is_enabled=True,
        )

        incoming = make_message(text="Вопрос", sender_id=200)
        outgoing = make_message(text="Ответ владельца", sender_id=100)

        self.assertFalse(
            await _is_outgoing_business_message(self.storage, incoming)
        )
        self.assertTrue(
            await _is_outgoing_business_message(self.storage, outgoing)
        )

    async def test_business_auto_reply_setting_is_persisted_per_owner(self) -> None:
        self.assertTrue(
            await self.storage.get_business_auto_reply_enabled(100, True)
        )

        await self.storage.set_business_auto_reply_enabled(100, False)

        self.assertFalse(
            await self.storage.get_business_auto_reply_enabled(100, True)
        )
        self.assertTrue(
            await self.storage.get_business_auto_reply_enabled(101, True)
        )

    async def test_processor_respects_monitoring_only_mode(self) -> None:
        await self.storage.save_business_connection(
            connection_id="connection-1",
            owner_user_id=100,
            owner_chat_id=100,
            is_enabled=True,
        )
        await self.storage.set_business_auto_reply_enabled(100, False)
        processor = object.__new__(MessageProcessor)
        processor.storage = self.storage
        processor.settings = SimpleNamespace(business_auto_reply_enabled=True)

        self.assertFalse(await processor.can_respond(make_message(text="Вопрос")))

    async def test_selected_business_chat_can_use_monitoring_only_mode(self) -> None:
        await self.storage.save_business_connection(
            connection_id="connection-1",
            owner_user_id=100,
            owner_chat_id=100,
            is_enabled=True,
        )
        processor = object.__new__(MessageProcessor)
        processor.storage = self.storage
        processor.settings = SimpleNamespace(business_auto_reply_enabled=True)

        self.assertTrue(await processor.can_respond(make_message(text="Вопрос")))

        await self.storage.set_business_chat_auto_reply_enabled(100, 200, False)

        self.assertFalse(await processor.can_respond(make_message(text="Вопрос")))
        self.assertTrue(
            await self.storage.get_effective_business_auto_reply_enabled(
                owner_user_id=100,
                chat_id=201,
                global_default=True,
            )
        )

    async def test_business_chat_list_contains_latest_contact_and_mode(self) -> None:
        await self.storage.save_business_connection(
            connection_id="connection-1",
            owner_user_id=100,
            owner_chat_id=100,
            is_enabled=True,
        )
        await self.storage.upsert_business_message(
            BusinessMessageRecord(
                connection_id="connection-1",
                chat_id=200,
                message_id=10,
                sender_user_id=200,
                sender_name="Иван",
                is_incoming=True,
                content="Привет",
            )
        )
        await self.storage.set_business_chat_auto_reply_enabled(100, 200, False)

        chats = await self.storage.list_business_chats(100)

        self.assertEqual(len(chats), 1)
        self.assertEqual(chats[0].sender_name, "Иван")
        self.assertFalse(chats[0].auto_reply_enabled)


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
            bot=self.bot,
            settings=make_settings(Path(self.temp_dir.name) / "bot.sqlite3"),
            storage=self.storage,
        )

    async def asyncTearDown(self) -> None:
        await self.storage.close()
        self.temp_dir.cleanup()

    async def test_photo_is_silent_until_deleted_then_uploaded_once(self) -> None:
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
        self.assertEqual(self.bot.sent_messages, [])
        self.assertEqual(self.bot.sent_photos, [])
        archive = await self.storage.get_business_media_archive(
            connection_id="connection-1", chat_id=200, message_id=10
        )
        self.assertIsNotNone(archive)
        assert archive is not None
        archive_path = Path(self.temp_dir.name) / "business_media" / archive.file_path
        self.assertTrue(archive_path.exists())

        event = BusinessMessagesDeleted(
            business_connection_id="connection-1",
            chat=Chat(id=200, type=ChatType.PRIVATE, first_name="Собеседник"),
            message_ids=[10],
        )
        await self.monitor.handle_deleted_messages(event)
        await self.monitor.handle_deleted_messages(event)

        self.assertEqual(len(self.bot.sent_photos), 1)
        self.assertEqual(self.bot.sent_photos[0]["chat_id"], 100)
        self.assertIn("Иван", self.bot.sent_photos[0]["caption"])
        self.assertIn("удалил", self.bot.sent_photos[0]["caption"])
        self.assertFalse(archive_path.exists())
        self.assertIsNone(
            await self.storage.get_business_media_archive(
                connection_id="connection-1", chat_id=200, message_id=10
            )
        )

    async def test_video_note_is_forwarded_only_after_deletion(self) -> None:
        message = make_message(
            video_note=VideoNote(
                file_id="video-note-id",
                file_unique_id="video-note-unique",
                length=240,
                duration=15,
                file_size=100,
            )
        )

        await self.monitor.capture_message(message)

        self.assertEqual(self.bot.sent_messages, [])
        self.assertEqual(self.bot.sent_video_notes, [])

        await self.monitor.handle_deleted_messages(
            BusinessMessagesDeleted(
                business_connection_id="connection-1",
                chat=Chat(id=200, type=ChatType.PRIVATE, first_name="Собеседник"),
                message_ids=[10],
            )
        )

        self.assertEqual(len(self.bot.sent_messages), 1)
        self.assertIn("удалил", self.bot.sent_messages[0]["text"])
        self.assertEqual(len(self.bot.sent_video_notes), 1)
        self.assertEqual(self.bot.sent_video_notes[0]["chat_id"], 100)

    async def test_ten_video_notes_create_no_immediate_duplicate_messages(self) -> None:
        message_ids = list(range(10, 20))
        for message_id in message_ids:
            await self.monitor.capture_message(
                make_message(
                    message_id=message_id,
                    video_note=VideoNote(
                        file_id=f"video-note-{message_id}",
                        file_unique_id=f"video-note-unique-{message_id}",
                        length=240,
                        duration=15,
                        file_size=100,
                    ),
                )
            )

        archives = await self.storage.get_business_media_archives(
            connection_id="connection-1",
            chat_id=200,
            message_ids=message_ids,
        )

        self.assertEqual(len(archives), 10)
        self.assertEqual(self.bot.sent_messages, [])
        self.assertEqual(self.bot.sent_video_notes, [])

    async def test_expired_silent_archive_is_removed_from_disk(self) -> None:
        await self.monitor.capture_message(
            make_message(
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
        )
        archive = await self.storage.get_business_media_archive(
            connection_id="connection-1", chat_id=200, message_id=10
        )
        assert archive is not None
        archive_path = Path(self.temp_dir.name) / "business_media" / archive.file_path

        removed = await self.monitor.prune_archived_media(-1)

        self.assertEqual(removed, 1)
        self.assertFalse(archive_path.exists())

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

    async def test_business_welcome_is_sent_once_to_incoming_contact(self) -> None:
        message = make_message(text="Здравствуйте")

        first = await self.monitor.welcome_contact(message)
        second = await self.monitor.welcome_contact(message)
        outgoing = await self.monitor.welcome_contact(
            make_message(message_id=11, text="Ответ владельца", sender_id=100)
        )

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertFalse(outgoing)
        self.assertEqual(len(self.bot.sent_messages), 1)
        self.assertEqual(
            self.bot.sent_messages[0]["business_connection_id"], "connection-1"
        )
        self.assertIn("Автоответчик", self.bot.sent_messages[0]["text"])

    async def test_monitoring_only_disables_welcome_but_keeps_delete_alerts(self) -> None:
        await self.storage.set_business_auto_reply_enabled(100, False)
        message = make_message(text="Секрет")

        await self.monitor.capture_message(message)
        welcomed = await self.monitor.welcome_contact(message)
        event = BusinessMessagesDeleted(
            business_connection_id="connection-1",
            chat=Chat(id=200, type=ChatType.PRIVATE, first_name="Собеседник"),
            message_ids=[10],
        )
        await self.monitor.handle_deleted_messages(event)

        self.assertFalse(welcomed)
        self.assertEqual(len(self.bot.sent_messages), 1)
        self.assertIn("удалил", self.bot.sent_messages[0]["text"])

    async def test_selected_chat_monitoring_only_disables_welcome(self) -> None:
        await self.storage.set_business_chat_auto_reply_enabled(100, 200, False)

        welcomed = await self.monitor.welcome_contact(
            make_message(text="Здравствуйте")
        )

        self.assertFalse(welcomed)
        self.assertEqual(self.bot.sent_messages, [])


if __name__ == "__main__":
    unittest.main()
