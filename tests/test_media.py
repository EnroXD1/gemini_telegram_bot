import io
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace

from aiogram.enums import ChatType
from aiogram.types import Chat, Message, PhotoSize, User

from bot.media import MediaExtractor, classify_mime


class FakeBot:
    async def download(self, file_id: str, destination: io.BytesIO) -> None:
        self.file_id = file_id
        destination.write(b"jpeg-data")


def make_settings() -> SimpleNamespace:
    return SimpleNamespace(
        max_media_items=10,
        max_media_bytes=1024,
        max_text_file_chars=1000,
        media_download_timeout_seconds=1.0,
    )


class MimeTests(unittest.TestCase):
    def test_supported_mime_types(self) -> None:
        self.assertEqual(classify_mime("image/jpeg"), "image")
        self.assertEqual(classify_mime("audio/ogg"), "audio")
        self.assertEqual(classify_mime("video/mp4"), "video")
        self.assertEqual(classify_mime("application/pdf"), "document")
        self.assertEqual(classify_mime("text/plain; charset=utf-8"), "text")
        self.assertIsNone(classify_mime("application/zip"))


class MediaExtractorTests(unittest.IsolatedAsyncioTestCase):
    async def test_photo_is_downloaded_and_labeled(self) -> None:
        message = Message(
            message_id=5,
            date=datetime.now(UTC),
            chat=Chat(id=10, type=ChatType.PRIVATE),
            from_user=User(id=10, is_bot=False, first_name="Анна"),
            caption="Что на фото?",
            photo=[
                PhotoSize(
                    file_id="photo-id",
                    file_unique_id="unique",
                    width=100,
                    height=100,
                    file_size=100,
                )
            ],
        )
        bot = FakeBot()

        bundle = await MediaExtractor(make_settings()).prepare(
            bot=bot, messages=[message], user_text=message.caption or ""
        )

        self.assertEqual(bot.file_id, "photo-id")
        self.assertEqual(len(bundle.media), 1)
        self.assertEqual(bundle.media[0].kind, "image")
        self.assertEqual(bundle.media[0].data, b"jpeg-data")
        self.assertIn("Анна", bundle.prompt)
        self.assertIn("Что на фото?", bundle.prompt)


if __name__ == "__main__":
    unittest.main()
