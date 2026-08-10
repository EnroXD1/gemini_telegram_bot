from __future__ import annotations

import asyncio
from pathlib import Path

from aiogram.types import (
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

GUIDE_CALLBACK_DATA = "guide:hidden-media"
GUIDE_VIDEO_PATH = Path(__file__).with_name("assets") / "hidden_media_guide.mp4"
GUIDE_CAPTION = (
    "🎬 Как сохранить исчезающее фото или видео\n\n"
    "Ответьте на него любым сообщением до открытия. Обычные медиа бот "
    "не дублирует."
)


def guide_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🎬 Гайд по скрытым медиа",
                    callback_data=GUIDE_CALLBACK_DATA,
                )
            ]
        ]
    )


class GuideVideoSender:
    """Upload the bundled guide once, then reuse Telegram's cached file ID."""

    def __init__(self, path: Path = GUIDE_VIDEO_PATH) -> None:
        self.path = path
        self._file_id: str | None = None
        self._lock = asyncio.Lock()

    async def send(self, message: Message) -> Message:
        async with self._lock:
            if self._file_id is None and not self.path.is_file():
                raise FileNotFoundError(self.path)
            video: str | FSInputFile
            if self._file_id is not None:
                video = self._file_id
            else:
                video = FSInputFile(self.path, filename="Guide.mp4")
            sent = await message.answer_video(
                video=video,
                caption=GUIDE_CAPTION,
                supports_streaming=True,
            )
            if self._file_id is None and sent.video is not None:
                self._file_id = sent.video.file_id
            return sent
