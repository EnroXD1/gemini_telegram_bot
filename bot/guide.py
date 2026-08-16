from __future__ import annotations

import asyncio
from pathlib import Path

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import (
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from .emoji_theme import EmojiTheme
from .wall import WALL_START_CALLBACK

GUIDE_CALLBACK_DATA = "guide:hidden-media"
GUIDE_VIDEO_PATH = Path(__file__).with_name("assets") / "hidden_media_guide.mp4"
GUIDE_CAPTION = (
    "Как сохранить исчезающее фото или видео\n\n"
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
            ],
            [
                InlineKeyboardButton(
                    text="🖼 Создать стенку из фото",
                    callback_data=WALL_START_CALLBACK,
                )
            ],
        ]
    )


class GuideVideoSender:
    """Upload the bundled guide once, then reuse Telegram's cached file ID."""

    def __init__(
        self,
        path: Path = GUIDE_VIDEO_PATH,
        emoji_theme: EmojiTheme | None = None,
    ) -> None:
        self.path = path
        self.emoji_theme = emoji_theme
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
            themed = None
            if self.emoji_theme is not None:
                themed = await self.emoji_theme.service_text(
                    f"🎬 {GUIDE_CAPTION}"
                )
            try:
                sent = await message.answer_video(
                    video=video,
                    caption=(themed.text if themed is not None else f"🎬 {GUIDE_CAPTION}"),
                    caption_entities=(themed.entities if themed is not None else None),
                    supports_streaming=True,
                )
            except TelegramBadRequest:
                if themed is None or themed.entities is None:
                    raise
                sent = await message.answer_video(
                    video=video,
                    caption=f"🎬 {GUIDE_CAPTION}",
                    supports_streaming=True,
                )
            if self._file_id is None and sent.video is not None:
                self._file_id = sent.video.file_id
            return sent
