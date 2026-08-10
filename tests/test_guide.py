from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from aiogram.types import FSInputFile

from bot.app import _set_commands
from bot.guide import (
    GUIDE_CALLBACK_DATA,
    GUIDE_VIDEO_PATH,
    GuideVideoSender,
    guide_keyboard,
)


class FakeMessage:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def answer_video(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(video=SimpleNamespace(file_id="telegram-guide-file-id"))


class FakeBot:
    def __init__(self) -> None:
        self.command_sets: list[list[Any]] = []

    async def set_my_commands(self, commands: list[Any], **_: Any) -> None:
        self.command_sets.append(commands)


@pytest.mark.asyncio
async def test_guide_upload_is_replaced_with_cached_telegram_file_id(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "guide.mp4"
    video_path.write_bytes(b"guide-video")
    sender = GuideVideoSender(video_path)
    message = FakeMessage()

    await sender.send(message)  # type: ignore[arg-type]
    await sender.send(message)  # type: ignore[arg-type]

    assert isinstance(message.calls[0]["video"], FSInputFile)
    assert message.calls[1]["video"] == "telegram-guide-file-id"
    assert message.calls[0]["supports_streaming"] is True


def test_guide_button_and_bundled_video_are_available() -> None:
    button = guide_keyboard().inline_keyboard[0][0]

    assert button.callback_data == GUIDE_CALLBACK_DATA
    assert "Гайд" in button.text
    assert GUIDE_VIDEO_PATH.is_file()
    assert GUIDE_VIDEO_PATH.stat().st_size > 0


@pytest.mark.asyncio
async def test_guide_is_present_in_telegram_command_menus() -> None:
    bot = FakeBot()

    await _set_commands(bot)  # type: ignore[arg-type]

    assert len(bot.command_sets) == 2
    assert all("guide" in {item.command for item in commands} for commands in bot.command_sets)
