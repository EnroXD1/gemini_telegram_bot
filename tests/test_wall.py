from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from aiogram.exceptions import TelegramRetryAfter
from aiogram.methods import SendDocument
from PIL import Image, ImageDraw

from bot.wall import (
    WALL_PART_COUNTS,
    WallError,
    WallService,
    WallSessions,
    WallSource,
    _attention_point,
    render_wall,
    wall_keyboard,
)


def _save_color_grid(path: Path) -> list[tuple[int, int, int]]:
    colors = [
        (240, 20, 20),
        (20, 240, 20),
        (20, 20, 240),
        (240, 220, 20),
        (220, 20, 240),
        (20, 220, 240),
    ]
    image = Image.new("RGB", (600, 400))
    draw = ImageDraw.Draw(image)
    for index, color in enumerate(colors):
        row, column = divmod(index, 3)
        draw.rectangle(
            (column * 200, row * 200, (column + 1) * 200 - 1, (row + 1) * 200 - 1),
            fill=color,
        )
    image.save(path, format="PNG")
    image.close()
    return colors


def test_render_wall_creates_square_tiles_in_publication_order(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    colors = _save_color_grid(source)

    result = render_wall(source, tmp_path, 6, tile_size=128, max_pixels=1_000_000)

    assert result.rows == 2
    assert result.source_width == 600
    assert result.source_height == 400
    assert [path.name for path in result.piece_paths] == [
        "01_of_06.jpg",
        "02_of_06.jpg",
        "03_of_06.jpg",
        "04_of_06.jpg",
        "05_of_06.jpg",
        "06_of_06.jpg",
    ]
    for path, expected in zip(result.piece_paths, reversed(colors), strict=True):
        with Image.open(path) as tile:
            assert tile.size == (128, 228)
            actual = tile.getpixel((64, 114))
            assert all(
                abs(value - target) < 12
                for value, target in zip(actual, expected, strict=True)
            )


def test_render_wall_creates_tall_preview_for_24_parts(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    image = Image.new("RGB", (600, 1600), "navy")
    image.save(source)
    image.close()

    result = render_wall(source, tmp_path, 24, tile_size=128, max_pixels=1_000_000)

    assert len(result.piece_paths) == 24
    with Image.open(result.preview_path) as preview:
        assert preview.size == (810, 2700)


def test_render_wall_rejects_animations_and_excessive_resolution(tmp_path: Path) -> None:
    animated = tmp_path / "animated.gif"
    first = Image.new("RGB", (90, 90), "red")
    second = Image.new("RGB", (90, 90), "blue")
    first.save(animated, save_all=True, append_images=[second], duration=100, loop=0)
    first.close()
    second.close()

    with pytest.raises(WallError, match="статичная"):
        render_wall(animated, tmp_path, 3, tile_size=128)

    large = tmp_path / "large.png"
    image = Image.new("RGB", (1001, 1000), "white")
    image.save(large)
    image.close()
    with pytest.raises(WallError, match="разрешение"):
        render_wall(large, tmp_path, 3, tile_size=128, max_pixels=1_000_000)


def test_transparent_image_is_flattened_to_white(tmp_path: Path) -> None:
    source = tmp_path / "alpha.png"
    image = Image.new("RGBA", (300, 100), (255, 0, 0, 0))
    image.save(source)
    image.close()

    result = render_wall(source, tmp_path, 3, tile_size=128)

    with Image.open(result.piece_paths[0]) as tile:
        red, green, blue = tile.getpixel((64, 64))
        assert red > 245 and green > 245 and blue > 245


def test_attention_point_moves_toward_detailed_area() -> None:
    image = Image.new("RGB", (900, 300), "black")
    draw = ImageDraw.Draw(image)
    for x in range(650, 880, 10):
        draw.line((x, 10, 880, 290), fill="white", width=3)

    attention_x, _ = _attention_point(image)
    image.close()

    assert attention_x > 450


def test_wall_keyboard_contains_every_size_and_cancel() -> None:
    keyboard = wall_keyboard()
    callback_data = [
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
    ]

    assert callback_data[:-1] == [f"wall:{parts}" for parts in WALL_PART_COUNTS]
    assert callback_data[-1] == "wall:cancel"


def test_wall_sessions_complete_both_dialog_orders_and_expire() -> None:
    now = [100.0]
    sessions = WallSessions(ttl_seconds=10, clock=lambda: now[0])
    source = WallSource("photo-id", 123, "photo.jpg")

    sessions.remember_source(1, 2, source)
    assert sessions.select_parts(1, 2, 9) == source

    assert sessions.select_parts(1, 2, 12) is None
    assert sessions.consume_photo(1, 2, source) == (12, source)
    assert sessions.consume_photo(1, 2, source) is None

    sessions.select_parts(1, 2, 6)
    now[0] = 111.0
    assert sessions.consume_photo(1, 2, source) is None


def test_invalid_wall_size_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    image = Image.new("RGB", (300, 300), "white")
    image.save(source)
    image.close()

    with pytest.raises(WallError, match="Доступны"):
        render_wall(source, tmp_path, 5, tile_size=128)


@pytest.mark.asyncio
async def test_wall_service_downloads_renders_sends_and_cleans_temp_files(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "telegram-source.png"
    _save_color_grid(source_path)
    observed_paths: list[Path] = []
    sent_names: list[str] = []

    class FakeBot:
        async def download(self, file_id: str, destination: Path) -> None:
            assert file_id == "telegram-photo"
            shutil.copyfile(source_path, destination)

    class FakeStatus:
        deleted = False

        async def delete(self) -> None:
            self.deleted = True

    class FakeAnchor:
        bot = FakeBot()
        status = FakeStatus()

        async def answer(self, text: str) -> FakeStatus:
            assert "3×2" in text
            return self.status

        async def answer_photo(self, media: Any, *, caption: str) -> None:
            path = Path(media.path)
            assert path.exists()
            assert "6 частей" in caption
            observed_paths.append(path)

        async def answer_document(
            self,
            *,
            document: Any,
            caption: str,
            disable_notification: bool,
        ) -> None:
            path = Path(document.path)
            assert path.exists()
            assert caption.startswith(f"Фрагмент {len(sent_names) + 1:02d}/06")
            assert disable_notification is True
            observed_paths.append(path)
            sent_names.append(document.filename)

    anchor = FakeAnchor()
    service = WallService(
        max_source_bytes=8 * 1024 * 1024,
        max_pixels=1_000_000,
        tile_size=128,
        download_timeout_seconds=5,
    )

    await service.create_and_send(
        anchor=anchor,  # type: ignore[arg-type]
        source=WallSource("telegram-photo", source_path.stat().st_size, "source.png"),
        parts=6,
        requester_id=42,
    )

    assert anchor.status.deleted is True
    assert sent_names == [f"{index:02d}_of_06.jpg" for index in range(1, 7)]
    assert observed_paths
    assert all(not path.exists() for path in observed_paths)


@pytest.mark.asyncio
async def test_wall_document_retries_same_file_after_flood_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "01_of_03.jpg"
    source.touch()
    attempts: list[str] = []
    sleeps: list[float] = []

    class FakeAnchor:
        async def answer_document(
            self,
            *,
            document: Any,
            caption: str,
            disable_notification: bool,
        ) -> None:
            attempts.append(document.filename)
            assert caption == "Фрагмент 01/03"
            assert disable_notification is True
            if len(attempts) == 1:
                raise TelegramRetryAfter(
                    method=SendDocument(chat_id=1, document="retry"),
                    message="Flood control exceeded",
                    retry_after=2,
                )

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("bot.wall.asyncio.sleep", fake_sleep)
    service = WallService(
        max_source_bytes=8 * 1024 * 1024,
        max_pixels=1_000_000,
        tile_size=128,
        download_timeout_seconds=5,
    )

    await service._send_document_with_retry(  # noqa: SLF001
        FakeAnchor(),  # type: ignore[arg-type]
        source,
        caption="Фрагмент 01/03",
    )

    assert attempts == ["01_of_03.jpg", "01_of_03.jpg"]
    assert sleeps == [2.1]
