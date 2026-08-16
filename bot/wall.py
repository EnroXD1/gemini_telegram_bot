from __future__ import annotations

import asyncio
import gc
import logging
import mimetypes
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from aiogram.filters import Filter
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaDocument,
    Message,
)
from PIL import (
    Image,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageOps,
    UnidentifiedImageError,
)

logger = logging.getLogger(__name__)

WALL_COLUMNS = 3
WALL_COVER_ASPECT = 4 / 5
WALL_STORY_ASPECT = 9 / 16
WALL_PART_COUNTS = tuple(range(3, 25, 3))
WALL_CALLBACK_PREFIX = "wall:"
WALL_CANCEL_CALLBACK = f"{WALL_CALLBACK_PREFIX}cancel"
WALL_START_CALLBACK = f"{WALL_CALLBACK_PREFIX}start"


class WallError(RuntimeError):
    """A wall could not be created from the supplied image."""


class WallBusyError(WallError):
    """The same user already has a wall request in progress."""


@dataclass(frozen=True, slots=True)
class WallSource:
    file_id: str
    file_size: int | None
    file_name: str


@dataclass(frozen=True, slots=True)
class WallRenderResult:
    preview_path: Path
    piece_paths: tuple[Path, ...]
    rows: int
    source_width: int
    source_height: int
    tile_width: int
    tile_height: int


@dataclass(slots=True)
class _PendingWall:
    expires_at: float
    source: WallSource | None = None
    parts: int | None = None


class WallSessions:
    """Small, expiring in-memory state for the two-step wall dialogue."""

    def __init__(
        self,
        ttl_seconds: float = 10 * 60,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._pending: dict[tuple[int, int], _PendingWall] = {}

    def remember_source(self, chat_id: int, user_id: int, source: WallSource) -> None:
        self._prune()
        self._pending[(chat_id, user_id)] = _PendingWall(
            expires_at=self._clock() + self._ttl_seconds,
            source=source,
        )

    def select_parts(
        self, chat_id: int, user_id: int, parts: int
    ) -> WallSource | None:
        validate_wall_parts(parts)
        self._prune()
        key = (chat_id, user_id)
        current = self._pending.get(key)
        if current is not None and current.source is not None:
            self._pending.pop(key, None)
            return current.source
        self._pending[key] = _PendingWall(
            expires_at=self._clock() + self._ttl_seconds,
            parts=parts,
        )
        return None

    def consume_photo(
        self, chat_id: int, user_id: int, source: WallSource
    ) -> tuple[int, WallSource] | None:
        self._prune()
        current = self._pending.get((chat_id, user_id))
        if current is None or current.parts is None:
            return None
        self._pending.pop((chat_id, user_id), None)
        return current.parts, source

    def cancel(self, chat_id: int, user_id: int) -> bool:
        self._prune()
        return self._pending.pop((chat_id, user_id), None) is not None

    def _prune(self) -> None:
        now = self._clock()
        expired = [key for key, item in self._pending.items() if item.expires_at <= now]
        for key in expired:
            self._pending.pop(key, None)


class PendingWallPhotoFilter(Filter):
    def __init__(self, sessions: WallSessions) -> None:
        self._sessions = sessions

    async def __call__(self, message: Message) -> bool | dict[str, object]:
        user = message.from_user
        if message.chat.type != "private" or user is None:
            return False
        source = wall_source_from_message(message, include_reply=False)
        if source is None:
            return False
        pending = self._sessions.consume_photo(message.chat.id, user.id, source)
        if pending is None:
            return False
        parts, selected_source = pending
        return {"wall_parts": parts, "wall_source": selected_source}


class WallService:
    def __init__(
        self,
        *,
        max_source_bytes: int,
        max_pixels: int,
        tile_size: int,
        download_timeout_seconds: float,
    ) -> None:
        self.max_source_bytes = max_source_bytes
        self.max_pixels = max_pixels
        self.tile_size = tile_size
        self.download_timeout_seconds = download_timeout_seconds
        self.sessions = WallSessions()
        self._render_semaphore = asyncio.Semaphore(1)
        self._active_users: set[int] = set()

    async def create_and_send(
        self,
        *,
        anchor: Message,
        source: WallSource,
        parts: int,
        requester_id: int,
    ) -> None:
        validate_wall_parts(parts)
        if source.file_size and source.file_size > self.max_source_bytes:
            raise WallError(
                "Файл слишком большой. Для стенки отправьте изображение не тяжелее "
                f"{_human_size(self.max_source_bytes)}."
            )
        if requester_id in self._active_users:
            raise WallBusyError(
                "Ваша предыдущая стенка ещё обрабатывается. Дождитесь результата."
            )
        if self._render_semaphore.locked():
            raise WallBusyError(
                "Сейчас обрабатывается другая стенка. Попробуйте через несколько секунд."
            )

        self._active_users.add(requester_id)
        status: Message | None = None
        try:
            async with self._render_semaphore:
                status = await anchor.answer(
                    f"✂️ Готовлю стенку 3×{parts // WALL_COLUMNS}. "
                    "Это может занять немного времени…"
                )
                with tempfile.TemporaryDirectory(prefix="telegram-wall-") as temp_name:
                    temp_dir = Path(temp_name)
                    source_path = temp_dir / "source-image"
                    try:
                        async with asyncio.timeout(self.download_timeout_seconds):
                            await anchor.bot.download(source.file_id, destination=source_path)
                    except TimeoutError as exc:
                        raise WallError(
                            "Telegram слишком долго отдавал фотографию. Попробуйте ещё раз."
                        ) from exc
                    except Exception as exc:
                        logger.warning("Could not download wall source: %s", type(exc).__name__)
                        raise WallError(
                            "Не удалось скачать фотографию из Telegram. "
                            "Попробуйте отправить её заново."
                        ) from exc

                    actual_size = source_path.stat().st_size
                    if actual_size > self.max_source_bytes:
                        raise WallError(
                            "Файл слишком большой. Для стенки отправьте изображение не тяжелее "
                            f"{_human_size(self.max_source_bytes)}."
                        )

                    result = await asyncio.to_thread(
                        render_wall,
                        source_path,
                        temp_dir,
                        parts,
                        tile_size=self.tile_size,
                        max_pixels=self.max_pixels,
                    )
                    await self._send_result(anchor, result)
        finally:
            self._active_users.discard(requester_id)
            if status is not None:
                try:
                    await status.delete()
                except Exception:
                    logger.debug("Could not delete wall progress message", exc_info=True)
            gc.collect()

    async def _send_result(self, anchor: Message, result: WallRenderResult) -> None:
        count = len(result.piece_paths)
        await anchor.answer_photo(
            FSInputFile(result.preview_path, filename="wall-preview.jpg"),
            caption=(
                f"✅ Стенка 3×{result.rows} готова: {count} частей.\n\n"
                "Ниже идут файлы в порядке публикации: начните с 01 и загружайте "
                f"по очереди до {count:02d}. Каждый файл подготовлен как Story "
                f"{result.tile_width}×{result.tile_height}, а центральная область "
                "образует бесшовную стенку в профиле."
            ),
        )

        for group_number, paths in enumerate(media_group_batches(result.piece_paths), start=1):
            media: list[InputMediaDocument] = []
            for index, path in enumerate(paths):
                caption = None
                if group_number == 1 and index == 0:
                    caption = "01 — загрузить первой. Файлы уже расположены по порядку."
                media.append(
                    InputMediaDocument(
                        media=FSInputFile(path, filename=path.name),
                        caption=caption,
                    )
                )
            await anchor.answer_media_group(media=media)


def validate_wall_parts(parts: int) -> None:
    if parts not in WALL_PART_COUNTS:
        allowed = ", ".join(str(value) for value in WALL_PART_COUNTS)
        raise WallError(f"Неизвестный размер стенки. Доступны: {allowed}.")


def wall_keyboard() -> InlineKeyboardMarkup:
    values = list(WALL_PART_COUNTS)
    rows = [
        [
            InlineKeyboardButton(
                text=f"{parts} ({WALL_COLUMNS}×{parts // WALL_COLUMNS})",
                callback_data=f"{WALL_CALLBACK_PREFIX}{parts}",
            )
            for parts in values[index : index + 2]
        ]
        for index in range(0, len(values), 2)
    ]
    rows.append(
        [InlineKeyboardButton(text="Отмена", callback_data=WALL_CANCEL_CALLBACK)]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def wall_source_from_message(
    message: Message, *, include_reply: bool = True
) -> WallSource | None:
    candidates = [message]
    if include_reply and message.reply_to_message is not None:
        candidates.append(message.reply_to_message)
    for candidate in candidates:
        if candidate.photo:
            photo = candidate.photo[-1]
            return WallSource(
                file_id=photo.file_id,
                file_size=photo.file_size,
                file_name="photo.jpg",
            )
        document = candidate.document
        if document is None:
            continue
        file_name = document.file_name or "image"
        mime_type = document.mime_type or mimetypes.guess_type(file_name)[0] or ""
        if mime_type.lower().startswith("image/"):
            return WallSource(
                file_id=document.file_id,
                file_size=document.file_size,
                file_name=file_name,
            )
    return None


def media_group_batches(paths: tuple[Path, ...]) -> tuple[tuple[Path, ...], ...]:
    """Split files into Telegram-valid 2..10 item media groups."""
    batches: list[tuple[Path, ...]] = []
    offset = 0
    while offset < len(paths):
        remaining = len(paths) - offset
        size = min(10, remaining)
        if remaining == 11:
            size = 9
        batch = paths[offset : offset + size]
        if len(batch) == 1:
            raise WallError("Не удалось безопасно сгруппировать части стенки.")
        batches.append(batch)
        offset += size
    return tuple(batches)


def render_wall(
    source_path: Path,
    output_dir: Path,
    parts: int,
    *,
    tile_size: int = 1080,
    max_pixels: int = 12_000_000,
) -> WallRenderResult:
    validate_wall_parts(parts)
    if tile_size < 128 or tile_size > 1440:
        raise WallError("Размер одного фрагмента должен быть от 128 до 1440 пикселей.")
    rows = parts // WALL_COLUMNS

    try:
        with Image.open(source_path) as opened:
            if getattr(opened, "n_frames", 1) != 1:
                raise WallError(
                    "Для стенки нужна статичная фотография, а не GIF или анимация."
                )
            width, height = opened.size
            if width < WALL_COLUMNS or height < rows:
                raise WallError("Разрешение фотографии слишком маленькое для этой сетки.")
            if width * height > max_pixels:
                raise WallError(
                    "Слишком большое разрешение фотографии. Отправьте изображение не более "
                    f"{max_pixels // 1_000_000} мегапикселей."
                )
            if opened.mode != "RGB" and width * height > min(max_pixels, 8_000_000):
                raise WallError(
                    "PNG/WEBP с прозрачностью или нестандартным цветовым режимом "
                    "должен быть не больше 8 мегапикселей."
                )

            # Pillow's default exif_transpose() returns a complete image copy even
            # when orientation is already correct. On Bothost that copy can exceed
            # the container's real memory cgroup, so mutate the decoded image core.
            ImageOps.exif_transpose(opened, in_place=True)
            opened.load()
            image = opened if opened.mode == "RGB" else _flatten_to_rgb(opened)
            try:
                return _render_loaded_wall(
                    image,
                    output_dir,
                    parts=parts,
                    rows=rows,
                    tile_size=tile_size,
                )
            finally:
                if image is not opened:
                    image.close()
    except WallError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise WallError(
            "Telegram прислал повреждённое или неподдерживаемое изображение. "
            "Попробуйте JPEG, PNG или WEBP."
        ) from exc



def _render_loaded_wall(
    image: Image.Image,
    output_dir: Path,
    *,
    parts: int,
    rows: int,
    tile_size: int,
) -> WallRenderResult:
    source_width, source_height = image.size
    crop_box = _smart_crop_box(
        image,
        columns=WALL_COLUMNS,
        rows=rows,
        tile_aspect=WALL_COVER_ASPECT,
    )
    preview_path = output_dir / "wall-preview.jpg"
    _save_preview(image, crop_box, preview_path, rows=rows)

    left, top, right, bottom = crop_box
    crop_width = right - left
    crop_height = bottom - top
    cells = [
        (row, column)
        for row in range(rows)
        for column in range(WALL_COLUMNS)
    ]
    piece_paths: list[Path] = []
    for upload_index, (row, column) in enumerate(reversed(cells), start=1):
        tile_box = (
            left + round(column * crop_width / WALL_COLUMNS),
            top + round(row * crop_height / rows),
            left + round((column + 1) * crop_width / WALL_COLUMNS),
            top + round((row + 1) * crop_height / rows),
        )
        cover_height = round(tile_size / WALL_COVER_ASPECT)
        cover = image.resize(
            (tile_size, cover_height),
            resample=Image.Resampling.LANCZOS,
            box=tile_box,
        )
        tile = _story_tile_from_cover(cover, tile_width=tile_size)
        cover.close()
        piece_path = output_dir / f"{upload_index:02d}_of_{parts:02d}.jpg"
        try:
            tile.save(
                piece_path,
                format="JPEG",
                quality=92,
                optimize=True,
                progressive=True,
            )
        finally:
            tile.close()
        piece_paths.append(piece_path)

    return WallRenderResult(
        preview_path=preview_path,
        piece_paths=tuple(piece_paths),
        rows=rows,
        source_width=source_width,
        source_height=source_height,
        tile_width=tile_size,
        tile_height=round(tile_size / WALL_STORY_ASPECT),
    )


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image if image.mode == "RGBA" else image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        try:
            background.paste(rgba, mask=rgba.getchannel("A"))
        finally:
            if rgba is not image:
                rgba.close()
        return background
    return image.convert("RGB")


def _smart_crop_box(
    image: Image.Image,
    *,
    columns: int,
    rows: int,
    tile_aspect: float = WALL_COVER_ASPECT,
) -> tuple[float, float, float, float]:
    width, height = image.size
    target_ratio = columns * tile_aspect / rows
    source_ratio = width / height
    attention_x, attention_y = _attention_point(image)

    if source_ratio > target_ratio:
        crop_height = float(height)
        crop_width = crop_height * target_ratio
        desired_left = attention_x - crop_width / 2
        left = min(max(0.0, desired_left), width - crop_width)
        return left, 0.0, left + crop_width, crop_height

    crop_width = float(width)
    crop_height = crop_width / target_ratio
    desired_top = attention_y - crop_height / 2
    top = min(max(0.0, desired_top), height - crop_height)
    return 0.0, top, crop_width, top + crop_height


def _attention_point(image: Image.Image) -> tuple[float, float]:
    """Estimate a stable detail-weighted focal point without heavyweight CV models."""
    width, height = image.size
    scale = min(1.0, 256 / max(width, height))
    thumb_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    thumb = image.resize(thumb_size, Image.Resampling.BILINEAR).convert("L")
    edges = thumb.filter(ImageFilter.FIND_EDGES)
    thumb.close()
    edge_width, edge_height = edges.size
    border = 2
    total = weighted_x = weighted_y = 0.0
    try:
        pixels = edges.load()
        for y in range(border, max(border, edge_height - border)):
            for x in range(border, max(border, edge_width - border)):
                weight = float(pixels[x, y])
                if weight < 12:
                    continue
                total += weight
                weighted_x += x * weight
                weighted_y += y * weight
    finally:
        edges.close()

    center_x = width / 2
    center_y = height / 2
    if total <= 0:
        return center_x, center_y
    detail_x = (weighted_x / total + 0.5) / edge_width * width
    detail_y = (weighted_y / total + 0.5) / edge_height * height
    # Keep the automatic crop calm: detail nudges the geometric center rather than
    # dragging the entire composition to one noisy edge.
    return (
        center_x * 0.65 + detail_x * 0.35,
        center_y * 0.65 + detail_y * 0.35,
    )


def _story_tile_from_cover(cover: Image.Image, *, tile_width: int) -> Image.Image:
    story_height = round(tile_width / WALL_STORY_ASPECT)
    background = ImageOps.fit(
        cover,
        (tile_width, story_height),
        method=Image.Resampling.BILINEAR,
        centering=(0.5, 0.5),
    )
    blurred = background.filter(
        ImageFilter.GaussianBlur(radius=max(4.0, tile_width / 45))
    )
    background.close()
    story = ImageEnhance.Brightness(blurred).enhance(0.52)
    blurred.close()
    top = (story_height - cover.height) // 2
    story.paste(cover, (0, top))
    return story


def _save_preview(
    image: Image.Image,
    crop_box: tuple[float, float, float, float],
    path: Path,
    *,
    rows: int,
) -> None:
    crop_width = crop_box[2] - crop_box[0]
    crop_height = crop_box[3] - crop_box[1]
    preview_width = 810
    preview_height = max(270, round(preview_width * crop_height / crop_width))
    preview = image.resize(
        (preview_width, preview_height),
        resample=Image.Resampling.LANCZOS,
        box=crop_box,
    )
    try:
        draw = ImageDraw.Draw(preview)
        line_width = max(2, preview_width // 270)
        for column in range(1, WALL_COLUMNS):
            x = round(column * preview_width / WALL_COLUMNS)
            draw.line((x, 0, x, preview_height), fill="white", width=line_width)
        for row in range(1, rows):
            y = round(row * preview_height / rows)
            draw.line((0, y, preview_width, y), fill="white", width=line_width)
        preview.save(path, format="JPEG", quality=88, optimize=True, progressive=True)
    finally:
        preview.close()


def callback_wall_parts(callback: CallbackQuery) -> int | None:
    data = callback.data or ""
    if not data.startswith(WALL_CALLBACK_PREFIX) or data == WALL_CANCEL_CALLBACK:
        return None
    try:
        parts = int(data.removeprefix(WALL_CALLBACK_PREFIX))
    except ValueError:
        return None
    return parts if parts in WALL_PART_COUNTS else None


def _human_size(value: int) -> str:
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.0f} МиБ"
    if value >= 1024:
        return f"{value / 1024:.0f} КиБ"
    return f"{value} байт"
