from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Iterator

import httpx
from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from .text_utils import split_text

logger = logging.getLogger(__name__)

# Репозиторий GitHub.
GITHUB_URL = "https://github.com/EnroXD1/gemini_telegram_bot"

# Прямая ссылка на актуальный README из ветки main.
README_URL = (
    "https://raw.githubusercontent.com/"
    "EnroXD1/gemini_telegram_bot/main/README.md"
)

# Максимальный размер одной страницы Telegram.
PAGE_SIZE = 3000

# Кеш между нажатиями кнопок — 5 минут.
CACHE_SECONDS = 300

# Максимальный размер загружаемого README.
MAX_README_BYTES = 1_000_000

# Запас относительно лимита встроенного Markdown-просмотра Android: 64 КиБ.
MAX_DOCUMENT_BYTES = 60 * 1024


def _opening_fence(line: str) -> str | None:
    match = re.fullmatch(r" {0,3}(`{3,}|~{3,})([^\r\n]*)(?:\r?\n)?", line)
    if match is None:
        return None
    marker, info = match.groups()
    if marker.startswith("`") and "`" in info:
        return None
    return marker


def _closing_fence(line: str, marker: str) -> bool:
    return re.fullmatch(
        rf" {{0,3}}{re.escape(marker[0])}{{{len(marker)},}}[ \t]*(?:\r?\n)?", line
    ) is not None


def _markdown_blocks(text: str) -> Iterator[str]:
    """Keep paragraphs and fenced code blocks together where possible."""
    lines: list[str] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        if fence is not None:
            lines.append(line)
            if _closing_fence(line, fence):
                yield "".join(lines)
                lines = []
                fence = None
            continue

        marker = _opening_fence(line)
        if marker is not None:
            if lines:
                yield "".join(lines)
            lines = [line]
            fence = marker
            continue

        # Заголовок начинает новый раздел; пустая строка завершает абзац.
        if lines and re.match(r" {0,3}#{1,6}[ \t]", line):
            yield "".join(lines)
            lines = []
        lines.append(line)
        if not line.strip():
            yield "".join(lines)
            lines = []
    if lines:
        yield "".join(lines)


def _split_utf8(text: str, limit: int) -> Iterator[str]:
    """Split oversized text without dropping whitespace or cutting UTF-8 bytes."""
    data = text.encode("utf-8")
    while data:
        if len(data) <= limit:
            yield data.decode("utf-8")
            return
        window = data[:limit].decode("utf-8", errors="ignore")
        cut = len(window)
        for separator in ("\n\n", "\n", " "):
            boundary = window.rfind(separator, len(window) // 2)
            if boundary >= 0:
                cut = boundary + len(separator)
                break
        chunk = window[:cut]
        if not chunk:
            raise ValueError("Лимит части слишком мал для символа UTF-8")
        yield chunk
        data = data[len(chunk.encode("utf-8")):]


def _split_large_block(block: str, limit: int) -> Iterator[str]:
    lines = block.splitlines(keepends=True)
    marker = _opening_fence(lines[0])
    if marker is None:
        yield from _split_utf8(block, limit)
        return

    # Большой блок кода закрываем и заново открываем в каждой части.
    opening = lines[0]
    closed = len(lines) > 1 and _closing_fence(lines[-1], marker)
    closing = lines[-1] if closed else marker + "\n"
    body = "".join(lines[1:-1] if closed else lines[1:])
    overhead = len((opening + closing).encode("utf-8")) + 1
    if overhead + 4 > limit:
        raise ValueError("Заголовок блока кода превышает размер части README")
    for chunk in _split_utf8(body, limit - overhead):
        newline = "" if chunk.endswith("\n") else "\n"
        yield opening + chunk + newline + closing


def split_readme_documents(
    readme: str, limit: int = MAX_DOCUMENT_BYTES,
) -> list[str]:
    """Build Markdown documents with a strict UTF-8 byte limit per file."""
    if limit < 64:
        raise ValueError("Размер части README должен быть не меньше 64 байт")
    parts: list[str] = []
    pending: list[str] = []
    pending_bytes = 0
    for block in _markdown_blocks(readme):
        blocks = (
            [block] if len(block.encode("utf-8")) <= limit
            else _split_large_block(block, limit)
        )
        for piece in blocks:
            piece_bytes = len(piece.encode("utf-8"))
            if pending and pending_bytes + piece_bytes > limit:
                parts.append("".join(pending))
                pending = []
                pending_bytes = 0
            pending.append(piece)
            pending_bytes += piece_bytes
    if pending:
        parts.append("".join(pending))
    return parts


def create_readme_router() -> Router:

    router = Router(name="github-readme")

    cached_readme: str | None = None
    cached_at = 0.0

    cache_lock = asyncio.Lock()

    # -----------------------------------------
    # Получение README с GitHub.
    # -----------------------------------------

    async def get_readme(
        *,
        refresh: bool = False,
    ) -> str:

        nonlocal cached_readme, cached_at

        async with cache_lock:

            now = time.monotonic()

            # Используем кеш при переключении страниц.
            if (
                not refresh
                and cached_readme is not None
                and now - cached_at < CACHE_SECONDS
            ):
                return cached_readme

            # Загружаем актуальный файл.
            async with httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=True,
            ) as client:

                response = await client.get(README_URL)

                response.raise_for_status()

            # Защита от слишком больших файлов.
            if len(response.content) > MAX_README_BYTES:
                raise ValueError("README слишком большой")

            content = response.content.decode("utf-8-sig")

            if not content.strip():
                raise ValueError("README пуст")

            # Обновляем кеш.
            cached_readme = content
            cached_at = time.monotonic()

            return content

    # -----------------------------------------
    # Создание кнопок навигации.
    # -----------------------------------------

    def readme_keyboard(
        page: int,
        total: int,
    ) -> InlineKeyboardMarkup:

        rows = []

        navigation = []

        if page > 0:
            navigation.append(
                InlineKeyboardButton(
                    text="◀️ Назад",
                    callback_data=f"readme:page:{page - 1}",
                )
            )

        if page < total - 1:
            navigation.append(
                InlineKeyboardButton(
                    text="Вперёд ▶️",
                    callback_data=f"readme:page:{page + 1}",
                )
            )

        if navigation:
            rows.append(navigation)

        rows.append(
            [
                InlineKeyboardButton(
                    text="📥 Скачать README",
                    callback_data="readme:download",
                ),
                InlineKeyboardButton(
                    text="🔗 GitHub",
                    url=GITHUB_URL,
                ),
            ]
        )

        return InlineKeyboardMarkup(
            inline_keyboard=rows
        )

    # -----------------------------------------
    # Формирование страницы документации.
    # -----------------------------------------

    def render_page(
        readme: str,
        page: int,
    ) -> tuple[str, InlineKeyboardMarkup]:

        pages = split_text(readme, PAGE_SIZE)

        page = max(0, min(page, len(pages) - 1))

        text = (
            f"📖 README.md\n"
            f"Страница {page + 1} из {len(pages)}\n\n"
            f"{pages[page]}"
        )

        keyboard = readme_keyboard(
            page,
            len(pages),
        )

        return text, keyboard

    # -----------------------------------------
    # Команда /readme.
    # -----------------------------------------

    @router.message(Command("readme"))
    async def readme_command(message: Message) -> None:

        try:
            # При новой команде всегда проверяем GitHub.
            readme = await get_readme(refresh=True)

            text, keyboard = render_page(readme, 0)

            await message.answer(
                text,
                reply_markup=keyboard,
                parse_mode=None,
            )

        except (httpx.HTTPError, UnicodeError, ValueError):
            logger.exception("Не удалось загрузить README")

            await message.answer(
                "❌ Не удалось загрузить README с GitHub."
            )

    # -----------------------------------------
    # Переключение страниц.
    # -----------------------------------------

    @router.callback_query(
        F.data.startswith("readme:page:")
    )
    async def readme_page(
        callback: CallbackQuery,
    ) -> None:

        if not isinstance(callback.message, Message):
            await callback.answer(
                "Сообщение недоступно.",
                show_alert=True,
            )
            return

        await callback.answer()

        try:
            page = int(callback.data.rsplit(":", 1)[1])

            readme = await get_readme()

            text, keyboard = render_page(
                readme,
                page,
            )

            await callback.message.edit_text(
                text,
                reply_markup=keyboard,
                parse_mode=None,
            )

        except (httpx.HTTPError, UnicodeError, ValueError):
            logger.exception("Ошибка чтения README")

            await callback.message.answer(
                "❌ Не удалось открыть страницу README."
            )

    # -----------------------------------------
    # Скачивание README частями, подходящими для просмотра в Telegram.
    # -----------------------------------------

    @router.callback_query(
        F.data == "readme:download"
    )
    async def readme_download(
        callback: CallbackQuery,
    ) -> None:

        if not isinstance(callback.message, Message):
            await callback.answer(
                "Сообщение недоступно.",
                show_alert=True,
            )
            return

        await callback.answer()

        try:
            readme = await get_readme()

            parts = split_readme_documents(readme)
            total = len(parts)
            for index, part in enumerate(parts, start=1):
                filename = "README.md" if total == 1 else f"README_{index}.md"
                caption = "📖 Актуальный README проекта с GitHub."
                if total > 1:
                    caption = f"📖 README — часть {index} из {total}."
                await callback.message.answer_document(
                    document=BufferedInputFile(part.encode("utf-8"), filename=filename),
                    caption=caption,
                    parse_mode=None,
                )

        except (httpx.HTTPError, UnicodeError, ValueError):
            logger.exception("Ошибка скачивания README")

            await callback.message.answer(
                "❌ Не удалось скачать README."
            )

    return router
