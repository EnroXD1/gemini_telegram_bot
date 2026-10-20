from __future__ import annotations

import asyncio
import logging
import time

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
    # Скачивание полного README.
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

            document = BufferedInputFile(
                readme.encode("utf-8"),
                filename="README.md",
            )

            await callback.message.answer_document(
                document=document,
                caption="📖 Актуальный README проекта с GitHub.",
            )

        except (httpx.HTTPError, UnicodeError, ValueError):
            logger.exception("Ошибка скачивания README")

            await callback.message.answer(
                "❌ Не удалось скачать README."
            )

    return router