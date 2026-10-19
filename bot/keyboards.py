from typing import Literal

from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)

from .guide import GUIDE_CALLBACK_DATA

RESET_BUTTON = "🆕 Новый диалог"
WALL_BUTTON = "🖼 Нарезать фото"
STATUS_BUTTON = "📊 Статус"
HELP_BUTTON = "ℹ️ Помощь"
SETTINGS_BUTTON = "⚙️ Настройки"
DIAGNOSTICS_BUTTON = "🩺 Диагностика"
HIDE_KEYBOARD_BUTTON = "⌨️ Скрыть клавиатуру"
MORE_BUTTON = "➡️ Другое"
BACK_BUTTON = "⬅️ Назад"


def main_keyboard(
    *, owner: bool = False, page: Literal["main", "other"] = "main"
) -> ReplyKeyboardMarkup:
    if page == "other":
        rows = [[STATUS_BUTTON, BACK_BUTTON]]
        if owner:
            rows.insert(0, [SETTINGS_BUTTON, DIAGNOSTICS_BUTTON])
    else:
        rows = [[RESET_BUTTON, WALL_BUTTON], [HELP_BUTTON, MORE_BUTTON]]
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=text) for text in row] for row in rows],
        resize_keyboard=True,
        is_persistent=False,
        input_field_placeholder="Напишите вопрос или выберите действие",
    )


def help_keyboard(*, owner: bool = False, show_menu: bool = True) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text="🎬 Видеоинструкция", callback_data=GUIDE_CALLBACK_DATA)],
    ]
    if show_menu:
        rows.append([InlineKeyboardButton(text="⌨️ Показать меню", callback_data="ui:menu")])
    if owner:
        rows.append([InlineKeyboardButton(text="⚙️ Настройки", callback_data="cfg:home")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def answer_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🆕 Новый диалог", callback_data="ui:reset"),
                InlineKeyboardButton(text="⌨️ Меню", callback_data="ui:menu"),
            ]
        ]
    )
