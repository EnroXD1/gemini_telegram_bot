from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from aiogram.client.session.middlewares.base import BaseRequestMiddleware
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, MessageEntity

from .emoji_theme import (
    EmojiTheme,
    find_first_service_custom_emoji,
    merge_service_custom_emojis,
)

logger = logging.getLogger(__name__)


class OutgoingEmojiMiddleware(BaseRequestMiddleware):
    """Apply the owner's custom-emoji map to every outgoing Bot API method."""

    def __init__(self, emoji_theme: EmojiTheme) -> None:
        self._emoji_theme = emoji_theme

    async def __call__(
        self,
        make_request: Any,
        bot: Any,
        method: Any,
    ) -> Any:
        service_ids = await self._emoji_theme.service_ids()
        themed_method = apply_outgoing_custom_emojis(method, service_ids)
        if themed_method is method:
            return await make_request(bot, method)
        try:
            return await make_request(bot, themed_method)
        except TelegramBadRequest as exc:
            logger.debug(
                "Custom emoji request was rejected; retrying with plain emoji: %s",
                exc,
            )
            return await make_request(bot, method)


def apply_outgoing_custom_emojis(method: Any, service_ids: Mapping[str, str]) -> Any:
    updates: dict[str, Any] = {}
    for text_field, entities_field in (
        ("text", "entities"),
        ("caption", "caption_entities"),
    ):
        text = getattr(method, text_field, None)
        if not isinstance(text, str) or not text:
            continue
        raw_entities = getattr(method, entities_field, None)
        existing_entities = (
            raw_entities
            if isinstance(raw_entities, list)
            and all(isinstance(item, MessageEntity) for item in raw_entities)
            else None
        )
        themed = merge_service_custom_emojis(text, existing_entities, service_ids)
        if themed.entities != existing_entities:
            updates[entities_field] = themed.entities

    reply_markup = getattr(method, "reply_markup", None)
    if isinstance(reply_markup, InlineKeyboardMarkup):
        themed_markup = _theme_inline_keyboard(reply_markup, service_ids)
        if themed_markup is not reply_markup:
            updates["reply_markup"] = themed_markup

    if not updates:
        return method
    return method.model_copy(update=updates)


def _theme_inline_keyboard(
    markup: InlineKeyboardMarkup,
    service_ids: Mapping[str, str],
) -> InlineKeyboardMarkup:
    changed = False
    rows: list[list[InlineKeyboardButton]] = []
    for row in markup.inline_keyboard:
        themed_row: list[InlineKeyboardButton] = []
        for button in row:
            themed_button = _theme_inline_button(button, service_ids)
            changed = changed or themed_button is not button
            themed_row.append(themed_button)
        rows.append(themed_row)
    if not changed:
        return markup
    return markup.model_copy(update={"inline_keyboard": rows})


def _theme_inline_button(
    button: InlineKeyboardButton,
    service_ids: Mapping[str, str],
) -> InlineKeyboardButton:
    if button.icon_custom_emoji_id:
        return button
    match = find_first_service_custom_emoji(button.text, service_ids)
    if match is None:
        return button
    start, end, custom_emoji_id = match
    text = (button.text[:start] + button.text[end:]).strip()
    if not text:
        return button
    return button.model_copy(
        update={"text": text, "icon_custom_emoji_id": custom_emoji_id}
    )
