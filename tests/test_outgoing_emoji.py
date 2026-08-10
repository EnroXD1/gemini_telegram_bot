from typing import Any

import pytest
from aiogram.enums import MessageEntityType
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import SendMessage, SendPhoto
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    MessageEntity,
)

from bot.outgoing_emoji import (
    OutgoingEmojiMiddleware,
    apply_outgoing_custom_emojis,
)

EMOJI_ID = "5000000000000000400"


def test_outgoing_text_preserves_formatting_and_adds_custom_emoji() -> None:
    method = SendMessage(
        chat_id=1,
        text="✅ Готово",
        entities=[
            MessageEntity(type=MessageEntityType.BOLD, offset=2, length=6)
        ],
    )

    themed = apply_outgoing_custom_emojis(method, {"✅": EMOJI_ID})

    assert themed is not method
    assert themed.entities is not None
    assert [entity.type for entity in themed.entities] == [
        MessageEntityType.CUSTOM_EMOJI,
        MessageEntityType.BOLD,
    ]
    assert themed.entities[0].custom_emoji_id == EMOJI_ID


def test_existing_custom_emoji_is_not_duplicated() -> None:
    entity = MessageEntity(
        type=MessageEntityType.CUSTOM_EMOJI,
        offset=0,
        length=1,
        custom_emoji_id=EMOJI_ID,
    )
    method = SendMessage(chat_id=1, text="✅", entities=[entity])

    themed = apply_outgoing_custom_emojis(method, {"✅": EMOJI_ID})

    assert themed is method


def test_outgoing_caption_and_inline_button_are_themed() -> None:
    method = SendPhoto(
        chat_id=1,
        photo="file-id",
        caption="🎬 Видео готово ✅",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🎬 Открыть гайд", callback_data="guide")]
            ]
        ),
    )

    themed = apply_outgoing_custom_emojis(
        method, {"🎬": "5000000000000000401", "✅": EMOJI_ID}
    )

    assert themed.caption_entities is not None
    assert len(themed.caption_entities) == 2
    button = themed.reply_markup.inline_keyboard[0][0]
    assert button.text == "Открыть гайд"
    assert button.icon_custom_emoji_id == "5000000000000000401"


class FakeTheme:
    async def service_ids(self) -> dict[str, str]:
        return {"✅": EMOJI_ID}


@pytest.mark.asyncio
async def test_middleware_retries_plain_request_when_telegram_rejects_custom_emoji() -> None:
    middleware = OutgoingEmojiMiddleware(FakeTheme())  # type: ignore[arg-type]
    original = SendMessage(chat_id=1, text="✅ Готово")
    calls: list[Any] = []

    async def make_request(_: Any, method: Any) -> str:
        calls.append(method)
        if method is not original:
            raise TelegramBadRequest(method=method, message="bad custom emoji")
        return "ok"

    result = await middleware(make_request, object(), original)

    assert result == "ok"
    assert len(calls) == 2
    assert calls[1] is original
