from datetime import UTC, datetime
from pathlib import Path

import pytest
from aiogram.enums import ChatType, MessageEntityType, StickerType
from aiogram.types import Chat, Message, MessageEntity, Sticker, StickerSet, User

from bot.emoji_theme import (
    CustomEmoji,
    EmojiTheme,
    OwnerEmojiPaletteFilter,
    extract_custom_emojis,
    extract_emoji_pack_names,
    extract_sticker_set_emojis,
)
from bot.storage import Storage


def make_message(*, user_id: int = 100, text: str = "😀 😎") -> Message:
    return Message(
        message_id=1,
        date=datetime.now(UTC),
        chat=Chat(id=user_id, type=ChatType.PRIVATE, first_name="Owner"),
        from_user=User(id=user_id, is_bot=False, first_name="Owner"),
        text=text,
        entities=[
            MessageEntity(
                type=MessageEntityType.CUSTOM_EMOJI,
                offset=0,
                length=2,
                custom_emoji_id="5000000000000000001",
            ),
            MessageEntity(
                type=MessageEntityType.CUSTOM_EMOJI,
                offset=3,
                length=2,
                custom_emoji_id="5000000000000000002",
            ),
        ],
    )


def test_custom_emoji_entities_are_extracted_with_utf16_offsets() -> None:
    emojis = extract_custom_emojis(make_message())

    assert emojis == (
        CustomEmoji("5000000000000000001", "😀"),
        CustomEmoji("5000000000000000002", "😎"),
    )


@pytest.mark.asyncio
async def test_only_owner_emoji_palette_messages_are_captured() -> None:
    palette_filter = OwnerEmojiPaletteFilter(frozenset({100}))

    owner_result = await palette_filter(make_message())
    other_result = await palette_filter(make_message(user_id=200))
    mixed_result = await palette_filter(make_message(text="😀 подпись 😎"))

    assert isinstance(owner_result, dict)
    assert len(owner_result["custom_emojis"]) == 2
    assert other_result is False
    assert mixed_result is False


@pytest.mark.asyncio
async def test_emoji_theme_is_persisted_and_decorates_service_text(
    tmp_path: Path,
) -> None:
    storage = Storage(tmp_path / "bot.sqlite3")
    await storage.open()
    try:
        theme = EmojiTheme(storage)
        added, total = await theme.add(extract_custom_emojis(make_message()))

        assert (added, total) == (2, 2)
        themed = await theme.decorate("Готовлю ответ", fallback="⏳")
        assert themed.text == "😀 Готовлю ответ"
        assert themed.entities is not None
        assert themed.entities[0].custom_emoji_id == "5000000000000000001"
        assert themed.entities[0].length == 2

        restored = EmojiTheme(storage)
        assert await restored.items() == (
            CustomEmoji("5000000000000000001", "😀"),
            CustomEmoji("5000000000000000002", "😎"),
        )
    finally:
        await storage.close()


def test_extract_emoji_pack_names_deduplicates_links() -> None:
    assert extract_emoji_pack_names(
        "https://t.me/addemoji/FirstPack и t.me/addemoji/Second_pack "
        "https://t.me/addemoji/FirstPack"
    ) == ("FirstPack", "Second_pack")


def test_extract_sticker_set_emojis_ignores_regular_sticker_sets() -> None:
    sticker = Sticker(
        file_id="file",
        file_unique_id="unique",
        type=StickerType.CUSTOM_EMOJI,
        width=100,
        height=100,
        is_animated=True,
        is_video=False,
        emoji="✨",
        custom_emoji_id="5368324170671202286",
    )
    custom_set = StickerSet(
        name="CustomPack",
        title="Custom pack",
        sticker_type=StickerType.CUSTOM_EMOJI,
        stickers=[sticker],
    )
    regular_set = custom_set.model_copy(
        update={"sticker_type": StickerType.REGULAR}
    )

    assert extract_sticker_set_emojis(custom_set)[0].custom_emoji_id == (
        "5368324170671202286"
    )
    assert extract_sticker_set_emojis(regular_set) == ()
