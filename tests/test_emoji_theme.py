from datetime import UTC, datetime
from pathlib import Path

import pytest
from aiogram.enums import ChatType, MessageEntityType, StickerType
from aiogram.types import Chat, Message, MessageEntity, Sticker, StickerSet, User

from bot.emoji_theme import (
    SERVICE_CUSTOM_EMOJI_IDS,
    CustomEmoji,
    EmojiTheme,
    GroupCustomEmojiOnlyFilter,
    OwnerEmojiPaletteFilter,
    apply_service_custom_emojis,
    extract_custom_emojis,
    extract_emoji_pack_names,
    extract_sticker_set_emojis,
    merge_service_custom_emojis,
)
from bot.storage import Storage


def make_message(
    *,
    user_id: int = 100,
    text: str = "😀 😎",
    chat_type: ChatType = ChatType.PRIVATE,
) -> Message:
    return Message(
        message_id=1,
        date=datetime.now(UTC),
        chat=Chat(id=user_id, type=chat_type, first_name="Owner"),
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
    group_result = await palette_filter(make_message(chat_type=ChatType.GROUP))

    assert isinstance(owner_result, dict)
    assert len(owner_result["custom_emojis"]) == 2
    assert other_result is False
    assert mixed_result is False
    assert group_result is False


@pytest.mark.asyncio
async def test_custom_emoji_only_messages_are_silently_consumed_in_groups() -> None:
    group_filter = GroupCustomEmojiOnlyFilter()

    assert await group_filter(make_message(chat_type=ChatType.GROUP)) is True
    assert await group_filter(make_message(chat_type=ChatType.SUPERGROUP)) is True
    assert await group_filter(make_message()) is False
    assert (
        await group_filter(
            make_message(text="😀 подпись 😎", chat_type=ChatType.GROUP)
        )
        is False
    )


@pytest.mark.asyncio
async def test_emoji_theme_uses_stable_service_emoji_instead_of_random_palette(
    tmp_path: Path,
) -> None:
    storage = Storage(tmp_path / "bot.sqlite3")
    await storage.open()
    try:
        theme = EmojiTheme(storage)
        themed = await theme.service_text("⏳ Готовлю ответ")

        assert themed.text == "⏳ Готовлю ответ"
        assert themed.entities is not None
        assert themed.entities[0].custom_emoji_id == SERVICE_CUSTOM_EMOJI_IDS["⏳"]
        assert themed.entities[0].length == 1
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_service_emoji_override_is_persisted_and_reset(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "bot.sqlite3")
    await storage.open()
    try:
        theme = EmojiTheme(storage)
        await theme.set_service_emoji("🔄", "5000000000000000099")

        themed = await EmojiTheme(storage).service_text("🔄 Переключаюсь")
        assert themed.entities is not None
        assert themed.entities[0].custom_emoji_id == "5000000000000000099"

        await theme.reset_service_emoji("🔄")
        reset = await theme.service_text("🔄 Переключаюсь")
        assert reset.entities is not None
        assert reset.entities[0].custom_emoji_id == (
            SERVICE_CUSTOM_EMOJI_IDS["🔄"]
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_any_emoji_mapping_can_be_persisted_removed_and_cleared(
    tmp_path: Path,
) -> None:
    storage = Storage(tmp_path / "bot.sqlite3")
    await storage.open()
    try:
        theme = EmojiTheme(storage)
        await theme.set_service_emoji("✅", "5000000000000000100")

        restored = EmojiTheme(storage)
        assert (await restored.service_ids())["✅"] == "5000000000000000100"

        await restored.remove_service_emoji("✅")
        assert "✅" not in await restored.service_ids()

        await restored.clear_service_emojis()
        assert await EmojiTheme(storage).service_ids() == {}
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


def test_service_custom_emoji_mapping_uses_configured_ids() -> None:
    text = "⏳ затем 🔎 затем ✍️ и 🔄"
    themed = apply_service_custom_emojis(text)

    assert themed.text == text
    assert themed.entities is not None
    assert [entity.extract_from(text) for entity in themed.entities] == [
        "⏳",
        "🔎",
        "✍️",
        "🔄",
    ]
    assert [entity.custom_emoji_id for entity in themed.entities] == [
        SERVICE_CUSTOM_EMOJI_IDS["⏳"],
        SERVICE_CUSTOM_EMOJI_IDS["🔎"],
        SERVICE_CUSTOM_EMOJI_IDS["✍️"],
        SERVICE_CUSTOM_EMOJI_IDS["🔄"],
    ]


def test_longest_custom_emoji_mapping_wins_without_overlapping_entities() -> None:
    text = "❤️ и ❤"
    themed = apply_service_custom_emojis(
        text,
        {"❤": "5000000000000000200", "❤️": "5000000000000000201"},
    )

    assert themed.entities is not None
    assert [entity.extract_from(text) for entity in themed.entities] == ["❤️", "❤"]
    assert [entity.custom_emoji_id for entity in themed.entities] == [
        "5000000000000000201",
        "5000000000000000200",
    ]


def test_custom_emoji_entities_merge_with_formatting_but_skip_code() -> None:
    text = "✅ ✅"
    existing = [
        MessageEntity(type=MessageEntityType.BOLD, offset=0, length=1),
        MessageEntity(type=MessageEntityType.CODE, offset=2, length=1),
    ]
    themed = merge_service_custom_emojis(
        text, existing, {"✅": "5000000000000000300"}
    )

    assert themed.entities is not None
    custom = [
        entity
        for entity in themed.entities
        if entity.type == MessageEntityType.CUSTOM_EMOJI
    ]
    assert len(custom) == 1
    assert custom[0].offset == 0
