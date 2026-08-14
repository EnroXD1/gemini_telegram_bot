from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

from aiogram.enums import ChatType, MessageEntityType, StickerType
from aiogram.filters import BaseFilter
from aiogram.types import Message, MessageEntity, StickerSet

from .storage import Storage

_SERVICE_SETTING_KEY = "service_custom_emoji_ids"
_MAX_SERVICE_ITEMS = 128
_VARIATION_SELECTORS = frozenset({"\ufe0e", "\ufe0f"})
_PACK_LINK_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?t\.me/addemoji/([A-Za-z0-9_]+)",
    re.IGNORECASE,
)

# Defaults used until the owner saves a custom mapping. Any other emoji can be
# added at runtime with /emoji set.
SERVICE_CUSTOM_EMOJI_IDS: dict[str, str] = {
    "🔄": "5345778951031658558",
    "⏳": "5345988476716226868",
    "🔎": "5231012545799666522",
    "✍️": "5213277341639254218",
}


@dataclass(frozen=True, slots=True)
class CustomEmoji:
    custom_emoji_id: str
    alternative: str


@dataclass(frozen=True, slots=True)
class ThemedText:
    text: str
    entities: list[MessageEntity] | None


class EmojiTheme:
    def __init__(self, storage: Storage) -> None:
        self._storage = storage
        self._service_ids: dict[str, str] | None = None
        self._lock = asyncio.Lock()

    async def service_ids(self) -> dict[str, str]:
        async with self._lock:
            return dict(await self._load_service_ids_locked())

    async def set_service_emoji(
        self, alternative: str, custom_emoji_id: str
    ) -> None:
        alternative = _canonical_service_alternative(alternative)
        _validate_service_alternative(alternative)
        if not custom_emoji_id.isdigit():
            raise ValueError("custom_emoji_id должен состоять только из цифр")
        async with self._lock:
            service_ids = dict(await self._load_service_ids_locked())
            if (
                alternative not in service_ids
                and len(service_ids) >= _MAX_SERVICE_ITEMS
            ):
                raise ValueError(
                    f"Можно сохранить не более {_MAX_SERVICE_ITEMS} соответствий"
                )
            service_ids[alternative] = custom_emoji_id
            await self._save_service_ids_locked(service_ids)

    async def reset_service_emoji(self, alternative: str | None = None) -> None:
        if alternative is not None:
            alternative = _canonical_service_alternative(alternative)
            _validate_service_alternative(alternative)
        async with self._lock:
            service_ids = dict(await self._load_service_ids_locked())
            if alternative is None:
                service_ids = _normalized_service_ids(SERVICE_CUSTOM_EMOJI_IDS)
            elif alternative in _normalized_service_ids(SERVICE_CUSTOM_EMOJI_IDS):
                service_ids[alternative] = _normalized_service_ids(
                    SERVICE_CUSTOM_EMOJI_IDS
                )[alternative]
            else:
                service_ids.pop(alternative, None)
            await self._save_service_ids_locked(service_ids)

    async def remove_service_emoji(self, alternative: str) -> None:
        alternative = _canonical_service_alternative(alternative)
        _validate_service_alternative(alternative)
        async with self._lock:
            service_ids = dict(await self._load_service_ids_locked())
            if alternative not in service_ids:
                raise ValueError(f"Для {alternative} соответствие не настроено")
            service_ids.pop(alternative)
            await self._save_service_ids_locked(service_ids)

    async def clear_service_emojis(self) -> None:
        async with self._lock:
            await self._save_service_ids_locked({})

    async def service_text(self, text: str) -> ThemedText:
        async with self._lock:
            service_ids = await self._load_service_ids_locked()
            return apply_service_custom_emojis(text, service_ids)

    async def _load_service_ids_locked(self) -> dict[str, str]:
        if self._service_ids is not None:
            return self._service_ids
        raw = await self._storage.get_runtime_setting(_SERVICE_SETTING_KEY)
        if raw is None:
            service_ids = _normalized_service_ids(SERVICE_CUSTOM_EMOJI_IDS)
        else:
            service_ids = {}
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    for alternative, custom_emoji_id in payload.items():
                        normalized = (
                            _canonical_service_alternative(alternative)
                            if isinstance(alternative, str)
                            else ""
                        )
                        if (
                            (normalized in service_ids or len(service_ids) < _MAX_SERVICE_ITEMS)
                            and isinstance(custom_emoji_id, str)
                            and custom_emoji_id.isdigit()
                            and _is_service_alternative(normalized)
                        ):
                            service_ids[normalized] = custom_emoji_id
            except (TypeError, ValueError):
                service_ids = _normalized_service_ids(SERVICE_CUSTOM_EMOJI_IDS)
        self._service_ids = service_ids
        return self._service_ids

    async def _save_service_ids_locked(self, service_ids: dict[str, str]) -> None:
        self._service_ids = service_ids
        await self._storage.set_runtime_setting(
            _SERVICE_SETTING_KEY,
            json.dumps(service_ids, ensure_ascii=False, separators=(",", ":")),
        )


class OwnerEmojiPaletteFilter(BaseFilter):
    def __init__(self, owner_ids: frozenset[int]) -> None:
        self._owner_ids = owner_ids

    async def __call__(self, message: Message) -> bool | dict[str, object]:
        user = message.from_user
        if (
            message.chat.type != ChatType.PRIVATE
            or user is None
            or user.id not in self._owner_ids
        ):
            return False
        emojis = extract_custom_emojis(message)
        pack_names = extract_emoji_pack_names(message.text or message.caption or "")
        if not pack_names and (not emojis or not _contains_only_custom_emojis(message)):
            return False
        return {
            "custom_emojis": emojis,
            "emoji_pack_names": pack_names,
        }


class GroupCustomEmojiOnlyFilter(BaseFilter):
    async def __call__(self, message: Message) -> bool:
        if message.chat.type == ChatType.PRIVATE:
            return False
        return bool(extract_custom_emojis(message)) and _contains_only_custom_emojis(
            message
        )


def extract_emoji_pack_names(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_PACK_LINK_PATTERN.findall(text)))


def apply_service_custom_emojis(
    text: str,
    service_ids: Mapping[str, str] = SERVICE_CUSTOM_EMOJI_IDS,
) -> ThemedText:
    return merge_service_custom_emojis(text, None, service_ids)


def merge_service_custom_emojis(
    text: str,
    existing_entities: list[MessageEntity] | None,
    service_ids: Mapping[str, str] = SERVICE_CUSTOM_EMOJI_IDS,
) -> ThemedText:
    entities = list(existing_entities or [])
    occupied: list[tuple[int, int]] = [
        (entity.offset, entity.offset + entity.length)
        for entity in entities
        if entity.type == MessageEntityType.CUSTOM_EMOJI
    ]
    blocked = [
        (entity.offset, entity.offset + entity.length)
        for entity in entities
        if entity.type
        not in {
            MessageEntityType.BOLD,
            MessageEntityType.ITALIC,
            MessageEntityType.UNDERLINE,
            MessageEntityType.STRIKETHROUGH,
            MessageEntityType.SPOILER,
            MessageEntityType.BLOCKQUOTE,
            MessageEntityType.EXPANDABLE_BLOCKQUOTE,
            MessageEntityType.CUSTOM_EMOJI,
        }
    ]
    normalized_text, original_positions = _text_without_variation_selectors(text)
    mappings = sorted(
        _normalized_service_ids(service_ids).items(),
        key=lambda item: (_utf16_length(item[0]), len(item[0])),
        reverse=True,
    )
    for alternative, custom_emoji_id in mappings:
        if not alternative or not custom_emoji_id.isdigit():
            continue
        start = 0
        while True:
            index = normalized_text.find(alternative, start)
            if index < 0:
                break
            normalized_end = index + len(alternative)
            original_start = original_positions[index]
            original_end = (
                original_positions[normalized_end]
                if normalized_end < len(original_positions)
                else len(text)
            )
            offset = _utf16_length(text[:original_start])
            end = _utf16_length(text[:original_end])
            overlaps = any(
                offset < existing_end and end > existing_start
                for existing_start, existing_end in (*occupied, *blocked)
            )
            if overlaps:
                start = normalized_end
                continue
            entities.append(
                MessageEntity(
                    type=MessageEntityType.CUSTOM_EMOJI,
                    offset=offset,
                    length=end - offset,
                    custom_emoji_id=custom_emoji_id,
                )
            )
            occupied.append((offset, end))
            start = normalized_end
    entities.sort(key=lambda entity: (entity.offset, -entity.length))
    return ThemedText(text=text, entities=entities or None)


def find_first_service_custom_emoji(
    text: str,
    service_ids: Mapping[str, str] = SERVICE_CUSTOM_EMOJI_IDS,
) -> tuple[int, int, str] | None:
    """Return the first matching emoji span as Python indexes and its custom ID."""
    normalized_text, original_positions = _text_without_variation_selectors(text)
    matches: list[tuple[int, int, str]] = []
    for alternative, custom_emoji_id in _normalized_service_ids(service_ids).items():
        index = normalized_text.find(alternative)
        if index < 0:
            continue
        normalized_end = index + len(alternative)
        original_start = original_positions[index]
        original_end = (
            original_positions[normalized_end]
            if normalized_end < len(original_positions)
            else len(text)
        )
        matches.append((original_start, original_end, custom_emoji_id))
    if not matches:
        return None
    return min(matches, key=lambda item: (item[0], -(item[1] - item[0])))


def extract_sticker_set_emojis(sticker_set: StickerSet) -> tuple[CustomEmoji, ...]:
    if sticker_set.sticker_type != StickerType.CUSTOM_EMOJI:
        return ()
    result: list[CustomEmoji] = []
    known_ids: set[str] = set()
    for sticker in sticker_set.stickers:
        custom_emoji_id = sticker.custom_emoji_id
        if not custom_emoji_id or custom_emoji_id in known_ids:
            continue
        item = _validated_custom_emoji(custom_emoji_id, sticker.emoji or "✨")
        if item is not None:
            result.append(item)
            known_ids.add(custom_emoji_id)
    return tuple(result)


def extract_custom_emojis(message: Message) -> tuple[CustomEmoji, ...]:
    text, entities = _text_and_entities(message)
    if not text or not entities:
        return ()
    encoded = text.encode("utf-16-le")
    result: list[CustomEmoji] = []
    known_ids: set[str] = set()
    for entity in entities:
        if entity.type != MessageEntityType.CUSTOM_EMOJI:
            continue
        custom_emoji_id = entity.custom_emoji_id
        if not custom_emoji_id or custom_emoji_id in known_ids:
            continue
        start = entity.offset * 2
        end = (entity.offset + entity.length) * 2
        try:
            alternative = encoded[start:end].decode("utf-16-le")
        except UnicodeDecodeError:
            continue
        item = _validated_custom_emoji(custom_emoji_id, alternative)
        if item is not None:
            result.append(item)
            known_ids.add(custom_emoji_id)
    return tuple(result)


def _contains_only_custom_emojis(message: Message) -> bool:
    text, entities = _text_and_entities(message)
    if not text or not entities:
        return False
    encoded = bytearray(text.encode("utf-16-le"))
    found = False
    for entity in entities:
        if entity.type != MessageEntityType.CUSTOM_EMOJI:
            continue
        found = True
        start = entity.offset * 2
        end = (entity.offset + entity.length) * 2
        encoded[start:end] = b" \x00" * entity.length
    try:
        remaining = bytes(encoded).decode("utf-16-le")
    except UnicodeDecodeError:
        return False
    return found and not remaining.strip()


def _text_and_entities(
    message: Message,
) -> tuple[str | None, list[MessageEntity] | None]:
    if message.text is not None:
        return message.text, message.entities
    return message.caption, message.caption_entities


def _validated_custom_emoji(
    custom_emoji_id: object, alternative: object
) -> CustomEmoji | None:
    if not isinstance(custom_emoji_id, str) or not custom_emoji_id.isdigit():
        return None
    if not isinstance(alternative, str) or not alternative.strip():
        return None
    if _utf16_length(alternative) > 32:
        return None
    return CustomEmoji(custom_emoji_id=custom_emoji_id, alternative=alternative)


def _validate_service_alternative(alternative: str) -> None:
    if not _is_service_alternative(alternative):
        raise ValueError(
            "Укажите один эмодзи без пробелов (составные эмодзи тоже поддерживаются)"
        )


def _canonical_service_alternative(alternative: str) -> str:
    return "".join(
        character
        for character in alternative
        if character not in _VARIATION_SELECTORS
    )


def _normalized_service_ids(
    service_ids: Mapping[str, str],
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for alternative, custom_emoji_id in service_ids.items():
        canonical = _canonical_service_alternative(alternative)
        if (
            canonical
            and isinstance(custom_emoji_id, str)
            and custom_emoji_id.isdigit()
        ):
            normalized[canonical] = custom_emoji_id
    return normalized


def _text_without_variation_selectors(text: str) -> tuple[str, list[int]]:
    characters: list[str] = []
    original_positions: list[int] = []
    for index, character in enumerate(text):
        if character in _VARIATION_SELECTORS:
            continue
        characters.append(character)
        original_positions.append(index)
    return "".join(characters), original_positions


def _is_service_alternative(alternative: str) -> bool:
    if (
        not alternative
        or alternative != alternative.strip()
        or any(character.isspace() for character in alternative)
        or _utf16_length(alternative) > 32
    ):
        return False
    return any(
        unicodedata.category(character).startswith("S")
        or character == "\N{COMBINING ENCLOSING KEYCAP}"
        for character in alternative
    )


def _utf16_length(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2
