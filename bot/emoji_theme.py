from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from aiogram.enums import MessageEntityType, StickerType
from aiogram.filters import BaseFilter
from aiogram.types import Message, MessageEntity, StickerSet

from .storage import Storage

_SETTING_KEY = "custom_emoji_theme"
_SERVICE_SETTING_KEY = "service_custom_emoji_ids"
_MAX_THEME_ITEMS = 128
_PACK_LINK_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?t\.me/addemoji/([A-Za-z0-9_]+)",
    re.IGNORECASE,
)

# Edit this table to change custom emoji used in service messages.
# The dictionary key stays as a readable fallback when Telegram cannot render
# the custom emoji entity.
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
        self._items: tuple[CustomEmoji, ...] | None = None
        self._service_ids: dict[str, str] | None = None
        self._cursor = 0
        self._lock = asyncio.Lock()

    async def items(self) -> tuple[CustomEmoji, ...]:
        async with self._lock:
            return await self._load_locked()

    async def first_id(self) -> str | None:
        items = await self.items()
        return items[0].custom_emoji_id if items else None

    async def service_ids(self) -> dict[str, str]:
        async with self._lock:
            return dict(await self._load_service_ids_locked())

    async def set_service_emoji(
        self, alternative: str, custom_emoji_id: str
    ) -> None:
        if alternative not in SERVICE_CUSTOM_EMOJI_IDS:
            raise ValueError("Этот служебный символ не поддерживается")
        if not custom_emoji_id.isdigit():
            raise ValueError("custom_emoji_id должен состоять только из цифр")
        async with self._lock:
            service_ids = dict(await self._load_service_ids_locked())
            service_ids[alternative] = custom_emoji_id
            await self._save_service_ids_locked(service_ids)

    async def reset_service_emoji(self, alternative: str | None = None) -> None:
        async with self._lock:
            service_ids = dict(await self._load_service_ids_locked())
            if alternative is None:
                service_ids = dict(SERVICE_CUSTOM_EMOJI_IDS)
            elif alternative in SERVICE_CUSTOM_EMOJI_IDS:
                service_ids[alternative] = SERVICE_CUSTOM_EMOJI_IDS[alternative]
            else:
                raise ValueError("Этот служебный символ не поддерживается")
            await self._save_service_ids_locked(service_ids)

    async def service_text(self, text: str) -> ThemedText:
        async with self._lock:
            service_ids = await self._load_service_ids_locked()
            return apply_service_custom_emojis(text, service_ids)

    async def add(
        self, incoming: tuple[CustomEmoji, ...]
    ) -> tuple[int, int]:
        async with self._lock:
            current = list(await self._load_locked())
            known_ids = {item.custom_emoji_id for item in current}
            added = 0
            for item in incoming:
                if item.custom_emoji_id in known_ids:
                    continue
                current.append(item)
                known_ids.add(item.custom_emoji_id)
                added += 1
            current = current[-_MAX_THEME_ITEMS:]
            self._items = tuple(current)
            await self._storage.set_runtime_setting(
                _SETTING_KEY,
                json.dumps(
                    [
                        {
                            "id": item.custom_emoji_id,
                            "alternative": item.alternative,
                        }
                        for item in current
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
            return added, len(current)

    async def decorate(self, text: str, *, fallback: str) -> ThemedText:
        async with self._lock:
            items = await self._load_locked()
            service_ids = await self._load_service_ids_locked()
            if not items:
                return apply_service_custom_emojis(
                    f"{fallback} {text}", service_ids
                )
            item = items[self._cursor % len(items)]
            self._cursor += 1
        decorated = f"{item.alternative} {text}"
        themed = apply_service_custom_emojis(text, service_ids)
        prefix_length = _utf16_length(f"{item.alternative} ")
        entities = [
            entity.model_copy(update={"offset": entity.offset + prefix_length})
            for entity in themed.entities or []
        ]
        entities.append(
            MessageEntity(
                type=MessageEntityType.CUSTOM_EMOJI,
                offset=0,
                length=_utf16_length(item.alternative),
                custom_emoji_id=item.custom_emoji_id,
            )
        )
        entities.sort(key=lambda entity: entity.offset)
        return ThemedText(
            text=decorated,
            entities=entities,
        )

    async def _load_locked(self) -> tuple[CustomEmoji, ...]:
        if self._items is not None:
            return self._items
        raw = await self._storage.get_runtime_setting(_SETTING_KEY)
        parsed: list[CustomEmoji] = []
        if raw:
            try:
                payload = json.loads(raw)
                if isinstance(payload, list):
                    for value in payload:
                        if not isinstance(value, dict):
                            continue
                        item = _validated_custom_emoji(
                            value.get("id"), value.get("alternative")
                        )
                        if item is not None:
                            parsed.append(item)
            except (TypeError, ValueError):
                parsed = []
        self._items = tuple(parsed[-_MAX_THEME_ITEMS:])
        return self._items

    async def _load_service_ids_locked(self) -> dict[str, str]:
        if self._service_ids is not None:
            return self._service_ids
        service_ids = dict(SERVICE_CUSTOM_EMOJI_IDS)
        raw = await self._storage.get_runtime_setting(_SERVICE_SETTING_KEY)
        if raw:
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    for alternative in SERVICE_CUSTOM_EMOJI_IDS:
                        custom_emoji_id = payload.get(alternative)
                        if (
                            isinstance(custom_emoji_id, str)
                            and custom_emoji_id.isdigit()
                        ):
                            service_ids[alternative] = custom_emoji_id
            except (TypeError, ValueError):
                pass
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
        if user is None or user.id not in self._owner_ids:
            return False
        emojis = extract_custom_emojis(message)
        pack_names = extract_emoji_pack_names(message.text or message.caption or "")
        if not pack_names and (not emojis or not _contains_only_custom_emojis(message)):
            return False
        return {
            "custom_emojis": emojis,
            "emoji_pack_names": pack_names,
        }


def extract_emoji_pack_names(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_PACK_LINK_PATTERN.findall(text)))


def apply_service_custom_emojis(
    text: str,
    service_ids: Mapping[str, str] = SERVICE_CUSTOM_EMOJI_IDS,
) -> ThemedText:
    entities: list[MessageEntity] = []
    for alternative, custom_emoji_id in service_ids.items():
        start = 0
        while True:
            index = text.find(alternative, start)
            if index < 0:
                break
            entities.append(
                MessageEntity(
                    type=MessageEntityType.CUSTOM_EMOJI,
                    offset=_utf16_length(text[:index]),
                    length=_utf16_length(alternative),
                    custom_emoji_id=custom_emoji_id,
                )
            )
            start = index + len(alternative)
    entities.sort(key=lambda entity: entity.offset)
    return ThemedText(text=text, entities=entities or None)


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


def _utf16_length(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2
