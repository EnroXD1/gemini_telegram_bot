from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from aiogram.types import MessageEntity
from telegramify_markdown import convert, split_entities

from .text_utils import split_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FormattedChunk:
    text: str
    entities: tuple[MessageEntity, ...] = ()


def render_markdown_chunks(markdown: str, limit: int) -> list[FormattedChunk]:
    """Convert regular Markdown into Telegram entities and safely split it."""
    if limit < 1:
        raise ValueError("limit must be positive")

    source = markdown.strip()
    if not source:
        return []

    try:
        text, entities = convert(source)
        chunks = split_entities(text, entities, max_utf16_len=limit)
        return [
            FormattedChunk(
                text=chunk_text,
                entities=tuple(
                    MessageEntity(**asdict(entity)) for entity in chunk_entities
                ),
            )
            for chunk_text, chunk_entities in chunks
        ]
    except Exception as exc:
        logger.warning(
            "Could not render Markdown; sending plain text type=%s",
            type(exc).__name__,
        )
        return [FormattedChunk(text=chunk) for chunk in split_text(source, limit)]
