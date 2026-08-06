from __future__ import annotations

from bot.markdown import render_markdown_chunks


def _utf16_length(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def test_render_markdown_creates_telegram_entities() -> None:
    chunks = render_markdown_chunks(
        "# Заголовок\n\n**важно**, *курсив*, `код` и [ссылка](https://example.com)",
        4096,
    )

    assert len(chunks) == 1
    entity_types = {str(entity.type) for entity in chunks[0].entities}
    assert "bold" in entity_types
    assert "italic" in entity_types
    assert "code" in entity_types
    assert "text_link" in entity_types
    assert "**" not in chunks[0].text


def test_render_markdown_splits_by_telegram_utf16_limit() -> None:
    chunks = render_markdown_chunks(
        "**" + ("🙂 слово " * 40).strip() + "**", 50
    )

    assert len(chunks) > 1
    assert all(0 < _utf16_length(chunk.text) <= 50 for chunk in chunks)
    assert all(any(str(entity.type) == "bold" for entity in chunk.entities) for chunk in chunks)
