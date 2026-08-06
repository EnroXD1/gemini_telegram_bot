from __future__ import annotations

import re


def split_text(text: str, limit: int = 4000) -> list[str]:
    """Split Telegram text near natural boundaries without losing characters."""
    if limit < 1:
        raise ValueError("limit must be positive")

    remaining = text.strip()
    if not remaining:
        return []

    chunks: list[str] = []
    while len(remaining) > limit:
        window = remaining[: limit + 1]
        minimum = max(1, int(limit * 0.55))
        cut = -1
        separator_length = 0

        for separator in ("\n\n", "\n", ". ", " "):
            candidate = window.rfind(separator, minimum, limit + 1)
            if candidate > cut:
                cut = candidate
                separator_length = len(separator)

        if cut < minimum:
            cut = limit
            separator_length = 0

        chunk = remaining[: cut + separator_length].strip()
        if not chunk:
            chunk = remaining[:limit]
            cut = limit
            separator_length = 0
        chunks.append(chunk)
        remaining = remaining[cut + separator_length :].lstrip()

    if remaining:
        chunks.append(remaining)
    return chunks


def strip_bot_mention(text: str, username: str | None) -> str:
    if not username:
        return text.strip()
    pattern = re.compile(rf"@{re.escape(username.lstrip('@'))}\b", re.IGNORECASE)
    return pattern.sub("", text).strip(" \t,;:-")


def remove_command(text: str, command: str, username: str | None = None) -> str:
    bot_suffix = rf"(?:@{re.escape(username.lstrip('@'))})?" if username else r"(?:@\w+)?"
    pattern = re.compile(
        rf"^/{re.escape(command.lstrip('/'))}{bot_suffix}(?:\s+|$)", re.IGNORECASE
    )
    return pattern.sub("", text, count=1).strip()
