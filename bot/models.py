from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MediaPayload:
    label: str
    kind: str
    mime_type: str
    data: bytes


@dataclass(frozen=True, slots=True)
class PromptBundle:
    prompt: str
    media: tuple[MediaPayload, ...] = ()


@dataclass(frozen=True, slots=True)
class GeminiResult:
    text: str
    interaction_id: str | None
    context_was_reset: bool = False
