from __future__ import annotations

from typing import Any


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def extract_interaction_text(interaction: Any) -> str:
    """Read text from current and older Interaction response shapes."""
    try:
        convenience = _value(interaction, "output_text")
    except (AttributeError, TypeError, ValueError):
        convenience = None
    if isinstance(convenience, str) and convenience.strip():
        return convenience.strip()

    outputs = _value(interaction, "outputs", []) or []
    output_parts = [
        str(_value(item, "text")).strip()
        for item in outputs
        if _value(item, "type") == "text" and _value(item, "text")
    ]
    if output_parts:
        return "\n".join(output_parts).strip()

    steps = _value(interaction, "steps", []) or []
    for step in reversed(steps):
        if _value(step, "type") != "model_output":
            continue
        content = _value(step, "content", []) or []
        parts = [
            str(_value(item, "text")).strip()
            for item in content
            if _value(item, "type") == "text" and _value(item, "text")
        ]
        if parts:
            return "\n".join(parts).strip()
    return ""


def extract_interaction_id(interaction: Any) -> str | None:
    value = _value(interaction, "id")
    return str(value) if value else None
