from __future__ import annotations


def build_scope_key(
    *,
    chat_id: int,
    thread_id: int | None,
    user_id: int | None,
    chat_type: str,
    scope_mode: str,
    business_connection_id: str | None = None,
) -> str:
    """Build a stable conversation key without exposing message content."""
    thread = thread_id or 0
    business = f":business:{business_connection_id}" if business_connection_id else ""
    base = f"chat:{chat_id}{business}:thread:{thread}"

    if chat_type == "private" or scope_mode == "chat_thread_user":
        return f"{base}:user:{user_id or 0}"
    return base
