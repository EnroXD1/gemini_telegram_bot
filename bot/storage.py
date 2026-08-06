from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from .models import ConversationMessage


@dataclass(frozen=True, slots=True)
class BusinessConnectionRecord:
    connection_id: str
    owner_user_id: int
    owner_chat_id: int
    is_enabled: bool


@dataclass(frozen=True, slots=True)
class BusinessMessageRecord:
    connection_id: str
    chat_id: int
    message_id: int
    sender_user_id: int
    sender_name: str
    is_incoming: bool
    content: str
    media_kind: str | None = None
    deleted_at: int | None = None


class Storage:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                scope_key TEXT PRIMARY KEY,
                previous_interaction_id TEXT,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_settings (
                chat_id INTEGER PRIMARY KEY,
                group_mode TEXT NOT NULL CHECK(group_mode IN ('mentions', 'all', 'off')),
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_exchanges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope_key TEXT NOT NULL,
                user_content TEXT NOT NULL,
                assistant_content TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_conversation_exchanges_scope
            ON conversation_exchanges(scope_key, id DESC);

            CREATE TABLE IF NOT EXISTS business_connections (
                connection_id TEXT PRIMARY KEY,
                owner_user_id INTEGER NOT NULL,
                owner_chat_id INTEGER NOT NULL,
                is_enabled INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS business_messages (
                connection_id TEXT NOT NULL,
                chat_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                sender_user_id INTEGER NOT NULL,
                sender_name TEXT NOT NULL,
                is_incoming INTEGER NOT NULL,
                content TEXT NOT NULL,
                media_kind TEXT,
                deleted_at INTEGER,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY(connection_id, chat_id, message_id)
            );

            CREATE INDEX IF NOT EXISTS idx_business_messages_updated_at
            ON business_messages(updated_at);

            CREATE TABLE IF NOT EXISTS business_greetings (
                connection_id TEXT NOT NULL,
                chat_id INTEGER NOT NULL,
                greeted_at INTEGER NOT NULL,
                PRIMARY KEY(connection_id, chat_id)
            );
            """
        )
        await self._db.commit()

    def _connection(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Storage is not open")
        return self._db

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def get_interaction_id(self, scope_key: str) -> str | None:
        db = self._connection()
        async with self._lock:
            cursor = await db.execute(
                "SELECT previous_interaction_id FROM conversations WHERE scope_key = ?",
                (scope_key,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return None if row is None else row["previous_interaction_id"]

    async def set_interaction_id(
        self, scope_key: str, interaction_id: str | None
    ) -> None:
        db = self._connection()
        async with self._lock:
            await db.execute(
                """
                INSERT INTO conversations(scope_key, previous_interaction_id, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(scope_key) DO UPDATE SET
                    previous_interaction_id = excluded.previous_interaction_id,
                    updated_at = excluded.updated_at
                """,
                (scope_key, interaction_id, int(time.time())),
            )
            await db.commit()

    async def reset_conversation(self, scope_key: str) -> None:
        db = self._connection()
        async with self._lock:
            await db.execute("DELETE FROM conversations WHERE scope_key = ?", (scope_key,))
            await db.execute(
                "DELETE FROM conversation_exchanges WHERE scope_key = ?", (scope_key,)
            )
            await db.commit()

    async def get_conversation_history(
        self, *, scope_key: str, max_exchanges: int, max_chars: int
    ) -> tuple[ConversationMessage, ...]:
        if max_exchanges <= 0 or max_chars <= 0:
            return ()
        db = self._connection()
        async with self._lock:
            cursor = await db.execute(
                """
                SELECT user_content, assistant_content
                FROM conversation_exchanges
                WHERE scope_key = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (scope_key, max_exchanges),
            )
            rows = await cursor.fetchall()
            await cursor.close()

        selected: list[tuple[str, str]] = []
        used_chars = 0
        for row in rows:
            user_content = str(row["user_content"])
            assistant_content = str(row["assistant_content"])
            pair_chars = len(user_content) + len(assistant_content)
            if used_chars + pair_chars > max_chars:
                if selected:
                    break
                half = max(1, max_chars // 2)
                selected.append(
                    (
                        _trim_history_text(user_content, half),
                        _trim_history_text(assistant_content, max_chars - half),
                    )
                )
                break
            selected.append((user_content, assistant_content))
            used_chars += pair_chars

        messages: list[ConversationMessage] = []
        for user_content, assistant_content in reversed(selected):
            messages.append(ConversationMessage(role="user", content=user_content))
            messages.append(
                ConversationMessage(role="assistant", content=assistant_content)
            )
        return tuple(messages)

    async def append_conversation_exchange(
        self,
        *,
        scope_key: str,
        user_content: str,
        assistant_content: str,
        max_exchanges: int,
    ) -> None:
        if max_exchanges <= 0:
            return
        db = self._connection()
        async with self._lock:
            await db.execute(
                """
                INSERT INTO conversation_exchanges(
                    scope_key, user_content, assistant_content, created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (scope_key, user_content, assistant_content, int(time.time())),
            )
            await db.execute(
                """
                DELETE FROM conversation_exchanges
                WHERE scope_key = ? AND id NOT IN (
                    SELECT id FROM conversation_exchanges
                    WHERE scope_key = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                """,
                (scope_key, scope_key, max_exchanges),
            )
            await db.commit()

    async def has_conversation_history(self, scope_key: str) -> bool:
        db = self._connection()
        async with self._lock:
            cursor = await db.execute(
                "SELECT 1 FROM conversation_exchanges WHERE scope_key = ? LIMIT 1",
                (scope_key,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return row is not None

    async def prune_conversation_history(self, retention_days: int) -> int:
        db = self._connection()
        cutoff = int(time.time()) - retention_days * 86_400
        async with self._lock:
            cursor = await db.execute(
                "DELETE FROM conversation_exchanges WHERE created_at < ?", (cutoff,)
            )
            removed = max(0, cursor.rowcount)
            await cursor.close()
            await db.commit()
        return removed

    async def get_group_mode(self, chat_id: int, default: str) -> str:
        db = self._connection()
        async with self._lock:
            cursor = await db.execute(
                "SELECT group_mode FROM chat_settings WHERE chat_id = ?", (chat_id,)
            )
            row = await cursor.fetchone()
            await cursor.close()
        return default if row is None else str(row["group_mode"])

    async def set_group_mode(self, chat_id: int, mode: str) -> None:
        if mode not in {"mentions", "all", "off"}:
            raise ValueError(f"Unsupported group mode: {mode}")
        db = self._connection()
        async with self._lock:
            await db.execute(
                """
                INSERT INTO chat_settings(chat_id, group_mode, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    group_mode = excluded.group_mode,
                    updated_at = excluded.updated_at
                """,
                (chat_id, mode, int(time.time())),
            )
            await db.commit()

    async def save_business_connection(
        self,
        *,
        connection_id: str,
        owner_user_id: int,
        owner_chat_id: int,
        is_enabled: bool,
    ) -> None:
        db = self._connection()
        async with self._lock:
            await db.execute(
                """
                INSERT INTO business_connections(
                    connection_id, owner_user_id, owner_chat_id, is_enabled, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(connection_id) DO UPDATE SET
                    owner_user_id = excluded.owner_user_id,
                    owner_chat_id = excluded.owner_chat_id,
                    is_enabled = excluded.is_enabled,
                    updated_at = excluded.updated_at
                """,
                (
                    connection_id,
                    owner_user_id,
                    owner_chat_id,
                    int(is_enabled),
                    int(time.time()),
                ),
            )
            await db.commit()

    async def claim_business_greeting(
        self, *, connection_id: str, chat_id: int
    ) -> bool:
        db = self._connection()
        async with self._lock:
            cursor = await db.execute(
                """
                INSERT OR IGNORE INTO business_greetings(
                    connection_id, chat_id, greeted_at
                )
                VALUES (?, ?, ?)
                """,
                (connection_id, chat_id, int(time.time())),
            )
            claimed = cursor.rowcount == 1
            await cursor.close()
            await db.commit()
        return claimed

    async def release_business_greeting(
        self, *, connection_id: str, chat_id: int
    ) -> None:
        db = self._connection()
        async with self._lock:
            await db.execute(
                "DELETE FROM business_greetings WHERE connection_id = ? AND chat_id = ?",
                (connection_id, chat_id),
            )
            await db.commit()

    async def get_business_connection(
        self, connection_id: str
    ) -> BusinessConnectionRecord | None:
        db = self._connection()
        async with self._lock:
            cursor = await db.execute(
                """
                SELECT connection_id, owner_user_id, owner_chat_id, is_enabled
                FROM business_connections
                WHERE connection_id = ?
                """,
                (connection_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        if row is None:
            return None
        return _business_connection_from_row(row)

    async def upsert_business_message(
        self, record: BusinessMessageRecord
    ) -> BusinessMessageRecord | None:
        """Store the newest version and return the version it replaced, if any."""
        db = self._connection()
        now = int(time.time())
        async with self._lock:
            cursor = await db.execute(
                """
                SELECT connection_id, chat_id, message_id, sender_user_id,
                       sender_name, is_incoming, content, media_kind, deleted_at
                FROM business_messages
                WHERE connection_id = ? AND chat_id = ? AND message_id = ?
                """,
                (record.connection_id, record.chat_id, record.message_id),
            )
            row = await cursor.fetchone()
            await cursor.close()
            previous = None if row is None else _business_message_from_row(row)
            await db.execute(
                """
                INSERT INTO business_messages(
                    connection_id, chat_id, message_id, sender_user_id, sender_name,
                    is_incoming, content, media_kind, deleted_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(connection_id, chat_id, message_id) DO UPDATE SET
                    sender_user_id = excluded.sender_user_id,
                    sender_name = excluded.sender_name,
                    is_incoming = excluded.is_incoming,
                    content = excluded.content,
                    media_kind = excluded.media_kind,
                    updated_at = excluded.updated_at
                """,
                (
                    record.connection_id,
                    record.chat_id,
                    record.message_id,
                    record.sender_user_id,
                    record.sender_name,
                    int(record.is_incoming),
                    record.content,
                    record.media_kind,
                    record.deleted_at,
                    now,
                    now,
                ),
            )
            await db.commit()
        return previous

    async def get_business_messages(
        self,
        *,
        connection_id: str,
        chat_id: int,
        message_ids: list[int],
    ) -> list[BusinessMessageRecord]:
        if not message_ids:
            return []
        db = self._connection()
        placeholders = ", ".join("?" for _ in message_ids)
        query = f"""
            SELECT connection_id, chat_id, message_id, sender_user_id,
                   sender_name, is_incoming, content, media_kind, deleted_at
            FROM business_messages
            WHERE connection_id = ? AND chat_id = ?
              AND message_id IN ({placeholders})
        """
        parameters = (connection_id, chat_id, *message_ids)
        async with self._lock:
            cursor = await db.execute(query, parameters)
            rows = await cursor.fetchall()
            await cursor.close()
        records = [_business_message_from_row(row) for row in rows]
        order = {message_id: index for index, message_id in enumerate(message_ids)}
        return sorted(records, key=lambda item: order.get(item.message_id, len(order)))

    async def mark_business_message_deleted(
        self, *, connection_id: str, chat_id: int, message_id: int
    ) -> None:
        db = self._connection()
        now = int(time.time())
        async with self._lock:
            await db.execute(
                """
                UPDATE business_messages
                SET deleted_at = ?, updated_at = ?
                WHERE connection_id = ? AND chat_id = ? AND message_id = ?
                """,
                (now, now, connection_id, chat_id, message_id),
            )
            await db.commit()

    async def prune_business_messages(self, retention_days: int) -> int:
        db = self._connection()
        cutoff = int(time.time()) - retention_days * 86_400
        async with self._lock:
            cursor = await db.execute(
                "DELETE FROM business_messages WHERE updated_at < ?", (cutoff,)
            )
            removed = max(0, cursor.rowcount)
            await cursor.close()
            await db.commit()
        return removed


def _business_connection_from_row(row: aiosqlite.Row) -> BusinessConnectionRecord:
    return BusinessConnectionRecord(
        connection_id=str(row["connection_id"]),
        owner_user_id=int(row["owner_user_id"]),
        owner_chat_id=int(row["owner_chat_id"]),
        is_enabled=bool(row["is_enabled"]),
    )


def _business_message_from_row(row: aiosqlite.Row) -> BusinessMessageRecord:
    deleted_at = row["deleted_at"]
    return BusinessMessageRecord(
        connection_id=str(row["connection_id"]),
        chat_id=int(row["chat_id"]),
        message_id=int(row["message_id"]),
        sender_user_id=int(row["sender_user_id"]),
        sender_name=str(row["sender_name"]),
        is_incoming=bool(row["is_incoming"]),
        content=str(row["content"]),
        media_kind=None if row["media_kind"] is None else str(row["media_kind"]),
        deleted_at=None if deleted_at is None else int(deleted_at),
    )


def _trim_history_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n[…история сокращена…]\n"
    if limit <= len(marker) + 2:
        return value[-limit:]
    head = max(1, (limit - len(marker)) // 3)
    tail = limit - len(marker) - head
    return value[:head] + marker + value[-tail:]
