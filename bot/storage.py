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


@dataclass(frozen=True, slots=True)
class BusinessMediaArchiveRecord:
    connection_id: str
    chat_id: int
    message_id: int
    kind: str
    file_name: str
    file_path: str
    file_size: int
    created_at: int


@dataclass(frozen=True, slots=True)
class BusinessChatRecord:
    chat_id: int
    sender_name: str
    updated_at: int
    auto_reply_enabled: bool


@dataclass(frozen=True, slots=True)
class BotUserRecord:
    user_id: int
    username: str | None
    display_name: str
    first_seen_at: int
    last_seen_at: int
    last_chat_id: int
    last_chat_type: str
    last_chat_title: str | None
    interaction_count: int
    ai_request_count: int
    openrouter_request_count: int
    groq_request_count: int
    google_request_count: int


@dataclass(frozen=True, slots=True)
class BotUsageStats:
    total_users: int
    active_24h: int
    active_7d: int
    active_30d: int
    private_users: int
    group_users: int
    interaction_count: int
    ai_request_count: int
    openrouter_request_count: int
    groq_request_count: int
    google_request_count: int


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

            CREATE TABLE IF NOT EXISTS business_media_archives (
                connection_id TEXT NOT NULL,
                chat_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY(connection_id, chat_id, message_id)
            );

            CREATE INDEX IF NOT EXISTS idx_business_media_archives_updated_at
            ON business_media_archives(updated_at);

            CREATE TABLE IF NOT EXISTS business_greetings (
                connection_id TEXT NOT NULL,
                chat_id INTEGER NOT NULL,
                greeted_at INTEGER NOT NULL,
                PRIMARY KEY(connection_id, chat_id)
            );

            CREATE TABLE IF NOT EXISTS business_owner_settings (
                owner_user_id INTEGER PRIMARY KEY,
                auto_reply_enabled INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS business_chat_settings (
                owner_user_id INTEGER NOT NULL,
                chat_id INTEGER NOT NULL,
                auto_reply_enabled INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY(owner_user_id, chat_id)
            );

            CREATE TABLE IF NOT EXISTS bot_users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                display_name TEXT NOT NULL,
                first_seen_at INTEGER NOT NULL,
                last_seen_at INTEGER NOT NULL,
                last_chat_id INTEGER NOT NULL,
                last_chat_type TEXT NOT NULL,
                last_chat_title TEXT,
                interaction_count INTEGER NOT NULL,
                ai_request_count INTEGER NOT NULL,
                openrouter_request_count INTEGER NOT NULL,
                groq_request_count INTEGER NOT NULL DEFAULT 0,
                google_request_count INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_bot_users_last_seen
            ON bot_users(last_seen_at DESC);

            CREATE TABLE IF NOT EXISTS runtime_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            );
            """
        )
        cursor = await self._db.execute("PRAGMA table_info(bot_users)")
        bot_user_columns = {str(row[1]) for row in await cursor.fetchall()}
        await cursor.close()
        if "groq_request_count" not in bot_user_columns:
            await self._db.execute(
                "ALTER TABLE bot_users ADD COLUMN "
                "groq_request_count INTEGER NOT NULL DEFAULT 0"
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

    async def get_ai_selection(self) -> tuple[str, str] | None:
        db = self._connection()
        async with self._lock:
            cursor = await db.execute(
                """
                SELECT setting_key, setting_value
                FROM runtime_settings
                WHERE setting_key IN ('ai_provider', 'ai_model')
                """
            )
            values = {
                str(row["setting_key"]): str(row["setting_value"])
                for row in await cursor.fetchall()
            }
            await cursor.close()
        provider = values.get("ai_provider")
        model = values.get("ai_model")
        if not provider or not model:
            return None
        return provider, model

    async def set_ai_selection(self, provider: str, model: str) -> None:
        db = self._connection()
        now = int(time.time())
        async with self._lock:
            await db.executemany(
                """
                INSERT INTO runtime_settings(setting_key, setting_value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(setting_key) DO UPDATE SET
                    setting_value = excluded.setting_value,
                    updated_at = excluded.updated_at
                """,
                (
                    ("ai_provider", provider, now),
                    ("ai_model", model, now),
                ),
            )
            await db.commit()

    async def clear_conversation_contexts(self) -> None:
        """Reset provider-specific context after a global model switch."""
        db = self._connection()
        async with self._lock:
            await db.execute("DELETE FROM conversations")
            await db.execute("DELETE FROM conversation_exchanges")
            await db.commit()

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

    async def delete_bot_users(self, user_ids: frozenset[int]) -> int:
        """Remove configured bot owners from the external-user audit."""
        if not user_ids:
            return 0
        db = self._connection()
        placeholders = ", ".join("?" for _ in user_ids)
        async with self._lock:
            cursor = await db.execute(
                f"DELETE FROM bot_users WHERE user_id IN ({placeholders})",
                tuple(user_ids),
            )
            removed = max(0, cursor.rowcount)
            await cursor.close()
            await db.commit()
        return removed

    async def is_business_owner(self, user_id: int) -> bool:
        db = self._connection()
        async with self._lock:
            cursor = await db.execute(
                "SELECT 1 FROM business_connections WHERE owner_user_id = ? LIMIT 1",
                (user_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return row is not None

    async def list_business_owner_chat_ids(self) -> list[int]:
        db = self._connection()
        async with self._lock:
            cursor = await db.execute(
                """
                SELECT DISTINCT owner_chat_id
                FROM business_connections
                """
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [int(row["owner_chat_id"]) for row in rows]

    async def record_bot_user_activity(
        self,
        *,
        user_id: int,
        username: str | None,
        display_name: str,
        chat_id: int,
        chat_type: str,
        chat_title: str | None,
        ai_provider: str | None,
    ) -> bool:
        """Record one handled interaction and return True for a new user."""
        db = self._connection()
        now = int(time.time())
        ai_increment = int(ai_provider is not None)
        openrouter_increment = int(ai_provider == "openrouter")
        groq_increment = int(ai_provider == "groq")
        google_increment = int(ai_provider == "google")
        async with self._lock:
            cursor = await db.execute(
                "SELECT 1 FROM bot_users WHERE user_id = ?",
                (user_id,),
            )
            is_new = await cursor.fetchone() is None
            await cursor.close()
            await db.execute(
                """
                INSERT INTO bot_users(
                    user_id, username, display_name, first_seen_at, last_seen_at,
                    last_chat_id, last_chat_type, last_chat_title,
                    interaction_count, ai_request_count,
                    openrouter_request_count, groq_request_count,
                    google_request_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    username = excluded.username,
                    display_name = excluded.display_name,
                    last_seen_at = excluded.last_seen_at,
                    last_chat_id = excluded.last_chat_id,
                    last_chat_type = excluded.last_chat_type,
                    last_chat_title = excluded.last_chat_title,
                    interaction_count = bot_users.interaction_count + 1,
                    ai_request_count = bot_users.ai_request_count + excluded.ai_request_count,
                    openrouter_request_count = (
                        bot_users.openrouter_request_count
                        + excluded.openrouter_request_count
                    ),
                    groq_request_count = (
                        bot_users.groq_request_count
                        + excluded.groq_request_count
                    ),
                    google_request_count = (
                        bot_users.google_request_count
                        + excluded.google_request_count
                    )
                """,
                (
                    user_id,
                    username,
                    display_name,
                    now,
                    now,
                    chat_id,
                    chat_type,
                    chat_title,
                    ai_increment,
                    openrouter_increment,
                    groq_increment,
                    google_increment,
                ),
            )
            await db.commit()
        return is_new

    async def list_bot_users(self, limit: int = 15) -> list[BotUserRecord]:
        if limit <= 0:
            return []
        db = self._connection()
        async with self._lock:
            cursor = await db.execute(
                """
                SELECT
                    user_id, username, display_name, first_seen_at, last_seen_at,
                    last_chat_id, last_chat_type, last_chat_title,
                    interaction_count, ai_request_count,
                    openrouter_request_count, groq_request_count,
                    google_request_count
                FROM bot_users
                ORDER BY last_seen_at DESC, user_id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [_bot_user_from_row(row) for row in rows]

    async def get_bot_usage_stats(self) -> BotUsageStats:
        db = self._connection()
        now = int(time.time())
        async with self._lock:
            cursor = await db.execute(
                """
                SELECT
                    COUNT(*) AS total_users,
                    COALESCE(SUM(last_seen_at >= ?), 0) AS active_24h,
                    COALESCE(SUM(last_seen_at >= ?), 0) AS active_7d,
                    COALESCE(SUM(last_seen_at >= ?), 0) AS active_30d,
                    COALESCE(SUM(last_chat_type = 'private'), 0) AS private_users,
                    COALESCE(SUM(last_chat_type IN ('group', 'supergroup')), 0)
                        AS group_users,
                    COALESCE(SUM(interaction_count), 0) AS interaction_count,
                    COALESCE(SUM(ai_request_count), 0) AS ai_request_count,
                    COALESCE(SUM(openrouter_request_count), 0)
                        AS openrouter_request_count,
                    COALESCE(SUM(groq_request_count), 0) AS groq_request_count,
                    COALESCE(SUM(google_request_count), 0) AS google_request_count
                FROM bot_users
                """,
                (now - 86_400, now - 7 * 86_400, now - 30 * 86_400),
            )
            row = await cursor.fetchone()
            await cursor.close()
        assert row is not None
        return BotUsageStats(
            total_users=int(row["total_users"]),
            active_24h=int(row["active_24h"]),
            active_7d=int(row["active_7d"]),
            active_30d=int(row["active_30d"]),
            private_users=int(row["private_users"]),
            group_users=int(row["group_users"]),
            interaction_count=int(row["interaction_count"]),
            ai_request_count=int(row["ai_request_count"]),
            openrouter_request_count=int(row["openrouter_request_count"]),
            groq_request_count=int(row["groq_request_count"]),
            google_request_count=int(row["google_request_count"]),
        )

    async def get_business_auto_reply_enabled(
        self, owner_user_id: int, default: bool
    ) -> bool:
        db = self._connection()
        async with self._lock:
            cursor = await db.execute(
                """
                SELECT auto_reply_enabled
                FROM business_owner_settings
                WHERE owner_user_id = ?
                """,
                (owner_user_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return default if row is None else bool(row["auto_reply_enabled"])

    async def set_business_auto_reply_enabled(
        self, owner_user_id: int, enabled: bool
    ) -> None:
        db = self._connection()
        async with self._lock:
            await db.execute(
                """
                INSERT INTO business_owner_settings(
                    owner_user_id, auto_reply_enabled, updated_at
                )
                VALUES (?, ?, ?)
                ON CONFLICT(owner_user_id) DO UPDATE SET
                    auto_reply_enabled = excluded.auto_reply_enabled,
                    updated_at = excluded.updated_at
                """,
                (owner_user_id, int(enabled), int(time.time())),
            )
            await db.commit()

    async def get_business_chat_auto_reply_enabled(
        self, owner_user_id: int, chat_id: int, default: bool = True
    ) -> bool:
        db = self._connection()
        async with self._lock:
            cursor = await db.execute(
                """
                SELECT auto_reply_enabled
                FROM business_chat_settings
                WHERE owner_user_id = ? AND chat_id = ?
                """,
                (owner_user_id, chat_id),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return default if row is None else bool(row["auto_reply_enabled"])

    async def set_business_chat_auto_reply_enabled(
        self, owner_user_id: int, chat_id: int, enabled: bool
    ) -> None:
        db = self._connection()
        async with self._lock:
            await db.execute(
                """
                INSERT INTO business_chat_settings(
                    owner_user_id, chat_id, auto_reply_enabled, updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(owner_user_id, chat_id) DO UPDATE SET
                    auto_reply_enabled = excluded.auto_reply_enabled,
                    updated_at = excluded.updated_at
                """,
                (owner_user_id, chat_id, int(enabled), int(time.time())),
            )
            await db.commit()

    async def get_effective_business_auto_reply_enabled(
        self,
        *,
        owner_user_id: int,
        chat_id: int,
        global_default: bool,
    ) -> bool:
        owner_enabled = await self.get_business_auto_reply_enabled(
            owner_user_id, global_default
        )
        if not owner_enabled:
            return False
        return await self.get_business_chat_auto_reply_enabled(
            owner_user_id, chat_id
        )

    async def is_known_business_chat(self, owner_user_id: int, chat_id: int) -> bool:
        db = self._connection()
        async with self._lock:
            cursor = await db.execute(
                """
                SELECT 1
                FROM business_messages AS message
                JOIN business_connections AS connection
                  ON connection.connection_id = message.connection_id
                WHERE connection.owner_user_id = ?
                  AND message.chat_id = ?
                  AND message.is_incoming = 1
                LIMIT 1
                """,
                (owner_user_id, chat_id),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return row is not None

    async def list_business_chats(
        self, owner_user_id: int, limit: int = 20
    ) -> list[BusinessChatRecord]:
        if limit <= 0:
            return []
        db = self._connection()
        async with self._lock:
            cursor = await db.execute(
                """
                WITH recent AS (
                    SELECT
                        message.chat_id,
                        message.sender_name,
                        message.updated_at,
                        ROW_NUMBER() OVER (
                            PARTITION BY message.chat_id
                            ORDER BY message.updated_at DESC, message.message_id DESC
                        ) AS position
                    FROM business_messages AS message
                    JOIN business_connections AS connection
                      ON connection.connection_id = message.connection_id
                    WHERE connection.owner_user_id = ?
                      AND message.is_incoming = 1
                )
                SELECT
                    recent.chat_id,
                    recent.sender_name,
                    recent.updated_at,
                    COALESCE(setting.auto_reply_enabled, 1) AS auto_reply_enabled
                FROM recent
                LEFT JOIN business_chat_settings AS setting
                  ON setting.owner_user_id = ?
                 AND setting.chat_id = recent.chat_id
                WHERE recent.position = 1
                ORDER BY recent.updated_at DESC
                LIMIT ?
                """,
                (owner_user_id, owner_user_id, limit),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [
            BusinessChatRecord(
                chat_id=int(row["chat_id"]),
                sender_name=str(row["sender_name"]),
                updated_at=int(row["updated_at"]),
                auto_reply_enabled=bool(row["auto_reply_enabled"]),
            )
            for row in rows
        ]

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

    async def get_business_media_archive(
        self, *, connection_id: str, chat_id: int, message_id: int
    ) -> BusinessMediaArchiveRecord | None:
        db = self._connection()
        async with self._lock:
            cursor = await db.execute(
                """
                SELECT connection_id, chat_id, message_id, kind, file_name,
                       file_path, file_size, created_at
                FROM business_media_archives
                WHERE connection_id = ? AND chat_id = ? AND message_id = ?
                """,
                (connection_id, chat_id, message_id),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return None if row is None else _business_media_archive_from_row(row)

    async def get_business_media_archives(
        self,
        *,
        connection_id: str,
        chat_id: int,
        message_ids: list[int],
    ) -> list[BusinessMediaArchiveRecord]:
        if not message_ids:
            return []
        db = self._connection()
        placeholders = ", ".join("?" for _ in message_ids)
        query = f"""
            SELECT connection_id, chat_id, message_id, kind, file_name,
                   file_path, file_size, created_at
            FROM business_media_archives
            WHERE connection_id = ? AND chat_id = ?
              AND message_id IN ({placeholders})
        """
        parameters = (connection_id, chat_id, *message_ids)
        async with self._lock:
            cursor = await db.execute(query, parameters)
            rows = await cursor.fetchall()
            await cursor.close()
        archives = [_business_media_archive_from_row(row) for row in rows]
        order = {message_id: index for index, message_id in enumerate(message_ids)}
        return sorted(archives, key=lambda item: order.get(item.message_id, len(order)))

    async def upsert_business_media_archive(
        self, record: BusinessMediaArchiveRecord
    ) -> BusinessMediaArchiveRecord | None:
        db = self._connection()
        now = int(time.time())
        async with self._lock:
            cursor = await db.execute(
                """
                SELECT connection_id, chat_id, message_id, kind, file_name,
                       file_path, file_size, created_at
                FROM business_media_archives
                WHERE connection_id = ? AND chat_id = ? AND message_id = ?
                """,
                (record.connection_id, record.chat_id, record.message_id),
            )
            row = await cursor.fetchone()
            await cursor.close()
            previous = None if row is None else _business_media_archive_from_row(row)
            await db.execute(
                """
                INSERT INTO business_media_archives(
                    connection_id, chat_id, message_id, kind, file_name,
                    file_path, file_size, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(connection_id, chat_id, message_id) DO UPDATE SET
                    kind = excluded.kind,
                    file_name = excluded.file_name,
                    file_path = excluded.file_path,
                    file_size = excluded.file_size,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at
                """,
                (
                    record.connection_id,
                    record.chat_id,
                    record.message_id,
                    record.kind,
                    record.file_name,
                    record.file_path,
                    record.file_size,
                    record.created_at,
                    now,
                ),
            )
            await db.commit()
        return previous

    async def delete_business_media_archive(
        self, *, connection_id: str, chat_id: int, message_id: int
    ) -> None:
        db = self._connection()
        async with self._lock:
            await db.execute(
                """
                DELETE FROM business_media_archives
                WHERE connection_id = ? AND chat_id = ? AND message_id = ?
                """,
                (connection_id, chat_id, message_id),
            )
            await db.commit()

    async def pop_expired_business_media_archives(
        self, retention_days: int
    ) -> list[BusinessMediaArchiveRecord]:
        db = self._connection()
        cutoff = int(time.time()) - retention_days * 86_400
        async with self._lock:
            cursor = await db.execute(
                """
                SELECT connection_id, chat_id, message_id, kind, file_name,
                       file_path, file_size, created_at
                FROM business_media_archives
                WHERE updated_at < ?
                """,
                (cutoff,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            await db.execute(
                "DELETE FROM business_media_archives WHERE updated_at < ?",
                (cutoff,),
            )
            await db.commit()
        return [_business_media_archive_from_row(row) for row in rows]

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


def _business_media_archive_from_row(
    row: aiosqlite.Row,
) -> BusinessMediaArchiveRecord:
    return BusinessMediaArchiveRecord(
        connection_id=str(row["connection_id"]),
        chat_id=int(row["chat_id"]),
        message_id=int(row["message_id"]),
        kind=str(row["kind"]),
        file_name=str(row["file_name"]),
        file_path=str(row["file_path"]),
        file_size=int(row["file_size"]),
        created_at=int(row["created_at"]),
    )


def _bot_user_from_row(row: aiosqlite.Row) -> BotUserRecord:
    return BotUserRecord(
        user_id=int(row["user_id"]),
        username=None if row["username"] is None else str(row["username"]),
        display_name=str(row["display_name"]),
        first_seen_at=int(row["first_seen_at"]),
        last_seen_at=int(row["last_seen_at"]),
        last_chat_id=int(row["last_chat_id"]),
        last_chat_type=str(row["last_chat_type"]),
        last_chat_title=(
            None if row["last_chat_title"] is None else str(row["last_chat_title"])
        ),
        interaction_count=int(row["interaction_count"]),
        ai_request_count=int(row["ai_request_count"]),
        openrouter_request_count=int(row["openrouter_request_count"]),
        groq_request_count=int(row["groq_request_count"]),
        google_request_count=int(row["google_request_count"]),
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
