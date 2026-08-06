from __future__ import annotations

import asyncio
import time
from pathlib import Path

import aiosqlite


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
            await db.commit()

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
