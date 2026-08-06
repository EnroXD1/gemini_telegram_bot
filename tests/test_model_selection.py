from pathlib import Path

from bot.storage import Storage


async def test_ai_selection_is_persisted(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "bot.sqlite3")
    await storage.open()
    try:
        assert await storage.get_ai_selection() is None

        await storage.set_ai_selection("groq", "llama-3.1-8b-instant")

        assert await storage.get_ai_selection() == (
            "groq",
            "llama-3.1-8b-instant",
        )
    finally:
        await storage.close()

