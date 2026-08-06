from __future__ import annotations

import pytest

from bot.config import Settings

TOKEN_NAMES = (
    "TELEGRAM_BOT_TOKEN",
    "APP_TELEGRAM_BOT_TOKEN",
    "BOT_TOKEN",
    "BOT_API_TOKEN",
    "TOKEN",
)


@pytest.mark.parametrize("alias", TOKEN_NAMES)
def test_settings_accept_telegram_token_aliases(monkeypatch, tmp_path, alias: str) -> None:
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(alias, "123456789:test-token")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    settings = Settings.from_env(tmp_path / "missing.env")

    assert settings.telegram_bot_token == "123456789:test-token"


def test_settings_ignore_non_telegram_host_token(monkeypatch, tmp_path) -> None:
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BOT_TOKEN", "bothost-internal-token")
    monkeypatch.setenv("TOKEN", "987654321:telegram-token")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    settings = Settings.from_env(tmp_path / "missing.env")

    assert settings.telegram_bot_token == "987654321:telegram-token"
