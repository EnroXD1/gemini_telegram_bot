from __future__ import annotations

import pytest

from bot.config import ConfigError, Settings

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


def test_settings_enable_vertex_ai(monkeypatch, tmp_path) -> None:
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_VERTEX_AI", "true")

    settings = Settings.from_env(tmp_path / "missing.env")

    assert settings.gemini_vertex_ai is True


def test_openrouter_provider_does_not_require_google_key(monkeypatch, tmp_path) -> None:
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    settings = Settings.from_env(tmp_path / "missing.env")

    assert settings.ai_provider == "openrouter"
    assert settings.gemini_api_key == ""
    assert settings.active_model == "google/gemini-3.5-flash"


def test_openrouter_provider_requires_its_key(monkeypatch, tmp_path) -> None:
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        Settings.from_env(tmp_path / "missing.env")


def test_business_auto_reply_can_default_to_monitoring_only(
    monkeypatch, tmp_path
) -> None:
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("BUSINESS_AUTO_REPLY_ENABLED", "false")

    settings = Settings.from_env(tmp_path / "missing.env")

    assert settings.business_auto_reply_enabled is False
