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
    assert settings.active_model == "openrouter/free"


def test_openrouter_provider_requires_its_key(monkeypatch, tmp_path) -> None:
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        Settings.from_env(tmp_path / "missing.env")


def test_groq_fallback_is_enabled_when_key_is_present(monkeypatch, tmp_path) -> None:
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    settings = Settings.from_env(tmp_path / "missing.env")

    assert settings.groq_fallback_ready is True
    assert settings.groq_model == "openai/gpt-oss-20b"
    assert settings.groq_fallback_models == (
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
    )
    assert settings.groq_max_output_tokens == 1024
    assert settings.groq_vision_model == "qwen/qwen3.8-27b"


@pytest.mark.parametrize("model", ["", "qwen/custom-vision"])
def test_groq_vision_model_can_be_overridden_or_disabled(monkeypatch, tmp_path, model):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:test-token")
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_VISION_MODEL", model)
    assert Settings.from_env(tmp_path / "missing.env").groq_vision_model == model


@pytest.mark.parametrize("model", ["qwen/invalid model", "qwen/invalid\nmodel", "x" * 201])
def test_invalid_groq_vision_model_fails_at_startup(monkeypatch, tmp_path, model):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:test-token")
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_VISION_MODEL", model)
    with pytest.raises(ConfigError, match="GROQ_VISION_MODEL"):
        Settings.from_env(tmp_path / "missing.env")


def test_free_model_pool_defaults_and_can_be_disabled(monkeypatch, tmp_path) -> None:
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("OPENROUTER_FALLBACK_MODELS", raising=False)

    settings = Settings.from_env(tmp_path / "missing.env")

    assert settings.openrouter_fallback_models == (
        "nvidia/nemotron-3.5-lightning:free",
        "openrouter/free",
    )

    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "")
    disabled = Settings.from_env(tmp_path / "missing.env")
    assert disabled.openrouter_fallback_models == ()


def test_groq_can_be_the_primary_provider(monkeypatch, tmp_path) -> None:
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    settings = Settings.from_env(tmp_path / "missing.env")

    assert settings.ai_provider == "groq"
    assert settings.active_model == "openai/gpt-oss-20b"


def test_ai_continuations_default_to_two(monkeypatch, tmp_path) -> None:
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("AI_MAX_CONTINUATIONS", raising=False)

    settings = Settings.from_env(tmp_path / "missing.env")

    assert settings.ai_max_continuations == 2


def test_media_limit_defaults_to_eight_mebibytes(monkeypatch, tmp_path) -> None:
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("MAX_MEDIA_BYTES", raising=False)

    settings = Settings.from_env(tmp_path / "missing.env")

    assert settings.max_media_bytes == 8 * 1024 * 1024


def test_wall_limits_are_safe_for_small_container(monkeypatch, tmp_path) -> None:
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("WALL_MAX_SOURCE_BYTES", raising=False)
    monkeypatch.delenv("WALL_MAX_PIXELS", raising=False)
    monkeypatch.delenv("WALL_TILE_SIZE", raising=False)

    settings = Settings.from_env(tmp_path / "missing.env")

    assert settings.wall_max_source_bytes == 8 * 1024 * 1024
    assert settings.wall_max_pixels == 12_000_000
    assert settings.wall_tile_size == 1080


def test_wall_tile_size_is_capped(monkeypatch, tmp_path) -> None:
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("WALL_TILE_SIZE", "1441")

    with pytest.raises(ConfigError, match="WALL_TILE_SIZE"):
        Settings.from_env(tmp_path / "missing.env")


def test_wall_pixel_limit_cannot_exceed_safe_cap(monkeypatch, tmp_path) -> None:
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("WALL_MAX_PIXELS", "12000001")

    with pytest.raises(ConfigError, match="WALL_MAX_PIXELS"):
        Settings.from_env(tmp_path / "missing.env")


def test_automatic_fallback_and_spam_protection_defaults(monkeypatch, tmp_path) -> None:
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("AI_AUTO_FALLBACK_ENABLED", raising=False)
    monkeypatch.delenv("AI_FALLBACK_COOLDOWN_SECONDS", raising=False)
    monkeypatch.delenv("AI_TRANSIENT_COOLDOWN_SECONDS", raising=False)
    monkeypatch.delenv("SPAM_VIOLATION_LIMIT", raising=False)
    monkeypatch.delenv("SPAM_BLOCK_SECONDS", raising=False)

    settings = Settings.from_env(tmp_path / "missing.env")

    assert settings.ai_auto_fallback_enabled is True
    assert settings.ai_fallback_cooldown_seconds == 600.0
    assert settings.ai_transient_cooldown_seconds == 60.0
    assert settings.spam_violation_limit == 3
    assert settings.spam_block_seconds == 300.0


@pytest.mark.parametrize("raw", ["0", "-1", "NaN", "inf", "not-a-number"])
def test_transient_cooldown_rejects_invalid_values(monkeypatch, tmp_path, raw):
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("AI_TRANSIENT_COOLDOWN_SECONDS", raw)
    with pytest.raises(ConfigError, match="AI_TRANSIENT_COOLDOWN_SECONDS"):
        Settings.from_env(tmp_path / "missing.env")


def test_ai_continuations_are_capped_at_five(monkeypatch, tmp_path) -> None:
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("AI_MAX_CONTINUATIONS", "6")

    with pytest.raises(ConfigError, match="AI_MAX_CONTINUATIONS"):
        Settings.from_env(tmp_path / "missing.env")


def test_inline_defaults_and_explicit_access_settings(monkeypatch, tmp_path):
    for name in TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("INLINE_ENABLED", raising=False)
    monkeypatch.delenv("INLINE_ALLOWED_USER_IDS", raising=False)
    settings = Settings.from_env(tmp_path / "missing.env")
    assert settings.inline_enabled and settings.inline_allowed_user_ids == frozenset()
    monkeypatch.setenv("INLINE_ENABLED", "false")
    monkeypatch.setenv("INLINE_ALLOWED_USER_IDS", "123456789, 987654321")
    settings = Settings.from_env(tmp_path / "missing.env")
    assert not settings.inline_enabled
    assert settings.inline_allowed_user_ids == frozenset({123456789, 987654321})


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


def test_free_only_flag_is_opt_in(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("AI_FREE_ONLY", raising=False)
    assert not Settings.from_env(tmp_path / "missing.env").ai_free_only
    monkeypatch.setenv("AI_FREE_ONLY", "true")
    assert Settings.from_env(tmp_path / "missing.env").ai_free_only
