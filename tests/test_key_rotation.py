import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from google.genai.errors import APIError

from bot.config import ConfigError, Settings
from bot.control_panel import diagnostics_view
from bot.gemini import GeminiRequestError, GeminiService
from bot.key_pool import KeyPool
from bot.models import ConversationMessage, MediaPayload, PromptBundle

GOOGLE_KEYS = ("FAKE-GOOGLE-SECRET-A", "FAKE-GOOGLE-SECRET-B", "FAKE-GOOGLE-SECRET-C")
GROQ_KEYS = ("FAKE-GROQ-SECRET-A", "FAKE-GROQ-SECRET-B", "FAKE-GROQ-SECRET-C")


class Clock:
    now = 100.0

    def monotonic(self):
        return self.now

    def time(self):
        return 1_800_000_000.0


def google_reply(text="Google ответ", finish="STOP"):
    return SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason=finish)])


def google_error(code=429, delay="60"):
    return APIError(
        code,
        {"error": {"message": "quota" if code == 429 else "failure"}},
        response=httpx.Response(code, headers={"Retry-After": delay}),
    )


def chat_reply(text="Резервный ответ", finish="stop"):
    return httpx.Response(
        200,
        json={
            "model": "actual/model",
            "choices": [{"message": {"content": text}, "finish_reason": finish}],
        },
    )


@pytest.fixture
async def factory(monkeypatch):
    clock = Clock()
    monkeypatch.setattr("bot.gemini.time", clock)
    services = []

    async def create(
        *, google=(), groq=(), google_handler=None, groq_handler=None, fallback=True
    ):
        calls, clients, fallback_calls, groq_calls = [], {}, [], []

        def make_client(*, api_key, **kwargs):
            async def generate(**values):
                calls.append((api_key, values))
                if google_handler is None:
                    return google_reply()
                value = google_handler(api_key, values)
                return await value if asyncio.iscoroutine(value) else value

            client = SimpleNamespace(
                aio=SimpleNamespace(
                    models=SimpleNamespace(generate_content=AsyncMock(side_effect=generate)),
                    interactions=SimpleNamespace(
                        create=AsyncMock(
                            return_value=SimpleNamespace(id="id", output_text="Один ключ")
                        )
                    ),
                    aclose=AsyncMock(),
                )
            )
            clients[api_key] = client
            return client

        monkeypatch.setattr("bot.gemini.genai.Client", make_client)
        settings = SimpleNamespace(
            ai_provider="google" if google else "groq",
            max_concurrent_requests=4,
            gemini_api_key=google[0] if google else "",
            google_api_keys=google,
            gemini_vertex_ai=False,
            gemini_model="google/top",
            gemini_store_interactions=True,
            groq_api_key=groq[0] if groq else "",
            groq_api_keys=groq,
            groq_model="groq/top",
            groq_fallback_models=(),
            groq_fallback_enabled=True,
            groq_max_output_tokens=100,
            openrouter_api_key="",
            openrouter_model="vendor/reserve",
            openrouter_fallback_models=(),
            gemini_timeout_seconds=2,
            gemini_retry_attempts=3,
            ai_auto_fallback_enabled=True,
            ai_max_continuations=2,
            ai_fallback_cooldown_seconds=600,
            ai_transient_cooldown_seconds=60,
            gemini_system_prompt="system",
            gemini_temperature=0.5,
            gemini_max_output_tokens=100,
        )
        service = GeminiService(settings)

        def fallback_handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": []})
            fallback_calls.append(request)
            return chat_reply()

        if fallback:
            service._openrouter_client = httpx.AsyncClient(
                base_url="https://openrouter.test/",
                transport=httpx.MockTransport(fallback_handler),
            )
        if groq:
            await service._groq_client.aclose()

            async def groq_transport(request):
                groq_calls.append(request)
                if groq_handler is None:
                    return chat_reply("Groq ответ")
                value = groq_handler(request)
                return await value if asyncio.iscoroutine(value) else value

            service._groq_client = httpx.AsyncClient(
                base_url="https://groq.test/",
                headers={"Authorization": f"Bearer {groq[0]}"},
                transport=httpx.MockTransport(groq_transport),
            )
        services.append(service)
        return SimpleNamespace(
            service=service,
            calls=calls,
            clients=clients,
            fallback=fallback_calls,
            groq_calls=groq_calls,
            clock=clock,
        )

    yield create
    for service in services:
        await service.close()


async def test_google_429_tries_next_key_same_model_before_provider_fallback(factory):
    def generate(key, kwargs):
        if key == GOOGLE_KEYS[0]:
            raise google_error()
        return google_reply()

    env = await factory(google=GOOGLE_KEYS, google_handler=generate)
    assert len(env.clients) == 1  # remaining SDK clients are created only when used
    result = await env.service.generate(PromptBundle("Первый"), None)
    await env.service.generate(PromptBundle("Второй"), None)
    assert result.provider == "google" and result.text == "Google ответ"
    assert [key for key, _ in env.calls] == [GOOGLE_KEYS[0], GOOGLE_KEYS[1], GOOGLE_KEYS[1]]
    assert all(kwargs["model"] == "google/top" for _, kwargs in env.calls)
    assert len(env.clients) == 2 and env.fallback == []
    assert env.service.cooldown_remaining("google", "google/top") == 0
    assert env.service.key_pool_status("google", "google/top") == (2, 3)


async def test_all_google_keys_cooling_skip_provider_until_earliest_key_recovers(factory):
    attempts = {key: 0 for key in GOOGLE_KEYS}

    def generate(key, kwargs):
        attempts[key] += 1
        if attempts[key] == 1:
            raise google_error(delay=str((10, 30, 60)[GOOGLE_KEYS.index(key)]))
        return google_reply("Восстановлено")

    env = await factory(google=GOOGLE_KEYS, google_handler=generate)
    first = await env.service.generate(PromptBundle("Первый"), None)
    assert first.provider == "openrouter" and len(env.calls) == 3
    assert env.service.cooldown_remaining("google", "google/top") == 10
    await env.service.generate(PromptBundle("Второй"), None)
    assert len(env.calls) == 3 and len(env.fallback) == 2
    env.clock.now += 11
    recovered = await env.service.generate(PromptBundle("Повторная проверка"), None)
    assert recovered.provider == "google" and recovered.text == "Восстановлено"
    assert env.calls[-1][0] == GOOGLE_KEYS[0]


async def test_google_local_history_and_media_survive_key_switch_without_server_ids(factory):
    def generate(key, kwargs):
        if key == GOOGLE_KEYS[0]:
            raise google_error()
        return google_reply()

    env = await factory(google=GOOGLE_KEYS, google_handler=generate)
    data = b"image bytes"
    bundle = PromptBundle(
        "Текущий вопрос", (MediaPayload("photo", "image", "image/jpeg", data),)
    )
    history = (
        ConversationMessage("user", "Меня зовут Иван"),
        ConversationMessage("assistant", "Привет"),
    )
    result = await env.service.generate(bundle, "server-id-from-other-project", history=history)
    assert env.service.uses_local_history and result.interaction_id is None
    assert result.context_was_reset
    for _, kwargs in env.calls:
        assert "previous_interaction_id" not in kwargs
        assert "Иван" in str(kwargs["contents"]) and "Текущий вопрос" in str(kwargs["contents"])
        assert kwargs["contents"][1].inline_data.data == data
    assert all(
        client.aio.interactions.create.await_count == 0 for client in env.clients.values()
    )


async def test_single_key_keeps_legacy_google_interactions(factory):
    env = await factory(google=GOOGLE_KEYS[:1])
    result = await env.service.generate(PromptBundle("Один ключ"), "previous-id")
    call = env.clients[GOOGLE_KEYS[0]].aio.interactions.create.await_args
    assert call.kwargs["previous_interaction_id"] == "previous-id"
    assert result.interaction_id == "id" and not env.service.uses_local_history


async def test_groq_rotation_uses_request_headers_without_mutating_shared_client(factory):
    def handle(request):
        if request.headers["Authorization"] == f"Bearer {GROQ_KEYS[0]}":
            return httpx.Response(429, headers={"Retry-After": "45"})
        return chat_reply("Ответ Groq")

    env = await factory(groq=GROQ_KEYS, groq_handler=handle)
    first = await env.service.generate(PromptBundle("Первый"), None)
    await env.service.generate(PromptBundle("Второй"), None)
    assert first.provider == "groq"
    assert [r.headers["Authorization"] for r in env.groq_calls] == [
        f"Bearer {GROQ_KEYS[0]}",
        f"Bearer {GROQ_KEYS[1]}",
        f"Bearer {GROQ_KEYS[1]}",
    ]
    assert env.service._groq_client.headers["Authorization"] == f"Bearer {GROQ_KEYS[0]}"
    assert {json.loads(r.content)["model"] for r in env.groq_calls} == {"groq/top"}
    assert env.fallback == []


@pytest.mark.parametrize("code", [401, 403, 503])
async def test_non_quota_failures_do_not_cycle_credentials(factory, code):
    def generate(key, kwargs):
        raise google_error(code)

    env = await factory(google=GOOGLE_KEYS, google_handler=generate)
    if code == 503:
        result = await env.service.generate(PromptBundle("Вопрос"), None)
        assert result.provider == "openrouter"
        assert env.service.cooldown_remaining("google", "google/top") == 60
    else:
        with pytest.raises(GeminiRequestError):
            await env.service.generate(PromptBundle("Вопрос"), None)
    assert len(env.calls) == 1
    assert env.service.key_pool_status("google", "google/top") == (3, 3)


async def test_key_failover_is_bounded_even_with_provider_fallback_disabled(factory):
    def generate(key, kwargs):
        raise google_error()

    env = await factory(google=GOOGLE_KEYS, google_handler=generate, fallback=False)
    env.service._settings.ai_auto_fallback_enabled = False
    for _ in range(2):
        with pytest.raises(GeminiRequestError):
            await env.service.generate(PromptBundle("Вопрос"), None)
    assert len(env.calls) == 3
    assert len({key for key, _ in env.calls}) == 3


async def test_partial_google_response_preserved_and_failed_key_stays_cooling(factory):
    first_key_calls = 0

    def generate(key, kwargs):
        nonlocal first_key_calls
        if key == GOOGLE_KEYS[0]:
            first_key_calls += 1
            if first_key_calls == 1:
                return google_reply("Часть ответа", finish="MAX_TOKENS")
            raise google_error()
        return google_reply("Следующий ответ")

    env = await factory(google=GOOGLE_KEYS, google_handler=generate)
    partial = await env.service.generate(PromptBundle("Длинный запрос"), None)
    assert partial.truncated and "Часть ответа" in partial.text
    assert env.service.key_pool_status("google", "google/top") == (2, 3)
    second = await env.service.generate(PromptBundle("Новый запрос"), None)
    assert second.text == "Следующий ответ" and env.calls[-1][0] == GOOGLE_KEYS[1]


async def test_diagnostics_and_logs_never_contain_key_values(factory, caplog):
    def generate(key, kwargs):
        if key == GOOGLE_KEYS[0]:
            # Even an SDK exception that echoes a credential must not enter a traceback log.
            error = google_error()
            error.args = (f"Provider echoed {key}",)
            raise error
        return google_reply()

    env = await factory(google=GOOGLE_KEYS, google_handler=generate)
    await env.service.generate(PromptBundle("Вопрос"), None)
    report, _ = diagnostics_view(env.service)
    assert "Пул ключей: доступно 2/3" in report
    for key in GOOGLE_KEYS:
        assert key not in report and key not in caplog.text


async def test_google_sdk_clients_are_reused_and_all_closed(factory):
    def generate(key, kwargs):
        if key == GOOGLE_KEYS[0]:
            raise google_error()
        return google_reply()

    env = await factory(google=GOOGLE_KEYS, google_handler=generate)
    await env.service.generate(PromptBundle("Вопрос"), None)
    await env.service.generate(PromptBundle("Ещё вопрос"), None)
    assert len(env.clients) == 2
    await env.service.close()
    assert all(client.aio.aclose.await_count == 1 for client in env.clients.values())
    # The fixture will close again; fake aclose is deliberately idempotent.


async def test_strict_free_mode_never_uses_google_or_groq_pool(factory):
    env = await factory(google=GOOGLE_KEYS, groq=GROQ_KEYS)
    env.service.set_free_only(True)
    await env.service.generate(PromptBundle("Только бесплатно"), None)
    assert env.calls == [] and env.groq_calls == []
    assert len(env.fallback) == 1


def test_pool_deduplication_per_model_cooldowns_and_single_recovery_probe():
    clock = Clock()
    pool = KeyPool(("key-a", "key-a", "key-b"), clock=clock.monotonic)
    assert len(pool) == 2
    assert pool.acquire("model", set()) == 0
    pool.pause(0, "model", 10)
    assert pool.acquire("model", set()) == 1
    pool.pause(1, "model", 30)
    assert pool.acquire("model", set()) is None and pool.retry_in("model") == 10
    assert pool.acquire("another-model", set()) == 0
    clock.now += 11
    assert pool.acquire("model", set()) == 0
    assert pool.acquire("model", set()) is None  # the recovering key has one probe only
    pool.release(0, "model")  # cancellation must allow a later recovery attempt
    assert pool.acquire("model", set()) == 0
    pool.release(0, "model", success=True)
    assert pool.available("model") == 1


def test_late_success_does_not_clear_a_newer_parallel_429():
    clock = Clock()
    pool = KeyPool(("a", "b"), clock=clock.monotonic)
    assert pool.acquire("model", set()) == 0
    pool.pause(0, "model", 60)
    pool.release(0, "model", success=True)
    assert pool.acquire("model", set()) == 1


@pytest.fixture
def key_env(monkeypatch):
    for name in (
        "GOOGLE_API_KEYS",
        "GEMINI_API_KEYS",
        "GROQ_API_KEYS",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("AI_PROVIDER", "google")


@pytest.mark.parametrize("name", ["GOOGLE_API_KEYS", "GEMINI_API_KEYS"])
def test_google_key_list_precedence_dedup_and_legacy_compatibility(
    key_env, monkeypatch, tmp_path, name
):
    monkeypatch.setenv(name, " key-a, key-b, key-a, ,key-c ")
    monkeypatch.setenv("GEMINI_API_KEY", "legacy-not-in-pool")
    settings = Settings.from_env(tmp_path / "missing.env")
    assert settings.google_api_keys == ("key-a", "key-b", "key-c")
    assert settings.gemini_api_key == "key-a"
    monkeypatch.delenv(name)
    legacy = Settings.from_env(tmp_path / "missing.env")
    assert legacy.google_api_keys == ("legacy-not-in-pool",)


def test_groq_pool_enables_provider_without_singular_key(key_env, monkeypatch, tmp_path):
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEYS", "groq-a,groq-b")
    settings = Settings.from_env(tmp_path / "missing.env")
    assert settings.groq_api_keys == ("groq-a", "groq-b") and settings.groq_fallback_ready


@pytest.mark.parametrize(
    "raw", ["secret with space", "secret\nvalue", ",".join(f"key-{n}" for n in range(17))]
)
def test_invalid_key_pools_fail_without_disclosing_their_contents(
    key_env, monkeypatch, tmp_path, raw
):
    monkeypatch.setenv("GOOGLE_API_KEYS", raw)
    with pytest.raises(ConfigError) as caught:
        Settings.from_env(tmp_path / "missing.env")
    assert raw not in str(caught.value) and "GOOGLE_API_KEYS" in str(caught.value)


def test_configuration_repr_omits_all_credentials(key_env, monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEYS", ",".join(GOOGLE_KEYS))
    monkeypatch.setenv("GROQ_API_KEYS", ",".join(GROQ_KEYS))
    monkeypatch.setenv("OPENROUTER_API_KEY", "FAKE-OPENROUTER-SECRET")
    settings = Settings.from_env(tmp_path / "missing.env")
    for secret in (
        *GOOGLE_KEYS,
        *GROQ_KEYS,
        settings.telegram_bot_token,
        settings.openrouter_api_key,
    ):
        assert secret not in repr(settings)


async def test_groq_all_quarantined_keys_skip_to_fallback(factory):
    env = await factory(
        groq=GROQ_KEYS,
        groq_handler=lambda request: httpx.Response(429, headers={"Retry-After": "30"}),
    )
    await env.service.generate(PromptBundle("Первый"), None)
    await env.service.generate(PromptBundle("Второй"), None)
    assert len(env.groq_calls) == 3 and len(env.fallback) == 2
    assert env.service.cooldown_remaining("groq", "groq/top") == 30


async def test_parallel_request_cannot_restore_a_key_quarantined_by_other_request(factory):
    first_started, release_old, backup_started = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    primary_calls = 0

    async def generate(key, kwargs):
        nonlocal primary_calls
        if key == GOOGLE_KEYS[0]:
            primary_calls += 1
            if primary_calls == 1:
                first_started.set()
                await release_old.wait()
                return google_reply("Старый запрос")
            raise google_error()
        backup_started.set()
        return google_reply("Новый ключ")

    env = await factory(google=GOOGLE_KEYS, google_handler=generate)
    old = asyncio.create_task(env.service.generate(PromptBundle("Первый"), None))
    try:
        await asyncio.wait_for(first_started.wait(), 2)
        new = await env.service.generate(PromptBundle("Второй"), None)
        assert new.text == "Новый ключ"
        await asyncio.wait_for(backup_started.wait(), 2)
        release_old.set()
        await asyncio.wait_for(old, 2)
        assert env.service.key_pool_status("google", "google/top") == (2, 3)
        await env.service.generate(PromptBundle("Третий"), None)
        assert env.calls[-1][0] == GOOGLE_KEYS[1]
    finally:
        release_old.set()
        old.cancel()
        await asyncio.gather(old, return_exceptions=True)


async def test_cancelled_half_open_key_probe_releases_pool_slot(factory):
    started = asyncio.Event()
    calls = 0

    async def generate(key, kwargs):
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise google_error(delay="10")
        started.set()
        await asyncio.Event().wait()

    env = await factory(google=GOOGLE_KEYS, google_handler=generate)
    await env.service.generate(PromptBundle("Все ключи на паузе"), None)
    env.clock.now += 11
    task = asyncio.create_task(env.service.generate(PromptBundle("Проверка"), None))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert env.service.key_pool_status("google", "google/top") == (3, 3)
    assert not env.service._key_pool("google")._probes


async def test_partial_groq_response_quarantines_credential(factory):
    first_calls = 0

    def handle(request):
        nonlocal first_calls
        if request.headers["Authorization"] == f"Bearer {GROQ_KEYS[0]}":
            first_calls += 1
            if first_calls == 1:
                return chat_reply("Частично", finish="length")
            return httpx.Response(429, headers={"Retry-After": "40"})
        return chat_reply("Продолжаем")

    env = await factory(groq=GROQ_KEYS, groq_handler=handle)
    first = await env.service.generate(PromptBundle("Большой ответ"), None)
    assert first.truncated and "Частично" in first.text
    assert env.service.key_pool_status("groq", "groq/top") == (2, 3)
    await env.service.generate(PromptBundle("Новый запрос"), None)
    assert env.groq_calls[-1].headers["Authorization"] == f"Bearer {GROQ_KEYS[1]}"


def test_sdk_initialization_error_does_not_echo_credential(monkeypatch):
    secret = "FAKE-SDK-SECRET"

    def fail(**kwargs):
        raise ValueError(f"bad api_key={secret}")

    monkeypatch.setattr("bot.gemini.genai.Client", fail)
    service = GeminiService.__new__(GeminiService)
    service._settings = SimpleNamespace(gemini_vertex_ai=False)
    with pytest.raises(GeminiRequestError) as caught:
        service._make_google_client(secret)
    assert secret not in str(caught.value) and caught.value.__suppress_context__
