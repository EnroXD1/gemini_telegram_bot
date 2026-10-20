import asyncio
import json
from datetime import UTC, datetime
from email.utils import format_datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from google.genai.errors import APIError

from bot.control_panel import diagnostics_view
from bot.gemini import (
    GeminiRequestError,
    GeminiService,
    _detached_request_error,
    _ModelRoute,
    _retry_after_seconds,
)
from bot.models import PromptBundle


class Clock:
    now = 100.0
    wall = 1_800_000_000.0

    def monotonic(self):
        return self.now

    def time(self):
        return self.wall


def completion(text="Ответ", *, finish="stop"):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}, "finish_reason": finish}],
        },
    )


@pytest.fixture
async def factory(monkeypatch):
    clock = Clock()
    monkeypatch.setattr("bot.gemini.time", clock)
    clients = []

    def create(handler, *, provider="openrouter", parallel=1):
        service = GeminiService.__new__(GeminiService)
        service._settings = SimpleNamespace(
            ai_auto_fallback_enabled=True,
            ai_fallback_cooldown_seconds=600,
            ai_transient_cooldown_seconds=60,
            gemini_retry_attempts=3,
            ai_max_continuations=2,
            gemini_timeout_seconds=2,
            gemini_system_prompt="system",
            gemini_temperature=0.5,
            gemini_max_output_tokens=100,
            gemini_store_interactions=True,
            openrouter_model="vendor/main",
            openrouter_fallback_models=("vendor/reserve",),
            groq_model="groq/main",
            groq_fallback_models=(),
            groq_max_output_tokens=100,
            groq_fallback_enabled=True,
            gemini_model="google/main",
        )
        client = httpx.AsyncClient(
            base_url="https://openrouter.test/",
            transport=httpx.MockTransport(handler),
        )
        clients.append(client)
        service._openrouter_client = client
        service._groq_client = None
        service._client = None
        service._provider = provider
        service._model = "vendor/main"
        service._interactions_available = True
        service._semaphore = asyncio.Semaphore(parallel)
        service._route_cooldowns = {}
        service._route_probes = set()
        service._free_only = False
        return service

    yield create, clock
    for client in clients:
        await client.aclose()


@pytest.mark.parametrize(
    "code,duration", [(429, 600), (503, 60), (500, 60), (502, 60), (504, 60), (408, 60)]
)
async def test_errors_quarantine_route_across_requests_then_recover(factory, code, duration):
    create, clock = factory
    calls = []

    def handle(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "vendor/main" and calls.count(model) == 1:
            return httpx.Response(code)
        return completion(model)

    service = create(handle)
    # Non-503 5xx normally permit in-request retries; this test isolates quarantine.
    service._settings.gemini_retry_attempts = 1
    notify = AsyncMock()
    await service.generate(PromptBundle("Первый"), None, on_fallback=notify)
    assert service.cooldown_remaining("openrouter", "vendor/main") == duration
    await service.generate(PromptBundle("Второй"), None, on_fallback=notify)
    assert calls == ["vendor/main", "vendor/reserve", "vendor/reserve"]
    assert service.health.routes[("openrouter", "vendor/main")].attempts == 1
    assert notify.await_count == 1
    clock.now += duration
    await service.generate(PromptBundle("Восстановление"), None)
    assert calls[-1] == "vendor/main"
    assert ("openrouter", "vendor/main") not in service._route_cooldowns
    await service.generate(PromptBundle("Основной маршрут"), None)
    assert calls[-1] == "vendor/main"


@pytest.mark.parametrize("code", [429, 503, 500])
async def test_retry_after_overrides_default_and_does_not_sleep(factory, monkeypatch, code):
    create, clock = factory
    calls = []

    def handle(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "vendor/main":
            return httpx.Response(code, headers={"Retry-After": "120"})
        return completion()

    service = create(handle)
    sleep = AsyncMock(side_effect=AssertionError("must use fallback, not sleep"))
    monkeypatch.setattr("bot.gemini.asyncio.sleep", sleep)
    await service.generate(PromptBundle("Запрос"), None)
    assert calls == ["vendor/main", "vendor/reserve"]
    assert service.cooldown_remaining("openrouter", "vendor/main") == 120
    sleep.assert_not_awaited()
    clock.now += 119
    await service.generate(PromptBundle("Ещё"), None)
    assert calls[-1] == "vendor/reserve" and calls.count("vendor/main") == 1


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("90", 90),
        ("0", 1),
        (" 15 ", 15),
        ("bad", None),
        ("NaN", None),
        ("inf", None),
        ("-1", None),
        ("1.5", None),
        ("", None),
        ("9" * 300, None),
    ],
)
def test_retry_after_parser_handles_invalid_values(factory, raw, expected):
    response = httpx.Response(503, headers={"Retry-After": raw})
    error = httpx.HTTPStatusError(
        "failure", request=httpx.Request("POST", "https://api.test"), response=response
    )
    assert _retry_after_seconds(error) == expected


async def test_retry_after_http_date_uses_wall_clock_then_monotonic_deadline(factory):
    create, clock = factory
    stamp = format_datetime(datetime.fromtimestamp(clock.wall + 75, UTC), usegmt=True)

    def handle(request):
        if json.loads(request.content)["model"] == "vendor/main":
            return httpx.Response(503, headers={"Retry-After": stamp})
        return completion()

    service = create(handle)
    await service.generate(PromptBundle("Вопрос"), None)
    assert service.cooldown_remaining("openrouter", "vendor/main") == 75
    clock.wall += 1_000_000  # wall-clock corrections must not alter a running quarantine
    clock.now += 10
    assert service.cooldown_remaining("openrouter", "vendor/main") == 65


async def test_google_sdk_response_headers_are_carried_without_retaining_response(factory):
    create, clock = factory
    service = create(lambda request: completion())
    response = httpx.Response(429, headers={"Retry-After": "45"})
    error = APIError(429, {"error": {"message": "quota exceeded"}}, response=response)
    generate = AsyncMock(side_effect=error)
    service._client = SimpleNamespace(
        aio=SimpleNamespace(
            interactions=SimpleNamespace(create=generate),
        )
    )
    service._provider, service._model = "google", "google/main"
    await service.generate(PromptBundle("Первый"), None)
    await service.generate(PromptBundle("Второй"), None)
    assert generate.await_count == 1
    assert service.cooldown_remaining("google", "google/main") == 45
    detached = _detached_request_error(GeminiRequestError("failure", retry_after_seconds=45))
    assert detached.retry_after_seconds == 45 and detached.__cause__ is None


async def test_groq_503_uses_same_breaker_and_header(factory):
    create, clock = factory
    service = create(lambda request: completion())
    groq_calls = []

    def handle(request):
        groq_calls.append(request)
        return httpx.Response(503, headers={"Retry-After": "80"})

    async with httpx.AsyncClient(
        base_url="https://groq.test/", transport=httpx.MockTransport(handle)
    ) as client:
        service._groq_client = client
        service._provider, service._model = "groq", "groq/main"
        await service.generate(PromptBundle("Первый"), None)
        await service.generate(PromptBundle("Второй"), None)
    assert len(groq_calls) == 1
    assert service.cooldown_remaining("groq", "groq/main") == 80


async def test_queued_requests_recheck_quarantine_after_semaphore(factory):
    create, _ = factory
    calls = []

    def handle(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        return httpx.Response(503) if model == "vendor/main" else completion()

    service = create(handle, parallel=1)
    await service._semaphore.acquire()
    tasks = [
        asyncio.create_task(service.generate(PromptBundle("Вопрос"), None)) for _ in range(2)
    ]
    try:
        # Both generate() calls choose the primary before either can call the API.
        for _ in range(100):
            if service._semaphore._waiters and len(service._semaphore._waiters) == 2:
                break
            await asyncio.sleep(0)
        assert len(service._semaphore._waiters) == 2
        service._semaphore.release()
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert calls.count("vendor/main") == 1 and calls.count("vendor/reserve") == 2
        assert service.health.routes[("openrouter", "vendor/main")].failures == 1
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_one_recovery_probe_other_requests_use_fallback(factory):
    create, clock = factory
    probing, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def handle(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "vendor/main":
            if calls.count(model) == 1:
                return httpx.Response(503)
            probing.set()
            await release.wait()
        return completion()

    service = create(handle, parallel=3)
    await service.generate(PromptBundle("Ошибка"), None)
    clock.now += 61
    task = asyncio.create_task(service.generate(PromptBundle("Проверка"), None))
    try:
        await asyncio.wait_for(probing.wait(), 2)
        assert service.route_is_probing("openrouter", "vendor/main")
        report, _ = diagnostics_view(service)
        assert "проверка восстановления" in report
        await asyncio.wait_for(service.generate(PromptBundle("Параллельный"), None), 2)
        assert calls.count("vendor/main") == 2 and calls[-1] == "vendor/reserve"
        release.set()
        await asyncio.wait_for(task, 2)
        assert not service.route_is_probing("openrouter", "vendor/main")
        assert ("openrouter", "vendor/main") not in service._route_cooldowns
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_failed_probe_reopens_breaker_and_cancellation_releases_probe(factory):
    create, clock = factory
    started, release = asyncio.Event(), asyncio.Event()
    cancel_probe = False

    async def handle(request):
        if json.loads(request.content)["model"] == "vendor/main":
            if cancel_probe:
                started.set()
                await release.wait()
            return httpx.Response(503)
        return completion()

    service = create(handle, parallel=2)
    await service.generate(PromptBundle("Первый сбой"), None)
    clock.now += 61
    await service.generate(PromptBundle("Повторный сбой"), None)
    assert service.cooldown_remaining("openrouter", "vendor/main") == 60
    clock.now += 61
    cancel_probe = True
    task = asyncio.create_task(service.generate(PromptBundle("Отмена"), None))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not service._route_probes
    release.set()
    await service.generate(PromptBundle("После отмены"), None)
    assert service.cooldown_remaining("openrouter", "vendor/main") == 60


async def test_all_routes_quarantined_fail_fast_with_earliest_retry(factory):
    create, clock = factory
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(503)

    service = create(handle)
    with pytest.raises(GeminiRequestError):
        await service.generate(PromptBundle("Сбой"), None)
    service._route_cooldowns[("openrouter", "vendor/main")] = clock.now + 120
    with pytest.raises(GeminiRequestError, match="60 сек") as raised:
        await service.generate(PromptBundle("Без вызовов API"), None)
    assert len(calls) == 2 and raised.value.retry_after_seconds == 60


async def test_quarantine_is_respected_when_automatic_fallback_disabled(factory):
    create, _ = factory
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(503)

    service = create(handle)
    service._settings.ai_auto_fallback_enabled = False
    for _ in range(2):
        with pytest.raises(GeminiRequestError):
            await service.generate(PromptBundle("Вопрос"), None)
    assert len(calls) == 1


@pytest.mark.parametrize("code", [400, 401, 403])
async def test_request_specific_errors_do_not_quarantine_other_requests(factory, code):
    create, _ = factory
    service = create(lambda request: httpx.Response(code))
    with pytest.raises(GeminiRequestError):
        await service.generate(PromptBundle("Неверный запрос"), None)
    assert service._route_cooldowns == {}


@pytest.mark.parametrize("error_class", [httpx.ConnectError, httpx.ReadTimeout])
async def test_transport_failure_falls_back_and_quarantines_without_fake_http_code(
    factory, error_class
):
    create, _ = factory
    calls = []

    def handle(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "vendor/main":
            raise error_class("transport unavailable", request=request)
        return completion()

    service = create(handle)
    await service.generate(PromptBundle("Первый"), None)
    await service.generate(PromptBundle("Второй"), None)
    assert calls == ["vendor/main", "vendor/reserve", "vendor/reserve"]
    assert service.cooldown_remaining("openrouter", "vendor/main") == 60
    assert service.health.routes[("openrouter", "vendor/main")].last_code is None


@pytest.mark.parametrize("failure,duration", [("503", 90), ("timeout", 60)])
async def test_partial_answer_error_still_quarantines_next_request(factory, failure, duration):
    create, _ = factory
    calls = []

    def handle(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "vendor/main":
            if calls.count(model) == 1:
                return completion("Часть ответа", finish="length")
            if failure == "timeout":
                raise TimeoutError
            return httpx.Response(503, headers={"Retry-After": "90"})
        return completion("Резерв")

    service = create(handle)
    result = await service.generate(PromptBundle("Первый"), None)
    assert result.truncated and "Часть ответа" in result.text
    assert service.cooldown_remaining("openrouter", "vendor/main") == duration
    await service.generate(PromptBundle("Второй"), None)
    assert calls == ["vendor/main", "vendor/main", "vendor/reserve"]


async def test_quota_is_model_local_but_existing_provider_wide_circuit_applies(factory):
    create, clock = factory
    service = create(lambda request: completion())
    service._start_route_cooldown(
        _ModelRoute("openrouter", "vendor/main"), GeminiRequestError("quota", status_code=429)
    )
    assert not service._route_is_cooling_down(_ModelRoute("openrouter", "vendor/reserve"))
    service._route_cooldowns[("openrouter", "*")] = clock.now + 30
    assert service._route_is_cooling_down(_ModelRoute("openrouter", "vendor/reserve"))


async def test_prompt_specific_safety_block_is_not_cached_as_provider_outage(factory):
    create, _ = factory
    calls = []

    def handle(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "vendor/main":
            return completion("", finish="content_filter")
        return completion()

    service = create(handle)
    await service.generate(PromptBundle("Первый"), None)
    await service.generate(PromptBundle("Другой запрос"), None)
    assert calls.count("vendor/main") == 2 and service._route_cooldowns == {}
