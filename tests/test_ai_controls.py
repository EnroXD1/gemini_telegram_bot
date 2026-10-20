import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.types import Chat, Message, ReplyKeyboardRemove, Update, User

from bot.control_panel import (
    create_control_router,
    diagnostics_view,
    persistent_diagnostics_view,
    restore_ai_settings,
)
from bot.gemini import GeminiRequestError, GeminiService, ModelSelectionError
from bot.handlers import create_router
from bot.keyboards import (
    BACK_BUTTON,
    DIAGNOSTICS_BUTTON,
    HELP_BUTTON,
    HIDE_KEYBOARD_BUTTON,
    MORE_BUTTON,
    RESET_BUTTON,
    SETTINGS_BUTTON,
    STATUS_BUTTON,
    WALL_BUTTON,
    main_keyboard,
)
from bot.model_health import DiagnosticObservation, DiagnosticsSnapshot, RouteAggregate
from bot.models import MediaPayload, PromptBundle
from bot.storage import Storage


@pytest.fixture
async def service_factory():
    services = []

    def create(handler, *, free_only=True, groq=False):
        service = GeminiService.__new__(GeminiService)
        service._settings = SimpleNamespace(
            ai_auto_fallback_enabled=True,
            openrouter_model="openrouter/free",
            openrouter_fallback_models=("vendor/text:free",),
            gemini_model="gemini-test",
            gemini_system_prompt="system",
            gemini_temperature=0.5,
            gemini_max_output_tokens=100,
            gemini_timeout_seconds=2,
            gemini_retry_attempts=1,
            ai_max_continuations=0,
            groq_model="openai/gpt-oss-20b",
            groq_fallback_models=(),
            groq_fallback_enabled=True,
            groq_max_output_tokens=100,
        )
        service._provider = "openrouter"
        service._model = "openrouter/free"
        service._client = None
        service._openrouter_client = httpx.AsyncClient(
            base_url="https://openrouter.test/", transport=httpx.MockTransport(handler)
        )
        service._groq_client = (
            httpx.AsyncClient(
                base_url="https://groq.test/", transport=httpx.MockTransport(handler)
            )
            if groq
            else None
        )
        service._semaphore = asyncio.Semaphore(1)
        service._free_only = free_only
        service._route_cooldowns = {}
        services.append(service)
        return service

    yield create
    for service in services:
        await service.close()


def catalogue():
    return httpx.Response(
        200,
        json={
            "data": [{"id": "vendor/text:free", "pricing": {"prompt": "0", "completion": "0"}}]
        },
    )


def completion():
    return httpx.Response(
        200,
        json={
            "model": "vendor/actual:free",
            "choices": [{"message": {"content": "ответ"}, "finish_reason": "stop"}],
        },
    )


@pytest.mark.parametrize(
    "provider,model",
    [
        ("google", "gemini-test"),
        ("groq", "openai/gpt-oss-20b"),
        ("openrouter", "vendor/paid"),
        ("openrouter", "vendor/paid:free:online"),
        ("openrouter", "@preset:free"),
    ],
)
async def test_free_only_rejects_paid_routes_and_preset_bypasses(
    service_factory, provider, model
):
    service = service_factory(lambda request: completion(), groq=True)
    service._client = SimpleNamespace()
    with pytest.raises(ModelSelectionError, match="Только бесплатно"):
        service.select_model(provider, model)
    service._client = None


async def test_free_mode_never_falls_back_to_groq(service_factory):
    calls = []

    def handle(request):
        calls.append((request.url.host, request.url.path))
        if request.url.path == "/models":
            return catalogue()
        return httpx.Response(429, json={"error": {"message": "SECRET user prompt"}})

    service = service_factory(handle, groq=True)
    with pytest.raises(GeminiRequestError):
        await service.generate(PromptBundle("PRIVATE QUESTION"), None)
    assert all(host == "openrouter.test" for host, _ in calls)
    report, _ = diagnostics_view(service)
    assert "429" in report and "исчерпан лимит" in report
    assert "SECRET" not in report and "PRIVATE QUESTION" not in report
    assert "пауза" in report


@pytest.mark.parametrize(
    "kind,mime", [(None, None), ("image", "image/jpeg"), ("document", "application/pdf")]
)
async def test_zero_price_cap_applies_to_text_and_streamed_media(service_factory, kind, mime):
    payloads = []

    def handle(request):
        if request.url.path == "/models":
            return catalogue()
        payloads.append(json.loads(request.content))
        return completion()

    service = service_factory(handle)
    media = (MediaPayload("file", kind, mime, b"content"),) if kind else ()
    result = await service.generate(PromptBundle("вопрос", media), None)
    assert result.model == "vendor/actual:free"
    assert payloads[0]["provider"]["max_price"] == {
        "prompt": 0,
        "completion": 0,
        "request": 0,
        "image": 0,
    }
    if kind == "document":
        assert payloads[0]["plugins"] == [
            {"id": "file-parser", "pdf": {"engine": "cloudflare-ai"}}
        ]


async def test_catalogue_filters_removed_or_repriced_free_models(service_factory):
    calls = []

    def handle(request):
        if request.url.path == "/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "vendor/text:free",
                            "pricing": {"prompt": "0", "completion": "1"},
                        }
                    ]
                },
            )
        calls.append(json.loads(request.content)["model"])
        return completion()

    service = service_factory(handle)
    service._settings.openrouter_fallback_models += ("removed/model:free",)
    await service.refresh_model_catalog()
    assert service.effective_routes() == (("openrouter", "openrouter/free"),)
    await service.generate(PromptBundle("вопрос"), None)
    assert calls == ["openrouter/free"]


async def test_catalogue_is_cached_without_test_generations(service_factory):
    requests = []

    def handle(request):
        requests.append(request)
        return catalogue()

    service = service_factory(handle)
    assert await service.refresh_model_catalog()
    assert await service.refresh_model_catalog()
    assert [(r.method, r.url.path) for r in requests] == [("GET", "/models")]
    assert not service.health.routes
    assert "запросов ещё не было" in diagnostics_view(service)[0]


async def test_catalogue_outage_does_not_enable_paid_fallback(service_factory):
    sent = []

    def handle(request):
        if request.url.path == "/models":
            return httpx.Response(503)
        sent.append(json.loads(request.content))
        return completion()

    service = service_factory(handle, groq=True)
    await service.generate(PromptBundle("вопрос"), None)
    assert sent[0]["model"] == "openrouter/free"
    assert "provider" in sent[0]
    assert service.catalog_state == "error"


async def test_diagnostics_report_actual_model_without_making_requests(service_factory):
    count = 0

    def handle(request):
        nonlocal count
        count += 1
        return catalogue() if request.url.path == "/models" else completion()

    service = service_factory(handle)
    await service.generate(PromptBundle("скрытый вопрос"), None)
    before = count
    report, markup = diagnostics_view(service)
    assert count == before
    assert "vendor/actual:free" in report
    assert "Успехов 1/1" in report
    assert "скрытый вопрос" not in report
    assert len(report) <= 4000
    assert markup.inline_keyboard


async def test_saved_free_guard_survives_a_paid_selection_and_missing_key(
    service_factory,
    tmp_path,
):
    storage = Storage(tmp_path / "settings.sqlite3")
    await storage.open()
    try:
        await storage.set_ai_preferences(free_only=True, provider="groq", model="paid")
        await storage.close()
        await storage.open()
        service = service_factory(lambda request: completion(), free_only=False, groq=True)
        await restore_ai_settings(storage, service)
        assert service.free_only
        assert service.current_model == "openrouter/free"
        await service._openrouter_client.aclose()
        service._openrouter_client = None
        await restore_ai_settings(storage, service)
        assert service.free_only
        assert service.effective_routes() == ()
    finally:
        await storage.close()


class RecordingSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def close(self):
        pass

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        if method.__api_method__ == "answerCallbackQuery":
            return True
        return Message(
            message_id=100,
            date=0,
            chat=Chat(id=method.chat_id, type="private"),
            text=getattr(method, "text", None),
            from_user=User(id=bot.id, is_bot=True, first_name="bot"),
        )

    async def stream_content(self, *args, **kwargs):
        yield b""


@pytest.fixture
async def panel(service_factory, tmp_path):
    service = service_factory(lambda request: catalogue())
    storage = Storage(tmp_path / "panel.sqlite3")
    await storage.open()
    session = RecordingSession()
    bot = Bot("123456789:fake-token", session=session)
    processor = SimpleNamespace(
        settings=SimpleNamespace(owner_ids={42}, business_auto_reply_enabled=True),
        storage=storage,
        gemini=service,
        cancel=Mock(),
        reset=AsyncMock(),
    )
    dispatcher = Dispatcher()
    dispatcher.include_router(create_control_router(processor))
    yield bot, dispatcher, processor, session
    await storage.close()
    await bot.session.close()


def callback_update(data, *, user_id=42, chat_id=42):
    return Update.model_validate(
        {
            "update_id": 1,
            "callback_query": {
                "id": "query",
                "chat_instance": "instance",
                "data": data,
                "from": {"id": user_id, "is_bot": False, "first_name": "user"},
                "message": {
                    "message_id": 2,
                    "date": 0,
                    "text": "panel",
                    "chat": {"id": chat_id, "type": "private"},
                },
            },
        }
    )


@pytest.mark.parametrize("user_id,chat_id", [(99, 99), (99, 42), (42, 99)])
async def test_unauthorized_and_forwarded_settings_callbacks_cannot_mutate(
    panel, user_id, chat_id
):
    bot, dispatcher, processor, session = panel
    await dispatcher.feed_update(
        bot, callback_update("cfg:free:confirm", user_id=user_id, chat_id=chat_id)
    )
    assert processor.gemini.free_only
    assert await processor.storage.get_runtime_setting("ai_free_only") is None
    assert session.calls[-1].__api_method__ == "answerCallbackQuery"
    assert session.calls[-1].show_alert


async def test_disabling_free_mode_requires_its_inline_confirmation(panel):
    bot, dispatcher, processor, session = panel
    await dispatcher.feed_update(bot, callback_update("cfg:free:off"))
    assert processor.gemini.free_only
    assert "Расходы" in session.calls[-1].text
    assert await processor.storage.get_runtime_setting("ai_free_only") is None
    await dispatcher.feed_update(bot, callback_update("cfg:free:confirm"))
    assert not processor.gemini.free_only
    assert await processor.storage.get_runtime_setting("ai_free_only") == "false"


async def test_enabling_free_mode_selects_and_persists_free_router(panel):
    bot, dispatcher, processor, _ = panel
    processor.gemini.set_free_only(False)
    processor.gemini.select_model("openrouter", "vendor/paid")
    await dispatcher.feed_update(bot, callback_update("cfg:free:on"))
    assert processor.gemini.free_only
    assert await processor.storage.get_ai_selection() == ("openrouter", "openrouter/free")
    assert await processor.storage.get_runtime_setting("ai_free_only") == "true"


async def test_inline_new_dialog_uses_clicking_user_not_bot_as_scope(panel):
    bot, dispatcher, processor, _ = panel
    await dispatcher.feed_update(bot, callback_update("ui:reset", user_id=99, chat_id=99))
    processor.reset.assert_awaited_once()
    assert processor.reset.await_args.args[0].from_user.id == 99
    assert processor.cancel.call_args.args[0].from_user.id == 99


@pytest.mark.parametrize("text", ["/diagnostics", DIAGNOSTICS_BUTTON])
async def test_diagnostics_command_and_reply_button_reach_same_report(panel, text):
    bot, dispatcher, _, session = panel
    update = Update.model_validate(
        {
            "update_id": 1,
            "message": {
                "message_id": 2,
                "date": 0,
                "text": text,
                "from": {"id": 42, "is_bot": False, "first_name": "owner"},
                "chat": {"id": 42, "type": "private"},
            },
        }
    )
    await dispatcher.feed_update(bot, update)
    assert "Диагностика AI-маршрутов" in session.calls[-1].text


def test_regular_users_keyboard_does_not_expose_owner_controls():
    for page in ("main", "other"):
        keyboard = main_keyboard(owner=False, page=page)
        labels = {button.text for row in keyboard.keyboard for button in row}
        assert SETTINGS_BUTTON not in labels
        assert DIAGNOSTICS_BUTTON not in labels
        assert HIDE_KEYBOARD_BUTTON not in labels


@pytest.mark.parametrize("user_id", [42, 99])
async def test_reply_menu_navigation_changes_pages_without_ai_requests(panel, user_id):
    bot, dispatcher, processor, session = panel
    processor.process = AsyncMock()

    async def press(text):
        update = Update.model_validate(
            {
                "update_id": 4,
                "message": {
                    "message_id": 4,
                    "date": 0,
                    "text": text,
                    "from": {"id": user_id, "is_bot": False, "first_name": "user"},
                    "chat": {"id": user_id, "type": "private"},
                },
            }
        )
        await dispatcher.feed_update(bot, update)
        return {
            button.text for row in session.calls[-1].reply_markup.keyboard for button in row
        }

    other = await press(MORE_BUTTON)
    assert BACK_BUTTON in other and STATUS_BUTTON in other
    assert MORE_BUTTON not in other and HIDE_KEYBOARD_BUTTON not in other
    assert (SETTINGS_BUTTON in other) == (user_id == 42)
    assert (DIAGNOSTICS_BUTTON in other) == (user_id == 42)
    main = await press(BACK_BUTTON)
    assert MORE_BUTTON in main and RESET_BUTTON in main and WALL_BUTTON in main
    assert BACK_BUTTON not in main and HIDE_KEYBOARD_BUTTON not in main
    assert await press("/menu") == main
    processor.process.assert_not_awaited()


@pytest.mark.parametrize("text", ["/hidekeyboard", HIDE_KEYBOARD_BUTTON])
async def test_hide_keyboard_button_and_command_remove_reply_keyboard(panel, text):
    bot, dispatcher, _, session = panel
    update = Update.model_validate(
        {
            "update_id": 3,
            "message": {
                "message_id": 3,
                "date": 0,
                "text": text,
                "from": {"id": 99, "is_bot": False, "first_name": "user"},
                "chat": {"id": 99, "type": "private"},
            },
        }
    )
    await dispatcher.feed_update(bot, update)
    response = session.calls[-1]
    assert isinstance(response.reply_markup, ReplyKeyboardRemove)
    assert response.reply_markup.remove_keyboard
    assert "/menu" in response.text


@pytest.mark.parametrize("text", [RESET_BUTTON, WALL_BUTTON, STATUS_BUTTON, HELP_BUTTON])
async def test_public_reply_buttons_are_actions_not_ai_prompts(panel, text):
    bot, dispatcher, processor, session = panel
    processor.settings.wall_max_source_bytes = 8 * 1024 * 1024
    processor.settings.wall_max_pixels = 12_000_000
    processor.settings.wall_tile_size = 1080
    processor.settings.media_download_timeout_seconds = 1
    processor.can_respond = AsyncMock(return_value=True)
    processor.process = AsyncMock()
    processor.scope_key = Mock(return_value="private:42")
    dispatcher.include_router(
        create_router(
            processor,
            albums=SimpleNamespace(add=AsyncMock()),
            monitor=SimpleNamespace(),
            usage=SimpleNamespace(record=AsyncMock()),
        )
    )
    update = Update.model_validate(
        {
            "update_id": 2,
            "message": {
                "message_id": 2,
                "date": 0,
                "text": text,
                "from": {"id": 42, "is_bot": False, "first_name": "owner"},
                "chat": {"id": 42, "type": "private"},
            },
        }
    )
    await dispatcher.feed_update(bot, update)
    assert session.calls[-1].__api_method__ == "sendMessage"
    processor.process.assert_not_awaited()
    if text == RESET_BUTTON:
        processor.reset.assert_awaited_once()


async def test_unknown_inline_model_cannot_overwrite_saved_selection(panel):
    bot, dispatcher, processor, session = panel
    await dispatcher.feed_update(bot, callback_update("cfg:model:forged"))
    assert await processor.storage.get_ai_selection() is None
    assert processor.gemini.current_model == "openrouter/free"
    assert "Список изменился" in session.calls[-1].text


async def test_real_generation_persists_route_failures_and_fallback_successes(
    service_factory, tmp_path
):
    def handle(request):
        model = json.loads(request.content)["model"]
        if model == "vendor/main":
            return httpx.Response(429, json={"error": {"message": "PRIVATE_PROVIDER_ERROR"}})
        return completion()

    storage = Storage(tmp_path / "generation.sqlite3")
    await storage.open()
    service = service_factory(handle, free_only=False)
    service._model = "vendor/main"
    service.health.attach_storage(storage)
    try:
        await service.generate(PromptBundle("PRIVATE_USER_PROMPT"), None)
        await service.generate(PromptBundle("PRIVATE_SECOND_PROMPT"), None)
        snapshot = await storage.get_model_diagnostics()
        routes = {route.model: route for route in snapshot.routes}
        assert routes["vendor/main"].failures == 1  # cooled route is skipped next time
        assert routes["openrouter/free"].successes == 2
        assert sum(route.attempts for route in snapshot.routes) == 3
        assert "PRIVATE" not in str(snapshot)

        # New service and reopened SQLite: counters survive, circuits and live state do not.
        await storage.close()
        await storage.open()
        restarted = service_factory(handle, free_only=False)
        restarted._model = "vendor/main"
        restarted.health.attach_storage(storage)
        report, _ = await persistent_diagnostics_view(
            SimpleNamespace(storage=storage, gemini=restarted)
        )
        assert "ошибок: 1" in report and "успешных: 2" in report
        assert "запросов ещё не было после запуска" in report
        assert "пауза ещё" not in report and "последняя попытка неуспешна" not in report
        assert restarted.cooldown_remaining("openrouter", "vendor/main") == 0
        assert restarted.health.last_response is None and not restarted.health.routes
    finally:
        await storage.close()


async def test_truncated_generation_is_saved_as_success_with_partial_marker(
    service_factory, tmp_path
):
    def handle(request):
        reply = completion().json()
        reply["choices"][0]["finish_reason"] = "length"
        return httpx.Response(200, json=reply)

    service = service_factory(handle, free_only=False)
    storage = Storage(tmp_path / "partial.sqlite3")
    await storage.open()
    service.health.attach_storage(storage)
    try:
        result = await service.generate(PromptBundle("Запрос"), None)
        (route,) = (await storage.get_model_diagnostics()).routes
        assert result.truncated and route.truncated == 1
        assert (route.attempts, route.successes, route.failures) == (1, 1, 0)
    finally:
        await storage.close()


@pytest.mark.parametrize("period", ["24h", "7d", "30d"])
async def test_diagnostic_period_buttons_use_sqlite_without_ai_calls(panel, period):
    bot, dispatcher, processor, session = panel
    processor.storage.get_model_diagnostics = AsyncMock(
        wraps=processor.storage.get_model_diagnostics
    )
    processor.gemini.refresh_model_catalog = AsyncMock()
    await dispatcher.feed_update(bot, callback_update(f"cfg:diagnostics:{period}:0"))
    processor.storage.get_model_diagnostics.assert_awaited_once_with(period)
    processor.gemini.refresh_model_catalog.assert_not_awaited()
    report = session.calls[-1]
    assert "История SQLite" in report.text and "почасовой учёт UTC" in report.text
    assert "попыток ещё нет" in report.text
    buttons = [button for row in report.reply_markup.inline_keyboard for button in row]
    assert {f"cfg:diagnostics:{value}:0" for value in ("24h", "7d", "30d")} <= {
        button.callback_data for button in buttons
    }
    assert any(button.text.startswith("• ") for button in buttons)


@pytest.mark.parametrize("user_id,chat_id", [(99, 99), (99, 42), (42, 99)])
async def test_history_periods_remain_owner_only(panel, user_id, chat_id):
    bot, dispatcher, processor, session = panel
    processor.storage.get_model_diagnostics = AsyncMock()
    await dispatcher.feed_update(
        bot, callback_update("cfg:diagnostics:30d:0", user_id=user_id, chat_id=chat_id)
    )
    processor.storage.get_model_diagnostics.assert_not_awaited()
    assert session.calls[-1].show_alert


@pytest.mark.parametrize(
    "action",
    [
        "diagnostics:all:0",
        "diagnostics:7d:-1",
        "diagnostics:7d:1000000",
        "catalog:7d",
        "catalog::0",
    ],
)
async def test_invalid_diagnostic_callbacks_are_rejected_without_io(panel, action):
    bot, dispatcher, processor, session = panel
    processor.storage.get_model_diagnostics = AsyncMock()
    processor.gemini.refresh_model_catalog = AsyncMock()
    await dispatcher.feed_update(bot, callback_update(f"cfg:{action}"))
    assert "кнопка устарела" in session.calls[-1].text
    processor.storage.get_model_diagnostics.assert_not_awaited()
    processor.gemini.refresh_model_catalog.assert_not_awaited()


async def test_pagination_and_catalog_refresh_preserve_selected_period(panel):
    bot, dispatcher, processor, session = panel
    for i in range(12):
        await processor.storage.record_model_diagnostic(
            DiagnosticObservation("google", f"model-{i}", 2, int(time.time()), False)
        )
    processor.gemini.refresh_model_catalog = AsyncMock()
    await dispatcher.feed_update(bot, callback_update("cfg:catalog:7d:1"))
    processor.gemini.refresh_model_catalog.assert_awaited_once()
    report = session.calls[-1]
    assert "7 дней" in report.text and "страница 2/" in report.text
    buttons = [
        button.callback_data for row in report.reply_markup.inline_keyboard for button in row
    ]
    assert "cfg:diagnostics:7d:1" in buttons and "cfg:catalog:7d:1" in buttons
    assert "cfg:diagnostics:7d:0" in buttons and "cfg:diagnostics:7d:2" in buttons


async def test_diagnostic_read_failure_shows_live_fallback_and_safe_warning(panel, caplog):
    _, _, processor, _ = panel
    processor.storage.get_model_diagnostics = AsyncMock(
        side_effect=RuntimeError("PRIVATE_SECRET")
    )
    processor.gemini.health.persistence_failures = 2
    report, _ = await persistent_diagnostics_view(processor, "30d")
    assert (
        "История SQLite недоступна" in report and "только показатели текущего запуска" in report
    )
    assert "Не удалось сохранить попыток за этот запуск: 2" in report
    assert "PRIVATE_SECRET" not in caplog.text + report


async def test_many_historical_routes_are_paginated_without_hiding_end_of_report(panel):
    _, _, processor, _ = panel
    snapshot = DiagnosticsSnapshot(
        "30d",
        int(time.time()) - 3600,
        int(time.time()),
        tuple(
            RouteAggregate(
                "google", f"route-{i}-" + "x" * 190, 2, 1, 1, 10, 0, int(time.time())
            )
            for i in range(30)
        ),
    )
    for page in (0, 1, 99999):
        report, markup = diagnostics_view(processor.gemini, snapshot, page=page)
        assert len(report) < 4000
        assert report.endswith("Карантин и последние ответы — только в ОЗУ.")
        assert "Доля успешных попыток: 50.0%" in report
        assert "среднее время: 5.0 сек." in report
        assert "История SQLite: 30 дней" in report
        assert len(markup.inline_keyboard) >= 4
