import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.types import Chat, Message, Update, User

from bot.control_panel import create_control_router, diagnostics_view, restore_ai_settings
from bot.gemini import GeminiRequestError, GeminiService, ModelSelectionError
from bot.handlers import create_router
from bot.keyboards import (
    DIAGNOSTICS_BUTTON,
    HELP_BUTTON,
    RESET_BUTTON,
    STATUS_BUTTON,
    WALL_BUTTON,
    main_keyboard,
)
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
    keyboard = main_keyboard(owner=False)
    assert len(keyboard.keyboard) == 2
    assert len(main_keyboard(owner=True).keyboard) == 3


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
