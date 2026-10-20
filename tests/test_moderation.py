import asyncio
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot, Dispatcher, Router
from aiogram.client.session.base import BaseSession
from aiogram.types import Chat, Message, Update, User

from bot.activity import ActivityTracker, CommandActivityMiddleware, history_pages
from bot.app import _set_commands
from bot.gemini import GeminiRequestError
from bot.models import GeminiResult, PromptBundle
from bot.moderation import BanMiddleware, create_moderation_router, users_view
from bot.processor import MessageProcessor
from bot.storage import Storage


class RecordingSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def close(self):
        pass

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        if method.__api_method__ in {"answerCallbackQuery", "setMyCommands"}:
            return True
        if method.__api_method__ == "getMe":
            return User(id=bot.id, is_bot=True, first_name="bot", username="testbot")
        return Message(
            message_id=100,
            date=0,
            chat=Chat(id=method.chat_id, type="private"),
            text=getattr(method, "text", None),
            from_user=User(id=bot.id, is_bot=True, first_name="bot"),
        )

    async def stream_content(self, *args, **kwargs):
        yield b""


def message(text="Вопрос", *, user_id=99, chat_id=None, **values):
    return Message(
        message_id=10,
        date=int(time.time()),
        chat=Chat(id=chat_id or user_id, type="private" if not chat_id else "supergroup"),
        from_user=User(id=user_id, is_bot=False, first_name="Тест", username="tester"),
        text=text,
        **values,
    )


def callback(data, *, user_id=42, chat_id=None, **values):
    return Update.model_validate(
        {
            "update_id": 2,
            "callback_query": {
                "id": "query",
                "chat_instance": "instance",
                "data": data,
                "from": {"id": user_id, "is_bot": False, "first_name": "user"},
                "message": {
                    "message_id": 2,
                    "date": int(time.time()),
                    "text": "panel",
                    "chat": {"id": chat_id or user_id, "type": "private"},
                    **values,
                },
            },
        }
    )


async def seed(storage, user_id=99, name="Тест"):
    await storage.record_bot_user_activity(
        user_id=user_id,
        username="tester",
        display_name=name,
        chat_id=user_id,
        chat_type="private",
        chat_title=None,
        ai_provider="openrouter",
    )


@pytest.fixture
async def env(tmp_path, monkeypatch):
    storage = Storage(tmp_path / "bot.sqlite3")
    await storage.open()
    session = RecordingSession()
    bot = Bot("123456789:fake-token", session=session)
    settings = SimpleNamespace(
        owner_ids=frozenset({42, 43}),
        rate_limit_requests=2,
        rate_limit_window_seconds=60,
        spam_violation_limit=3,
        spam_block_seconds=300,
        allowed_chat_ids=frozenset(),
        group_default_mode="mentions",
        conversation_scope="chat_thread_user",
        openrouter_history_turns=3,
        openrouter_history_max_chars=10000,
        openrouter_history_item_chars=2000,
        business_auto_reply_enabled=True,
    )
    tracker = ActivityTracker(storage, settings.owner_ids, "testbot", bot_id=bot.id)
    processor = MessageProcessor(
        bot=bot,
        bot_user=User(id=bot.id, is_bot=True, first_name="bot", username="testbot"),
        settings=settings,
        storage=storage,
        activity=tracker,
        gemini=SimpleNamespace(
            generate=AsyncMock(return_value=GeminiResult("Ответ", None, provider="openrouter")),
            current_provider="openrouter",
        ),
        media=SimpleNamespace(prepare=AsyncMock(return_value=PromptBundle("Вопрос"))),
        usage=SimpleNamespace(record=AsyncMock()),
    )
    processor.reply = AsyncMock(return_value=[message("Ответ")])
    processor.cancel_user = Mock(wraps=processor.cancel_user)

    @asynccontextmanager
    async def progress(*args, **kwargs):
        yield SimpleNamespace(show_fallback=AsyncMock())

    monkeypatch.setattr("bot.processor.show_progress", progress)
    dispatcher = Dispatcher()
    gate = BanMiddleware(storage, settings.owner_ids, tracker)
    dispatcher.message.outer_middleware(gate)
    dispatcher.edited_message.outer_middleware(gate)
    dispatcher.callback_query.outer_middleware(gate)
    dispatcher.message.outer_middleware(CommandActivityMiddleware(tracker))
    dispatcher.include_router(create_moderation_router(processor))
    fallback = Router()
    received = []

    @fallback.message()
    @fallback.callback_query()
    @fallback.business_message()
    async def catch(event):
        received.append(event)

    dispatcher.include_router(fallback)

    async def send(text, *, user_id=42, **values):
        await dispatcher.feed_update(
            bot,
            Update(update_id=1, message=message(text, user_id=user_id, **values)),
        )

    async def click(data, **values):
        await dispatcher.feed_update(bot, callback(data, **values))

    yield SimpleNamespace(
        storage=storage,
        processor=processor,
        tracker=tracker,
        bot=bot,
        session=session,
        dispatcher=dispatcher,
        received=received,
        send=send,
        click=click,
    )
    await storage.close()
    await bot.session.close()


async def test_bans_persist_without_changing_user_stats(env):
    await seed(env.storage)
    await env.send("/ban 99")
    ban = await env.storage.get_user_ban(99)
    assert ban.banned_by == 42
    assert (await env.storage.get_bot_user(99)).is_banned
    await env.send("/ban 99", user_id=43)
    assert await env.storage.get_user_ban(99) == ban
    await env.storage.close()
    await env.storage.open()
    assert await env.storage.is_user_banned(99)
    assert (await env.storage.get_bot_user(99)).interaction_count == 1
    await env.send("/unban 99")
    assert not await env.storage.is_user_banned(99)
    assert not (await env.storage.get_bot_user(99)).is_banned
    env.processor.cancel_user.assert_any_call(99)


async def test_unknown_id_can_be_banned_before_first_visit(env):
    await env.send("/ban 500")
    assert await env.storage.is_user_banned(500)
    assert (await env.storage.get_bot_usage_stats()).total_users == 0
    await env.send("/banlist")
    assert "500" in env.session.calls[-1].text
    await env.click("mod:view:500:b:10:0")
    assert "Разблокировать" in env.session.calls[-1].reply_markup.inline_keyboard[0][0].text
    await env.click("mod:unban:500:b:10:0")
    assert not await env.storage.is_user_banned(500)
    await env.click("mod:list:0:b:10:0")
    assert "Список блокировок пуст" in env.session.calls[-1].text


@pytest.mark.parametrize("target", ["42", "43", "123456789"])
async def test_cannot_ban_owners_or_bot(env, target):
    await env.send(f"/ban {target}")
    assert not await env.storage.is_user_banned(int(target))
    assert "Нельзя" in env.session.calls[-1].text


@pytest.mark.parametrize(
    "target",
    ["0", "-99", "@user", "99 100", "99; DELETE", "９９", "9223372036854775808", "9" * 100],
)
async def test_invalid_ban_ids_never_write(env, target):
    await env.send(f"/ban {target}")
    assert await env.storage.count_user_bans() == 0


async def test_ban_and_unban_by_reply_in_group(env):
    replied = message("Плохое сообщение", chat_id=-100)
    await env.send("/ban", chat_id=-100, reply_to_message=replied)
    ban = await env.storage.get_user_ban(99)
    assert ban and ban.display_name == "Тест" and ban.username == "tester"
    await env.send("/unban", chat_id=-100, reply_to_message=replied)
    assert not await env.storage.is_user_banned(99)


async def test_ban_uses_visible_forward_author_not_forwarder(env):
    replied = message(
        "Пересылка",
        forward_origin={
            "type": "user",
            "date": 0,
            "sender_user": {"id": 555, "is_bot": False, "first_name": "Автор"},
        },
    )
    await env.send("/ban", reply_to_message=replied)
    assert await env.storage.is_user_banned(555)
    assert not await env.storage.is_user_banned(99)


@pytest.mark.parametrize(
    "values",
    [
        {"forward_origin": {"type": "hidden_user", "date": 0, "sender_user_name": "Автор"}},
        {"sender_chat": {"id": -100, "type": "channel", "title": "Канал"}},
    ],
)
async def test_anonymous_or_hidden_reply_requires_numeric_id(env, values):
    await env.send("/ban", reply_to_message=message("Сообщение", **values))
    assert await env.storage.count_user_bans() == 0


@pytest.mark.parametrize("command", ["/ban 500", "/unban 500", "/banlist", "/users"])
async def test_non_owner_cannot_administer_or_read_users(env, command):
    await env.storage.ban_user(500, 42)
    await env.send(command, user_id=99)
    assert await env.storage.is_user_banned(500)
    assert "владел" in env.session.calls[-1].text


async def test_users_buttons_choose_ban_and_unban_idempotently(env):
    await seed(env.storage)
    await env.send("/users")
    await env.click(env.session.calls[-1].reply_markup.inline_keyboard[0][0].callback_data)
    button = env.session.calls[-1].reply_markup.inline_keyboard[0][0]
    assert "Заблокировать" in button.text
    await env.click(button.callback_data)
    assert await env.storage.is_user_banned(99)
    assert "Разблокировать" in env.session.calls[-1].reply_markup.inline_keyboard[0][0].text
    await env.click(button.callback_data)
    assert await env.storage.is_user_banned(99)  # stale ban must never toggle to unban
    await env.click("mod:unban:99:u:10:0")
    assert not await env.storage.is_user_banned(99)


@pytest.mark.parametrize("user_id,chat_id", [(99, 99), (99, 42), (42, 99)])
async def test_forged_or_forwarded_moderation_callback_is_denied(env, user_id, chat_id):
    await seed(env.storage)
    await env.click("mod:ban:99:u:10:0", user_id=user_id, chat_id=chat_id)
    assert not await env.storage.is_user_banned(99)
    assert env.session.calls[-1].show_alert


@pytest.mark.parametrize(
    "data",
    [
        "mod:ban:500:u:10:0",
        "mod:ban:99:u:0:0",
        "mod:ban:99:u:10:-1",
        "mod:ban:99:wrong:10:0",
        "mod:ban:-99:u:10:0",
        "mod:ban:99:u",
        "mod:toggle:99:u:10:0",
    ],
)
async def test_invalid_or_unknown_callback_cannot_write(env, data):
    await env.click(data)
    assert await env.storage.count_user_bans() == 0
    assert env.session.calls[-1].show_alert


async def test_users_and_banlist_paginate_within_telegram_limits(env):
    for user_id in range(100, 116):
        await seed(env.storage, user_id=user_id, name="😀" * 100)
        await env.storage.ban_user(user_id, 42)
    text, markup = await users_view(env.storage, limit=15)
    assert "Последние 15" in text and "Страница 1/3" in text
    assert len(text.encode("utf-16-le")) // 2 < 4096
    assert all(
        len(b.callback_data.encode()) <= 64 for row in markup.inline_keyboard for b in row
    )
    await env.click("mod:list:0:u:15:2")
    assert "Страница 3/3" in env.session.calls[-1].text
    await env.click("mod:list:0:b:10:3")
    assert "Страница 4/4" in env.session.calls[-1].text
    await env.storage.unban_user(100)
    await env.click("mod:list:0:b:10:3")
    assert "Страница 3/3" in env.session.calls[-1].text


@pytest.mark.parametrize("text", ["Вопрос", "/start", "/menu", "/wall", "/ask Вопрос"])
async def test_banned_messages_stopped_before_commands_and_ai(env, text):
    await env.storage.ban_user(99, 42)
    await env.send(text, user_id=99)
    assert env.received == [] and env.session.calls == []
    events = await env.storage.list_user_events(99)
    assert len(events) == 1 and events[0].reason == "banned"
    assert events[0].kind == ("command" if text in {"/start", "/menu", "/wall"} else "ai")


async def test_banned_callback_is_acknowledged_and_unban_restores_access(env):
    await env.storage.ban_user(99, 42)
    await env.click("ui:new", user_id=99)
    assert not env.received and env.session.calls[-1].show_alert
    assert (await env.storage.list_user_events(99))[0].kind == "callback"
    await env.storage.unban_user(99)
    await env.click("ui:new", user_id=99)
    await env.send("Вопрос", user_id=99)
    assert len(env.received) == 2


async def test_owner_always_has_access_and_business_remains_separate(env):
    await env.storage.ban_user(42, 43)  # legacy/manual DB row cannot lock out owner
    await env.send("/unban 42")
    assert not await env.storage.is_user_banned(42)
    await env.storage.ban_user(99, 42)
    incoming = message(business_connection_id="business")
    await env.dispatcher.feed_update(env.bot, Update(update_id=3, business_message=incoming))
    assert len(env.received) == 1
    assert await env.storage.list_user_events(99) == []


async def test_banned_group_chatter_not_logged_but_mentions_are(env):
    await env.storage.ban_user(99, 42)
    await env.send("Разговор с другом", user_id=99, chat_id=-100)
    assert await env.storage.list_user_events(99) == []
    await env.send("@testbot помоги", user_id=99, chat_id=-100)
    assert len(await env.storage.list_user_events(99)) == 1


@pytest.mark.parametrize("command", ["/history 99", "/activity 99"])
async def test_hidden_commands_silent_for_non_owner_and_groups(env, command):
    await env.send(command, user_id=99)
    await env.send(command, chat_id=-100)
    assert env.session.calls == [] and env.received == []
    assert await env.storage.list_user_events(99) == []


async def test_hidden_commands_absent_from_registered_menus(env):
    await _set_commands(env.bot)
    assert len(env.session.calls) == 2
    for request in env.session.calls:
        assert not {"history", "activity"} & {c.command for c in request.commands}


async def test_history_owner_only_plain_text_with_dates_and_no_generated_answers(env):
    await env.processor.process([message("<b>Личный вопрос</b>")])
    await env.send("/history 99")
    text = env.session.calls[-1].text
    assert "<b>Личный вопрос</b>" in text and "Ответ" not in text
    assert "МСК" in text and "✅ обработано" in text
    assert env.session.calls[-1].parse_mode is None
    assert await env.storage.list_user_events(42) == []


async def test_history_and_activity_report_empty_id_and_invalid_input(env):
    await env.send("/history 999")
    assert "Записей нет" in env.session.calls[-1].text
    await env.send("/activity 999")
    assert "Обращений: 0" in env.session.calls[-1].text
    await env.send("/history @user")
    assert "Формат" in env.session.calls[-1].text


async def test_busy_and_rate_limit_rejections_are_counted_once_without_ai_calls(env):
    env.processor._active_users.add("99")
    await env.processor.process([message("Повтор")])
    env.processor._active_users.clear()
    env.processor._rate_limiter = SimpleNamespace(check=Mock(return_value=30))
    await env.processor.process([message("Слишком часто")])
    events = await env.storage.list_user_events(99)
    assert {event.reason for event in events} == {"busy", "rate_limit"}
    assert all(event.outcome == "rejected" for event in events)
    env.processor.gemini.generate.assert_not_awaited()
    env.processor.media.prepare.assert_not_awaited()
    await env.send("/activity 99")
    assert "Обращений: 2" in env.session.calls[-1].text
    assert "Антиспам: 2" in env.session.calls[-1].text
    assert "Принято: 0; отклонено: 2" in env.session.calls[-1].text


async def test_ban_prevents_queued_album_downloading_or_spending(env):
    await env.storage.ban_user(99, 42)
    await env.processor.process([message("Вопрос")])
    env.processor.gemini.generate.assert_not_awaited()
    env.processor.media.prepare.assert_not_awaited()
    env.processor.reply.assert_not_awaited()
    assert (await env.storage.list_user_events(99))[0].reason == "banned"


async def test_ban_cancels_running_generation_and_records_cancellation(env):
    started = asyncio.Event()

    async def generate(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    env.processor.gemini.generate.side_effect = generate
    task = asyncio.create_task(env.processor.process([message()]))
    await asyncio.wait_for(started.wait(), timeout=2)
    await env.send("/ban 99")
    await asyncio.wait_for(task, timeout=2)
    assert not env.processor._active_users and not env.processor._active_tasks
    env.processor.reply.assert_not_awaited()
    assert (await env.storage.list_user_events(99))[0].outcome == "cancelled"


async def test_cancelled_queued_request_releases_user_guard(env):
    item = message()
    lock = env.processor._scope_locks[env.processor.scope_key(item)]
    await lock.acquire()
    original = env.processor._rate_limiter.check
    queued = asyncio.Event()

    def check(key):
        queued.set()
        return original(key)

    env.processor._rate_limiter.check = check
    task = asyncio.create_task(env.processor.process([item]))
    await asyncio.wait_for(queued.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    lock.release()
    assert not env.processor._active_users
    assert (await env.storage.list_user_events(99))[0].outcome == "cancelled"


async def test_provider_failure_not_misclassified_as_spam(env):
    env.processor.gemini.generate.side_effect = GeminiRequestError("Сбой AI")
    await env.processor.process([message()])
    stats = await env.storage.get_user_activity_stats(99, since=0)
    assert stats.total == stats.errors == 1
    assert stats.rejected == stats.busy == stats.rate_limit == 0


async def test_journal_truncates_content_records_caption_not_files_and_excludes_owners(env):
    photo = [
        {"file_id": "SECRET_FILE_ID", "file_unique_id": "unique", "width": 10, "height": 10}
    ]
    await env.tracker.record([message(None, caption="Подпись", photo=photo)])
    record = (await env.storage.list_user_events(99))[0]
    assert record.content == "Подпись\n[фото]" and "SECRET" not in record.content
    await env.tracker.record([message("😀" * 5000)])
    await env.tracker.record([message(user_id=42)])
    await env.tracker.record([message(business_connection_id="business")])
    records = await env.storage.list_user_events(99)
    assert len(records) == 2 and len(records[0].content) == 1000
    assert await env.storage.list_user_events(42) == []
    pages = history_pages(99, records * 10)
    assert len(pages) > 1 and all(len(p.encode("utf-16-le")) // 2 <= 3800 for p in pages)


async def test_day_week_windows_retention_and_restart_preserve_counts(env, monkeypatch):
    now = int(time.time())
    for ago, outcome, reason in [
        (0, "completed", None),
        (2, "rejected", "busy"),
        (8, "rejected", "rate_limit"),
        (31, "completed", None),
    ]:
        monkeypatch.setattr("bot.storage.time.time", lambda ago=ago: now - ago * 86_400)
        await env.storage.record_user_event(
            user_id=99,
            chat_id=99,
            message_id=1,
            kind="ai",
            content="История",
            outcome=outcome,
            reason=reason,
        )
    monkeypatch.setattr("bot.storage.time.time", lambda: now)
    assert (await env.storage.get_user_activity_stats(99, since=now - 86_400)).total == 1
    stats = await env.storage.get_user_activity_stats(99, since=now - 7 * 86_400)
    assert stats.total == 2 and stats.busy == 1
    assert len(await env.storage.list_user_events(99)) == 3
    assert await env.storage.prune_user_events() == 1
    await env.storage.close()
    await env.storage.open()
    assert await env.storage.get_user_activity_stats(99, since=now - 7 * 86_400) == stats


async def test_commands_recorded_once_but_ask_left_to_processor(env):
    await env.send("/start", user_id=99)
    await env.send("/ask Вопрос", user_id=99)
    await env.send("/start@otherbot", user_id=99)
    events = await env.storage.list_user_events(99)
    assert len(events) == 1 and events[0].kind == "command"


async def test_ordinary_group_chat_never_becomes_ai_audit(env):
    await env.processor.process([message("Общаюсь с другом", chat_id=-100)])
    assert await env.storage.list_user_events(99) == []
    env.processor.gemini.generate.assert_not_awaited()


async def test_user_banned_while_waiting_for_scope_never_calls_ai(env):
    item = message()
    lock = env.processor._scope_locks[env.processor.scope_key(item)]
    await lock.acquire()
    task = asyncio.create_task(env.processor.process([item]))
    # Wait for the audit INSERT, then a small bounded event-loop turn for the guard.
    for _ in range(100):
        if "99" in env.processor._active_users:
            break
        await asyncio.sleep(0.001)
    assert "99" in env.processor._active_users
    await env.send("/ban 99")
    lock.release()
    await asyncio.wait_for(task, timeout=2)
    env.processor.gemini.generate.assert_not_awaited()
    assert (await env.storage.list_user_events(99))[0].reason == "banned"
    assert not env.processor._active_users


async def test_ban_cancels_only_selected_users_non_business_tasks(env):
    tasks = [asyncio.create_task(asyncio.Event().wait()) for _ in range(3)]
    env.processor._active_tasks.update(zip(("first", "other", "business"), tasks, strict=True))
    env.processor._active_task_users.update({"first": 99, "other": 100})
    try:
        assert env.processor.cancel_user(99) == 1
        await asyncio.gather(tasks[0], return_exceptions=True)
        assert tasks[0].cancelled() and not tasks[1].done() and not tasks[2].done()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_legacy_business_owner_has_no_global_moderation_or_audit_access(env):
    await env.storage.save_business_connection(
        connection_id="business",
        owner_user_id=99,
        owner_chat_id=99,
        is_enabled=True,
    )
    await env.send("/history 500", user_id=99)
    await env.send("/activity 500", user_id=99)
    assert env.session.calls == []
    await env.send("/ban 500", user_id=99)
    assert not await env.storage.is_user_banned(500)


async def test_history_and_activity_not_exposed_in_public_help(env):
    # Inspect the real help handler through the complete app router order.
    from bot.album import AlbumBuffer
    from bot.control_panel import create_control_router
    from bot.handlers import create_router

    env.processor.settings.wall_max_source_bytes = 1024
    env.processor.settings.wall_max_pixels = 10000
    env.processor.settings.wall_tile_size = 1080
    env.processor.settings.media_download_timeout_seconds = 5
    dispatcher = Dispatcher()
    dispatcher.include_router(create_moderation_router(env.processor))
    dispatcher.include_router(create_control_router(env.processor))
    albums = AlbumBuffer(0.01)
    dispatcher.include_router(create_router(env.processor, albums, Mock(), env.processor.usage))
    await dispatcher.feed_update(env.bot, Update(update_id=1, message=message("/help")))
    assert all(
        "/history" not in getattr(call, "text", "")
        and "/activity" not in getattr(call, "text", "")
        for call in env.session.calls
    )
    await albums.close()
