import asyncio

import pytest
from aiogram import Dispatcher, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import InlineQuery, Update, User
from test_moderation import env as env
from test_moderation import message

from bot.calculator import CalculationError, calculate
from bot.gemini import GeminiRequestError
from bot.inline import InlineService, create_inline_router
from bot.models import GeminiResult
from bot.moderation import BanMiddleware, create_moderation_router, users_view
from bot.rate_limit import SlidingWindowRateLimiter
from bot.usage import UsageTracker


@pytest.fixture
async def inline_env(env):
    original = env.session.make_request

    async def request(bot, method, timeout=None):
        if method.__api_method__ == "answerInlineQuery":
            env.session.calls.append(method)
            return True
        return await original(bot, method, timeout)

    env.session.make_request = request
    env.processor.gemini.current_model = "test/model:free"
    env.processor.gemini.free_only = True
    env.processor.usage = UsageTracker(
        bot=env.bot,
        settings=env.processor.settings,
        storage=env.storage,
    )
    env.processor._rate_limiter = SlidingWindowRateLimiter(100, 60)
    service = InlineService(env.processor)
    service.debounce_seconds = 0
    dispatcher = Dispatcher()
    dispatcher.message.outer_middleware(BanMiddleware(env.storage, {42, 43}))
    dispatcher.include_router(create_moderation_router(env.processor))
    dispatcher.include_router(create_inline_router(service))
    fallback = Router()
    other = []

    @fallback.message()
    async def collect(item):
        other.append(item)

    dispatcher.include_router(fallback)

    def query(text, *, user_id=99, query_id="q", **values):
        return InlineQuery(
            id=query_id,
            from_user=User(id=user_id, is_bot=False, first_name="User", username="same_name"),
            query=text,
            offset=values.pop("offset", ""),
            **values,
        ).as_(env.bot)

    async def ask(text, **values):
        item = query(text, **values)
        await dispatcher.feed_update(env.bot, Update(update_id=1, inline_query=item))

    def answers():
        return [
            call for call in env.session.calls if call.__api_method__ == "answerInlineQuery"
        ]

    env.service, env.dispatcher = service, dispatcher
    env.query, env.ask, env.answers, env.other = query, ask, answers, other
    yield env
    await service.close()


@pytest.mark.parametrize(
    "expression,result",
    [
        ("(15+5)*3", "60"),
        ("0.1+0.2", "0.3"),
        ("2^10", "1024"),
        ("2,5 × 4", "10"),
        ("−5 // 2", "-3"),
        ("-5 % 2", "1"),
        ("1e3 / 4", "250"),
        ("2 ** -3", "0.125"),
        ("1_000+2", "1002"),
    ],
)
def test_local_calculator(expression, result):
    assert calculate(expression) == result


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').getcwd()",
        "open('data/bot.sqlite3')",
        "x + 1",
        "True + 1",
        "(1).__class__",
        "[1,2]",
        "2<<999999",
        "9**9**9",
        "1e99999999",
        "0**-1",
        "5/0",
        "2 ** 0.5",
        "1+" * 200,
        "9" * 200,
        "1+" * 40 + "1",
    ],
)
def test_calculator_rejects_unsafe_or_unbounded_input(expression):
    with pytest.raises(CalculationError):
        calculate(expression)


@pytest.mark.parametrize("text", ["calc (15+5)*3", "посчитай (15+5)*3", "(15+5)*3"])
async def test_inline_calc_has_preview_and_never_calls_ai(inline_env, text):
    env = inline_env
    await env.ask(text, chat_type="channel")
    reply = env.answers()[-1]
    assert reply.results[0].input_message_content.message_text == "60"
    assert reply.cache_time == 0 and reply.is_personal and reply.next_offset == ""
    assert len(reply.results[0].id.encode()) <= 64
    env.processor.gemini.generate.assert_not_awaited()
    events = await env.storage.list_user_events(99)
    assert events[0].kind == "command" and events[0].outcome == "completed"


@pytest.mark.parametrize(
    "text,marker",
    [
        ("tr en Привет!", "Переведи"),
        ("исправь Превет", "Исправь"),
        ("Что такое HTTP?", "Кратко ответь"),
    ],
)
async def test_inline_ai_uses_isolated_prompt_without_chat_context(inline_env, text, marker):
    env = inline_env
    await env.ask(text)
    args = env.processor.gemini.generate.await_args
    assert marker in args.args[0].prompt
    assert args.args[1] is None and args.kwargs["history"] == ()
    assert env.answers()[-1].results[0].input_message_content.message_text == "Ответ"
    assert env.answers()[-1].results[0].input_message_content.parse_mode is None


@pytest.mark.parametrize("text", ["", "  ", "tr", "fix", "calc"])
async def test_empty_or_incomplete_query_only_offers_private_help(inline_env, text):
    env = inline_env
    await env.ask(text)
    assert env.answers()[-1].results == []
    assert env.answers()[-1].button.start_parameter == "inline"
    env.processor.gemini.generate.assert_not_awaited()
    assert await env.storage.list_user_events(99) == []


@pytest.mark.parametrize("user_id", [99, 100])
async def test_banned_user_cannot_generate_or_receive_cached_results(inline_env, user_id):
    env = inline_env
    await env.ask("fix Текст", user_id=user_id)
    await env.storage.ban_user(user_id, 42)
    env.processor.gemini.generate.reset_mock()
    await env.ask("fix Текст", user_id=user_id)
    await env.ask("calc 2+2", user_id=user_id)
    assert all(not response.results for response in env.answers()[-2:])
    env.processor.gemini.generate.assert_not_awaited()


async def test_acl_checks_user_id_not_username_or_destination_type(inline_env):
    env = inline_env
    env.processor.settings.inline_allowed_user_ids = frozenset({99})
    await env.ask("calc 2+2", user_id=100, chat_type="sender")
    assert not env.answers()[-1].results
    await env.ask("calc 2+2", user_id=99, chat_type="channel")
    assert env.answers()[-1].results
    await env.ask("calc 2+2", user_id=42)
    assert env.answers()[-1].results  # owners always allowed unless globally disabled
    env.processor.settings.inline_enabled = False
    await env.ask("calc 2+2", user_id=42)
    assert not env.answers()[-1].results


async def test_allowed_chats_do_not_silently_allow_inline_from_any_user(inline_env):
    env = inline_env
    env.processor.settings.allowed_chat_ids = frozenset({-100})
    await env.ask("calc 2+2")
    assert not env.answers()[-1].results
    env.processor.settings.inline_allowed_user_ids = frozenset({99})
    await env.ask("calc 2+2")
    assert env.answers()[-1].results


async def test_cache_is_personal_and_invalidated_by_model_selection(inline_env):
    env = inline_env
    await env.ask("fix Текст")
    await env.ask("fix Текст")
    assert env.processor.gemini.generate.await_count == 1
    assert len(await env.storage.list_user_events(99)) == 1
    await env.ask("fix Текст", user_id=100)
    assert env.processor.gemini.generate.await_count == 2
    env.processor.gemini.current_model = "new/model:free"
    await env.ask("fix Текст")
    assert env.processor.gemini.generate.await_count == 3


async def test_typing_debounce_only_generates_latest_query(inline_env):
    env = inline_env
    env.service.debounce_seconds = 0.02
    task = asyncio.create_task(env.service.handle(env.query("fix П", query_id="old")))
    await asyncio.sleep(0)
    await env.service.handle(env.query("fix Привет", query_id="new"))
    await task
    assert env.processor.gemini.generate.await_count == 1
    assert [answer.inline_query_id for answer in env.answers()] == ["new"]
    assert not env.service._jobs and not env.service._latest


async def test_duplicate_inflight_query_does_not_generate_twice(inline_env):
    env = inline_env
    started, release = asyncio.Event(), asyncio.Event()

    async def generate(*args, **kwargs):
        started.set()
        await release.wait()
        return GeminiResult("Исправлено", None)

    env.processor.gemini.generate.side_effect = generate
    first = asyncio.create_task(env.service.handle(env.query("fix Текст", query_id="old")))
    await asyncio.wait_for(started.wait(), 2)
    second = asyncio.create_task(env.service.handle(env.query("fix Текст", query_id="new")))
    await asyncio.sleep(0)
    release.set()
    await asyncio.wait_for(asyncio.gather(first, second), 2)
    assert env.processor.gemini.generate.await_count == 1
    assert [answer.inline_query_id for answer in env.answers()] == ["new"]


async def test_owner_ban_cancels_active_inline_ai_and_sends_no_result(inline_env):
    env = inline_env
    started = asyncio.Event()

    async def generate(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    env.processor.gemini.generate.side_effect = generate
    task = asyncio.create_task(env.service.handle(env.query("fix Текст")))
    await asyncio.wait_for(started.wait(), 2)
    await env.dispatcher.feed_update(
        env.bot, Update(update_id=2, message=message("/ban 99", user_id=42))
    )
    await asyncio.wait_for(task, 2)
    assert env.answers() == []
    assert not env.processor._active_users and not env.processor._active_tasks
    assert (await env.storage.list_user_events(99))[0].outcome == "cancelled"


@pytest.mark.parametrize("reason", ["busy", "rate_limit"])
async def test_inline_ai_shares_antispam_and_records_reason(inline_env, reason):
    env = inline_env
    if reason == "busy":
        env.processor._active_users.add("99")
    else:
        env.processor._rate_limiter = SlidingWindowRateLimiter(1, 60)
        env.processor._rate_limiter.check("99")
    await env.ask("fix Текст")
    assert not env.answers()[-1].results
    env.processor.gemini.generate.assert_not_awaited()
    assert (await env.storage.list_user_events(99))[0].reason == reason


async def test_running_inline_ai_prevents_parallel_normal_ai_request(inline_env):
    env = inline_env
    started = asyncio.Event()

    async def generate(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    env.processor.gemini.generate.side_effect = generate
    task = asyncio.create_task(env.service.handle(env.query("fix Текст")))
    await asyncio.wait_for(started.wait(), 2)
    await env.processor.process([message("Вопрос")])
    assert env.processor.gemini.generate.await_count == 1
    assert (await env.storage.list_user_events(99))[0].reason == "busy"
    await env.service.close()
    await task


async def test_inline_timeout_releases_guard_and_does_not_publish_error(inline_env):
    env = inline_env
    env.service.ai_timeout_seconds = 0.01
    env.processor.gemini.generate.side_effect = lambda *args, **kwargs: None

    async def generate(*args, **kwargs):
        await asyncio.Event().wait()

    env.processor.gemini.generate.side_effect = generate
    await env.ask("fix Текст")
    assert not env.answers()[-1].results
    assert not env.processor._active_users and not env.processor._active_tasks
    assert (await env.storage.list_user_events(99))[0].outcome == "error"


async def test_provider_error_and_invalid_math_never_become_postable_articles(inline_env):
    env = inline_env
    env.processor.gemini.generate.side_effect = GeminiRequestError("SECRET RESPONSE with key")
    await env.ask("fix Текст")
    await env.ask("calc __import__('os')")
    for response in env.answers():
        assert not response.results and "SECRET" not in response.button.text


async def test_inline_activity_has_unknown_destination_and_owner_exclusion(inline_env):
    env = inline_env
    await env.ask("fix Текст", chat_type="channel")
    await env.ask("calc 2+2", user_id=42)
    record = await env.storage.get_bot_user(99)
    assert record.last_chat_type == "inline" and record.last_chat_id == 0
    stats = await env.storage.get_bot_usage_stats()
    assert stats.inline_users == stats.total_users == 1
    assert not stats.private_users and not stats.group_users
    assert (await env.storage.list_user_events(99))[0].chat_id == 0
    assert await env.storage.list_user_events(42) == []
    text, _ = await users_view(env.storage)
    assert "inline (чат назначения неизвестен)" in text


async def test_plain_text_utf16_limit_and_cache_bounds(inline_env):
    env = inline_env
    env.service.max_cache = 2
    env.processor.gemini.generate.return_value = GeminiResult("<b>" + "😀" * 5000, None)
    for text in ("fix Один", "fix Два", "fix Три"):
        await env.ask(text)
    body = env.answers()[-1].results[0].input_message_content
    assert body.message_text.startswith("<b>") and body.parse_mode is None
    assert len(body.message_text.encode("utf-16-le")) // 2 <= 3800
    assert len(env.service._cache) == 2


async def test_pagination_and_oversized_queries_do_not_call_ai(inline_env):
    env = inline_env
    await env.ask("fix Текст", offset="unexpected")
    await env.ask("fix " + "a" * 300)
    env.processor.gemini.generate.assert_not_awaited()
    assert all(not answer.results for answer in env.answers())


async def test_inline_deep_link_help_preserves_other_start_parameters(inline_env):
    env = inline_env
    for text in ("/inline", "/start inline"):
        await env.dispatcher.feed_update(env.bot, Update(update_id=2, message=message(text)))
        assert "@testbot tr en" in env.session.calls[-1].text
    await env.dispatcher.feed_update(
        env.bot, Update(update_id=2, message=message("/start other"))
    )
    assert env.other[-1].text == "/start other"
    assert "inline_query" in env.dispatcher.resolve_used_update_types()


async def test_expired_inline_query_is_handled_without_retrying_generation(inline_env):
    env = inline_env

    async def expired(bot, method, timeout=None):
        raise TelegramBadRequest(method=method, message="query is too old")

    env.session.make_request = expired
    # Owner avoids the best-effort new-user notification in this test.
    await env.ask("calc 2+2", user_id=42)


async def test_service_close_cancels_pending_workers_and_prevents_more_ai(inline_env):
    env = inline_env
    env.service.debounce_seconds = 1
    task = asyncio.create_task(env.service.handle(env.query("fix Текст")))
    await asyncio.sleep(0)
    await env.service.close()
    await task
    await env.ask("fix Новый текст")
    env.processor.gemini.generate.assert_not_awaited()
    assert not env.service._workers and not env.service._jobs and not env.service._cache


async def test_new_query_cancels_active_generation_and_only_new_preview_survives(inline_env):
    env = inline_env
    started = asyncio.Event()

    async def generate(bundle, *args, **kwargs):
        if "Старый" in bundle.prompt:
            started.set()
            await asyncio.Event().wait()
        return GeminiResult("Новый результат", None)

    env.processor.gemini.generate.side_effect = generate
    first = asyncio.create_task(env.service.handle(env.query("fix Старый", query_id="old")))
    await asyncio.wait_for(started.wait(), 2)
    await env.service.handle(env.query("fix Новый", query_id="new"))
    await asyncio.wait_for(first, 2)
    assert [answer.inline_query_id for answer in env.answers()] == ["new"]
    assert env.answers()[0].results[0].input_message_content.message_text == "Новый результат"
    assert not env.processor._active_users and not env.processor._active_tasks
    assert {event.outcome for event in await env.storage.list_user_events(99)} == {
        "completed",
        "cancelled",
    }


async def test_cache_expiration_and_bounded_pending_workers(inline_env, monkeypatch):
    from types import SimpleNamespace

    env = inline_env
    clock = [100.0]
    monkeypatch.setattr("bot.inline.time", SimpleNamespace(monotonic=lambda: clock[0]))
    await env.ask("fix Один")
    clock[0] += env.service.cache_seconds + 1
    await env.ask("fix Один")
    assert env.processor.gemini.generate.await_count == 2
    env.service.max_jobs = 1
    started = asyncio.Event()

    async def generate(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    env.processor.gemini.generate.side_effect = generate
    first = asyncio.create_task(env.service.handle(env.query("fix Два")))
    await asyncio.wait_for(started.wait(), 2)
    await env.ask("fix Три", user_id=100)
    assert not env.answers()[-1].results and "Бот занят" in env.answers()[-1].button.text
    assert len(env.service._workers) == 1
    await env.service.close()
    await first


async def test_legacy_owner_ban_never_locks_owner_out_of_inline(inline_env):
    env = inline_env
    await env.storage.ban_user(42, 43)
    await env.ask("calc 2+2", user_id=42)
    assert env.answers()[-1].results
    assert await env.storage.get_bot_user(42) is None


async def test_slow_access_check_for_old_query_cannot_cancel_new_query(inline_env):
    env = inline_env
    started, release = asyncio.Event(), asyncio.Event()
    original = env.service._denied

    async def delayed(query):
        if query.id == "old":
            started.set()
            await release.wait()
        return await original(query)

    env.service._denied = delayed
    first = asyncio.create_task(env.service.handle(env.query("fix Старый", query_id="old")))
    await asyncio.wait_for(started.wait(), 2)
    await env.service.handle(env.query("fix Новый", query_id="new"))
    release.set()
    await first
    assert env.processor.gemini.generate.await_count == 1
    assert [response.inline_query_id for response in env.answers()] == ["new"]
