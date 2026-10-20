from types import SimpleNamespace

import pytest
from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramNotFound,
    TelegramRetryAfter,
)
from aiogram.methods import SendMessage, SendRichMessage
from aiogram.types import Chat, InputRichMessage, Message, User
from test_moderation import env as env
from test_moderation import message as incoming_message

from bot.math_formatting import MathRichChunk
from bot.models import GeminiResult
from bot.processor import MessageProcessor, _has_attachment_input


def _message(**values: object) -> SimpleNamespace:
    attributes = {
        "reply_to_message": None,
        "external_reply": None,
    }
    attributes.update(values)
    return SimpleNamespace(**attributes)


def test_attachment_input_detects_current_media() -> None:
    assert _has_attachment_input([_message(video=object())])


def test_attachment_input_detects_replied_media() -> None:
    reply = _message(photo=[object()])
    assert _has_attachment_input([_message(reply_to_message=reply)])


def test_attachment_input_ignores_text_only_messages() -> None:
    assert not _has_attachment_input([_message(text="только текст")])


@pytest.mark.asyncio
async def test_reply_skips_unavailable_business_peer() -> None:
    class BusinessMessage:
        business_connection_id = "business-connection"
        chat = SimpleNamespace(id=42)

        async def reply(self, *args: object, **kwargs: object) -> None:
            raise TelegramBadRequest(
                method=SendMessage(chat_id=42, text="test"),
                message="Bad Request: BUSINESS_PEER_INVALID",
            )

    processor = object.__new__(MessageProcessor)
    processor.settings = SimpleNamespace(reply_chunk_size=4000)

    sent = await processor.reply(BusinessMessage(), "ответ")

    assert sent == []


class MathSession(BaseSession):
    def __init__(self, handler=None):
        super().__init__()
        self.calls = []
        self.handler = handler

    async def close(self):
        pass

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        if self.handler:
            self.handler(method, self.calls)
        return Message(
            message_id=100 + len(self.calls),
            date=0,
            chat=Chat(id=method.chat_id, type="private"),
            from_user=User(id=bot.id, is_bot=True, first_name="bot"),
            text=getattr(method, "text", "native math sent"),
        )

    async def stream_content(self, *args, **kwargs):
        yield b""


@pytest.fixture
async def math_reply():
    session = MathSession()
    bot = Bot("123456789:test-token", session=session)
    processor = object.__new__(MessageProcessor)
    processor.settings = SimpleNamespace(reply_chunk_size=80, math_render_mode="auto")
    message = Message(
        message_id=42,
        date=0,
        chat=Chat(id=42, type="private"),
        from_user=User(id=42, is_bot=False, first_name="user"),
        text="Реши уравнение",
    ).as_(bot)
    yield processor, message, session
    await bot.session.close()


async def test_math_reply_uses_native_latex_and_keeps_answer_controls(math_reply):
    processor, message, session = math_reply
    sent = await processor.reply(message, r"Ответ: $$x=\frac{\pi}{4}+3\pi n$$", controls=True)
    assert len(sent) == 1 and len(session.calls) == 1
    method = session.calls[0]
    assert isinstance(method, SendRichMessage)
    assert (
        "<tg-math-block>" in method.rich_message.html and r"\frac" in method.rich_message.html
    )
    assert method.reply_parameters.message_id == 42
    assert method.reply_parameters.allow_sending_without_reply
    assert method.reply_markup.inline_keyboard
    assert not method.allow_paid_broadcast


@pytest.mark.parametrize("mode,formatted", [("text", True), ("auto", False)])
async def test_native_math_can_be_disabled_without_touching_plain_service_messages(
    math_reply, mode, formatted
):
    processor, message, session = math_reply
    processor.settings.math_render_mode = mode
    await processor.reply(message, r"$$\frac{1}{2}$$", formatted=formatted)
    assert all(isinstance(method, SendMessage) for method in session.calls)
    assert session.calls[0].text == ("½" if formatted else r"$$\frac{1}{2}$$")


@pytest.mark.parametrize(
    "error_type", [TelegramBadRequest, TelegramNotFound, TelegramForbiddenError]
)
async def test_rich_rejection_falls_back_to_readable_text(math_reply, error_type):
    processor, message, session = math_reply

    def fail_rich(method, calls):
        if isinstance(method, SendRichMessage):
            raise error_type(method=method, message="Rich messages are not supported")

    session.handler = fail_rich
    sent = await processor.reply(message, r"Ответ: \[x=\frac{\pi}{4}\]", controls=True)
    assert len(session.calls) == 2 and len(sent) == 1
    assert isinstance(session.calls[0], SendRichMessage)
    plain = session.calls[1]
    assert isinstance(plain, SendMessage) and "x=π/4" in plain.text
    assert "\\frac" not in plain.text and "$$" not in plain.text
    assert plain.reply_markup.inline_keyboard


async def test_math_reply_respects_flood_wait_without_duplicate_plain_fallback(
    math_reply, monkeypatch
):
    processor, message, session = math_reply
    from unittest.mock import AsyncMock

    delay = AsyncMock()
    monkeypatch.setattr("bot.processor.asyncio.sleep", delay)

    def retry_once(method, calls):
        if len(calls) == 1:
            raise TelegramRetryAfter(method=method, message="Flood control", retry_after=3)

    session.handler = retry_once
    sent = await processor.reply(message, "$x^2$")
    delay.assert_awaited_once_with(3.0)
    assert len(sent) == 1 and len(session.calls) == 2
    assert all(isinstance(method, SendRichMessage) for method in session.calls)


@pytest.mark.parametrize("kind", ["network", "blocked"])
async def test_ambiguous_network_errors_and_blocked_bots_are_not_replayed(math_reply, kind):
    processor, message, session = math_reply
    error_type = TelegramNetworkError if kind == "network" else TelegramForbiddenError

    def fail(method, calls):
        raise error_type(
            method=method, message="Network down" if kind == "network" else "Bot blocked"
        )

    session.handler = fail
    with pytest.raises(error_type):
        await processor.reply(message, "$x^2$")
    assert len(session.calls) == 1


async def test_failed_later_chunk_does_not_repeat_already_sent_rich_chunks(
    math_reply, monkeypatch
):
    processor, message, session = math_reply
    chunks = [
        MathRichChunk(InputRichMessage(html="<p>first</p>"), "first text"),
        MathRichChunk(InputRichMessage(html="<p>second</p>"), "second text"),
        MathRichChunk(InputRichMessage(html="<p>third</p>"), "third text"),
    ]
    monkeypatch.setattr("bot.processor.render_math_rich_chunks", lambda source: chunks)

    def reject_second(method, calls):
        if isinstance(method, SendRichMessage) and len(calls) == 2:
            raise TelegramBadRequest(method=method, message="Invalid math expression")

    session.handler = reject_second
    sent = await processor.reply(message, "$x$", controls=True)
    assert len(sent) == 3
    assert sum(isinstance(method, SendRichMessage) for method in session.calls) == 2
    plain = [method for method in session.calls if isinstance(method, SendMessage)]
    assert [method.text for method in plain] == ["second text", "third text"]
    assert not plain[0].reply_markup and plain[1].reply_markup.inline_keyboard


async def test_native_math_preserves_topic_and_business_connection(math_reply):
    processor, message, session = math_reply
    message = message.model_copy(
        update={
            "business_connection_id": "connection",
            "is_topic_message": True,
            "message_thread_id": 17,
        }
    ).as_(message.bot)
    await processor.reply(message, "$x^2$", controls=True)
    method = session.calls[0]
    assert method.business_connection_id == "connection" and method.message_thread_id == 17
    assert method.reply_markup is None


async def test_unavailable_business_peer_does_not_trigger_math_fallback(math_reply):
    processor, message, session = math_reply
    message = message.model_copy(update={"business_connection_id": "connection"}).as_(
        message.bot
    )

    def fail(method, calls):
        raise TelegramBadRequest(method=method, message="BUSINESS_PEER_INVALID")

    session.handler = fail
    assert await processor.reply(message, "$x^2$") == []
    assert len(session.calls) == 1


async def test_partial_ai_answer_has_readable_formulas_but_keeps_original_history(env):
    original = r"Часть решения: \[x=\frac{\pi}{4}\]. Ответ сокращён."
    env.processor.gemini.generate.return_value = GeminiResult(
        original, None, provider="openrouter", truncated=True
    )
    await env.processor.process([incoming_message("Реши уравнение").as_(env.bot)])
    delivered = env.processor.reply.await_args
    assert "x=π/4" in delivered.args[1] and "\\frac" not in delivered.args[1]
    assert delivered.kwargs["formatted"] is False
    cursor = await env.storage._connection().execute(
        "SELECT assistant_content FROM conversation_exchanges"
    )
    assert (await cursor.fetchone())[0] == original
    await cursor.close()
