from types import SimpleNamespace

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import SendMessage

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
