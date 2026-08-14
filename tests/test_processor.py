from types import SimpleNamespace

from bot.processor import _has_attachment_input


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
