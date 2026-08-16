import pytest

from bot.handlers import SERVICE_EMOJI_GUIDE, _first_command_argument


def test_service_emoji_guide_contains_all_owner_actions() -> None:
    assert "/emoji set ✅ 123456789" in SERVICE_EMOJI_GUIDE
    assert "/emoji test ⚡" in SERVICE_EMOJI_GUIDE
    assert "/emoji reset 🔄" in SERVICE_EMOJI_GUIDE
    assert "/emoji remove ✅" in SERVICE_EMOJI_GUIDE
    assert "/emoji defaults" in SERVICE_EMOJI_GUIDE
    assert "/emoji clear" in SERVICE_EMOJI_GUIDE
    assert "любой эмодзи" in SERVICE_EMOJI_GUIDE
    assert "отдельным сообщением" in SERVICE_EMOJI_GUIDE


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        (None, ""),
        ("", ""),
        ("   ", ""),
        ("9", "9"),
        ("  12   ignored", "12"),
    ],
)
def test_first_command_argument_handles_empty_wall_command(
    argument: str | None, expected: str
) -> None:
    assert _first_command_argument(argument) == expected
