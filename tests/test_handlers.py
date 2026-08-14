from bot.handlers import SERVICE_EMOJI_GUIDE


def test_service_emoji_guide_contains_all_owner_actions() -> None:
    assert "/emoji set ✅ 123456789" in SERVICE_EMOJI_GUIDE
    assert "/emoji test ⚡" in SERVICE_EMOJI_GUIDE
    assert "/emoji reset 🔄" in SERVICE_EMOJI_GUIDE
    assert "/emoji remove ✅" in SERVICE_EMOJI_GUIDE
    assert "/emoji defaults" in SERVICE_EMOJI_GUIDE
    assert "/emoji clear" in SERVICE_EMOJI_GUIDE
    assert "любой эмодзи" in SERVICE_EMOJI_GUIDE
    assert "отдельным сообщением" in SERVICE_EMOJI_GUIDE
