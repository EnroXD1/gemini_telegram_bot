import unittest

from bot.scope import build_scope_key


class ScopeTests(unittest.TestCase):
    def test_group_thread_is_shared_by_default(self) -> None:
        first = build_scope_key(
            chat_id=-1001,
            thread_id=42,
            user_id=1,
            chat_type="supergroup",
            scope_mode="chat_thread",
        )
        second = build_scope_key(
            chat_id=-1001,
            thread_id=42,
            user_id=2,
            chat_type="supergroup",
            scope_mode="chat_thread",
        )
        self.assertEqual(first, second)

    def test_private_scope_includes_user(self) -> None:
        key = build_scope_key(
            chat_id=10,
            thread_id=None,
            user_id=10,
            chat_type="private",
            scope_mode="chat_thread",
        )
        self.assertIn(":user:10", key)

    def test_business_connection_is_isolated(self) -> None:
        regular = build_scope_key(
            chat_id=10,
            thread_id=None,
            user_id=10,
            chat_type="private",
            scope_mode="chat_thread",
        )
        business = build_scope_key(
            chat_id=10,
            thread_id=None,
            user_id=10,
            chat_type="private",
            scope_mode="chat_thread",
            business_connection_id="connection-1",
        )
        self.assertNotEqual(regular, business)


if __name__ == "__main__":
    unittest.main()
