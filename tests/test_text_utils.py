import unittest

from bot.text_utils import remove_command, split_text, strip_bot_mention


class SplitTextTests(unittest.TestCase):
    def test_short_text_is_unchanged(self) -> None:
        self.assertEqual(split_text("Привет", 20), ["Привет"])

    def test_long_text_respects_limit(self) -> None:
        chunks = split_text("слово " * 100, 40)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 40 for chunk in chunks))
        self.assertEqual(" ".join(chunks).split(), ("слово " * 100).split())

    def test_unbroken_text_is_split(self) -> None:
        chunks = split_text("x" * 101, 25)
        self.assertEqual("".join(chunks), "x" * 101)
        self.assertTrue(all(len(chunk) <= 25 for chunk in chunks))


class CleanupTests(unittest.TestCase):
    def test_mention_is_case_insensitive(self) -> None:
        self.assertEqual(strip_bot_mention("@MyBot, привет", "mybot"), "привет")

    def test_ask_command_with_username_is_removed(self) -> None:
        self.assertEqual(
            remove_command("/ask@MyBot объясни фото", "ask", "mybot"),
            "объясни фото",
        )


if __name__ == "__main__":
    unittest.main()
