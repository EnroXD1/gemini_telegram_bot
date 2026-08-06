import unittest
from types import SimpleNamespace

from bot.gemini_response import extract_interaction_id, extract_interaction_text


class GeminiResponseTests(unittest.TestCase):
    def test_convenience_property(self) -> None:
        response = SimpleNamespace(id="abc", output_text=" готово ")
        self.assertEqual(extract_interaction_text(response), "готово")
        self.assertEqual(extract_interaction_id(response), "abc")

    def test_steps_shape(self) -> None:
        response = {
            "id": "xyz",
            "steps": [
                {"type": "thought", "content": []},
                {
                    "type": "model_output",
                    "content": [
                        {"type": "text", "text": "часть 1"},
                        {"type": "text", "text": "часть 2"},
                    ],
                },
            ],
        }
        self.assertEqual(extract_interaction_text(response), "часть 1\nчасть 2")

    def test_empty_response(self) -> None:
        self.assertEqual(extract_interaction_text({"steps": []}), "")
        self.assertIsNone(extract_interaction_id({}))


if __name__ == "__main__":
    unittest.main()
