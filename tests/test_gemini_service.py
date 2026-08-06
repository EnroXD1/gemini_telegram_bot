import asyncio
import base64
import unittest
from types import SimpleNamespace

from bot.gemini import GeminiService
from bot.models import MediaPayload, PromptBundle


class FakeInteractions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.aio = SimpleNamespace(interactions=FakeInteractions(responses))


class ContextError(RuntimeError):
    code = 404


def make_service(responses: list[object]) -> GeminiService:
    service = GeminiService.__new__(GeminiService)
    service._settings = SimpleNamespace(
        gemini_store_interactions=True,
        gemini_model="gemini-test",
        gemini_system_prompt="system",
        gemini_temperature=0.5,
        gemini_max_output_tokens=100,
        gemini_timeout_seconds=2.0,
        gemini_retry_attempts=1,
    )
    service._client = FakeClient(responses)
    service._semaphore = asyncio.Semaphore(1)
    return service


class GeminiServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_multimodal_request(self) -> None:
        service = make_service(
            [SimpleNamespace(id="new-id", output_text="ответ")]
        )
        bundle = PromptBundle(
            prompt="вопрос",
            media=(
                MediaPayload(
                    label="фото",
                    kind="image",
                    mime_type="image/jpeg",
                    data=b"image-bytes",
                ),
            ),
        )

        result = await service.generate(bundle, "old-id")

        self.assertEqual(result.text, "ответ")
        self.assertEqual(result.interaction_id, "new-id")
        call = service._client.aio.interactions.calls[0]
        self.assertEqual(call["previous_interaction_id"], "old-id")
        self.assertEqual(call["model"], "gemini-test")
        request_input = call["input"]
        self.assertEqual(request_input[1]["type"], "image")
        self.assertEqual(
            base64.b64decode(request_input[1]["data"]), b"image-bytes"
        )

    async def test_expired_context_is_retried_without_previous_id(self) -> None:
        service = make_service(
            [
                ContextError("previous_interaction_id not found"),
                SimpleNamespace(id="fresh", output_text="новый ответ"),
            ]
        )

        result = await service.generate(PromptBundle(prompt="вопрос"), "expired")

        self.assertTrue(result.context_was_reset)
        calls = service._client.aio.interactions.calls
        self.assertEqual(calls[0]["previous_interaction_id"], "expired")
        self.assertNotIn("previous_interaction_id", calls[1])


if __name__ == "__main__":
    unittest.main()
