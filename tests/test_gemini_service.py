import asyncio
import base64
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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


class FakeModels:
    def __init__(self, responses: list[object] | None = None) -> None:
        self.responses = responses or []
        self.calls: list[dict[str, object]] = []

    async def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(
        self,
        interaction_responses: list[object],
        model_responses: list[object] | None = None,
    ) -> None:
        self.aio = SimpleNamespace(
            interactions=FakeInteractions(interaction_responses),
            models=FakeModels(model_responses),
        )


class FakeOpenRouterResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


class FakeOpenRouterClient:
    def __init__(self, responses: list[FakeOpenRouterResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def post(self, url: str, *, json: dict[str, object]) -> FakeOpenRouterResponse:
        self.calls.append((url, json))
        return self.responses.pop(0)


class ContextError(RuntimeError):
    code = 404


class LocationError(RuntimeError):
    code = 400


def make_service(
    responses: list[object], model_responses: list[object] | None = None
) -> GeminiService:
    service = GeminiService.__new__(GeminiService)
    service._settings = SimpleNamespace(
        ai_provider="google",
        gemini_store_interactions=True,
        gemini_model="gemini-test",
        gemini_system_prompt="system",
        gemini_temperature=0.5,
        gemini_max_output_tokens=100,
        gemini_timeout_seconds=2.0,
        gemini_retry_attempts=1,
    )
    service._client = FakeClient(responses, model_responses)
    service._openrouter_client = None
    service._semaphore = asyncio.Semaphore(1)
    service._interactions_available = True
    return service


def make_openrouter_service(response_text: str) -> GeminiService:
    service = GeminiService.__new__(GeminiService)
    service._settings = SimpleNamespace(
        ai_provider="openrouter",
        openrouter_model="google/gemini-3.5-flash",
        gemini_system_prompt="system",
        gemini_temperature=0.5,
        gemini_max_output_tokens=100,
        gemini_timeout_seconds=2.0,
        gemini_retry_attempts=1,
    )
    service._client = None
    service._openrouter_client = FakeOpenRouterClient(
        [
            FakeOpenRouterResponse(
                {"choices": [{"message": {"content": response_text}}]}
            )
        ]
    )
    service._semaphore = asyncio.Semaphore(1)
    service._interactions_available = False
    return service


class GeminiServiceTests(unittest.IsolatedAsyncioTestCase):
    @patch("bot.gemini.genai.Client")
    def test_vertex_express_client(self, client_factory) -> None:
        settings = SimpleNamespace(
            ai_provider="google",
            gemini_api_key="AQ.test-key",
            gemini_vertex_ai=True,
            max_concurrent_requests=1,
        )

        service = GeminiService(settings)

        client_factory.assert_called_once_with(
            api_key="AQ.test-key", vertexai=True
        )
        self.assertFalse(service._interactions_available)

    async def test_openrouter_text_request(self) -> None:
        service = make_openrouter_service("ответ OpenRouter")

        result = await service.generate(PromptBundle(prompt="вопрос"), None)

        self.assertEqual(result.text, "ответ OpenRouter")
        self.assertIsNone(result.interaction_id)
        url, payload = service._openrouter_client.calls[0]
        self.assertEqual(url, "chat/completions")
        self.assertEqual(payload["model"], "google/gemini-3.5-flash")
        self.assertEqual(payload["messages"][1]["content"], "вопрос")

    async def test_openrouter_multimodal_request(self) -> None:
        service = make_openrouter_service("мультимодальный ответ")
        bundle = PromptBundle(
            prompt="проанализируй всё",
            media=(
                MediaPayload("фото", "image", "image/jpeg", b"image"),
                MediaPayload("голос", "audio", "audio/ogg", b"audio"),
                MediaPayload("кружок", "video", "video/mp4", b"video"),
                MediaPayload("PDF", "document", "application/pdf", b"pdf"),
            ),
        )

        result = await service.generate(bundle, "old-google-context")

        self.assertEqual(result.text, "мультимодальный ответ")
        self.assertTrue(result.context_was_reset)
        content = service._openrouter_client.calls[0][1]["messages"][1]["content"]
        typed_parts = {part["type"]: part for part in content if part["type"] != "text"}
        self.assertTrue(
            typed_parts["image_url"]["image_url"]["url"].startswith(
                "data:image/jpeg;base64,"
            )
        )
        self.assertEqual(typed_parts["input_audio"]["input_audio"]["format"], "ogg")
        self.assertTrue(
            typed_parts["video_url"]["video_url"]["url"].startswith(
                "data:video/mp4;base64,"
            )
        )
        self.assertTrue(
            typed_parts["file"]["file"]["file_data"].startswith(
                "data:application/pdf;base64,"
            )
        )

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

    async def test_location_error_falls_back_to_generate_content(self) -> None:
        service = make_service(
            [LocationError("This API is not available in your current location")],
            [SimpleNamespace(text="резервный ответ")],
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

        self.assertEqual(result.text, "резервный ответ")
        self.assertIsNone(result.interaction_id)
        self.assertTrue(result.context_was_reset)
        self.assertFalse(service._interactions_available)
        call = service._client.aio.models.calls[0]
        parts = call["contents"]
        self.assertEqual(parts[1].inline_data.data, b"image-bytes")
        self.assertEqual(parts[1].inline_data.mime_type, "image/jpeg")

    async def test_generate_content_stays_enabled_after_location_error(self) -> None:
        service = make_service(
            [LocationError("Interactions API is not available")],
            [SimpleNamespace(text="первый"), SimpleNamespace(text="второй")],
        )

        first = await service.generate(PromptBundle(prompt="раз"), None)
        second = await service.generate(PromptBundle(prompt="два"), None)

        self.assertEqual(first.text, "первый")
        self.assertEqual(second.text, "второй")
        self.assertEqual(len(service._client.aio.interactions.calls), 1)
        self.assertEqual(len(service._client.aio.models.calls), 2)


if __name__ == "__main__":
    unittest.main()
