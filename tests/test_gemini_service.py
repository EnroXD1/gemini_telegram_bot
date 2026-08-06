import asyncio
import base64
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from bot.gemini import GeminiRequestError, GeminiService, ModelSelectionError
from bot.models import ConversationMessage, MediaPayload, PromptBundle


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
    def __init__(
        self, payload: dict[str, object], *, error_code: int | None = None
    ) -> None:
        self.payload = payload
        self.error_code = error_code

    def raise_for_status(self) -> None:
        if self.error_code is not None:
            raise FakeHttpError(self.error_code)

    def json(self) -> dict[str, object]:
        return self.payload


class FakeOpenRouterClient:
    def __init__(self, responses: list[FakeOpenRouterResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def post(self, url: str, *, json: dict[str, object]) -> FakeOpenRouterResponse:
        self.calls.append((url, json))
        return self.responses.pop(0)


class FakeHttpError(RuntimeError):
    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


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
        ai_max_continuations=2,
        gemini_timeout_seconds=2.0,
        gemini_retry_attempts=1,
    )
    service._client = FakeClient(responses, model_responses)
    service._openrouter_client = None
    service._groq_client = None
    service._semaphore = asyncio.Semaphore(1)
    service._interactions_available = True
    service._provider = "google"
    service._model = "gemini-test"
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
        groq_model="llama-3.1-8b-instant",
        groq_max_output_tokens=256,
    )
    service._client = None
    service._openrouter_client = FakeOpenRouterClient(
        [
            FakeOpenRouterResponse(
                {"choices": [{"message": {"content": response_text}}]}
            )
        ]
    )
    service._groq_client = None
    service._semaphore = asyncio.Semaphore(1)
    service._interactions_available = False
    service._provider = "openrouter"
    service._model = "google/gemini-3.5-flash"
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
        self.assertEqual(result.provider, "openrouter")
        url, payload = service._openrouter_client.calls[0]
        self.assertEqual(url, "chat/completions")
        self.assertEqual(payload["model"], "google/gemini-3.5-flash")
        self.assertEqual(payload["messages"][1]["content"], "вопрос")

    async def test_openrouter_continues_response_stopped_by_token_limit(self) -> None:
        service = make_openrouter_service("unused")
        service._openrouter_client = FakeOpenRouterClient(
            [
                FakeOpenRouterResponse(
                    {
                        "model": "free/model-a",
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {"content": "Начало подробного ответа."},
                            }
                        ],
                    }
                ),
                FakeOpenRouterResponse(
                    {
                        "model": "free/model-b",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": "Окончание ответа."},
                            }
                        ],
                    }
                ),
            ]
        )

        result = await service.generate(PromptBundle(prompt="реши всё"), None)

        self.assertEqual(
            result.text, "Начало подробного ответа.\n\nОкончание ответа."
        )
        self.assertFalse(result.truncated)
        self.assertEqual(len(service._openrouter_client.calls), 2)
        continuation_messages = service._openrouter_client.calls[1][1]["messages"]
        self.assertEqual(continuation_messages[-2]["role"], "assistant")
        self.assertEqual(
            continuation_messages[-2]["content"], "Начало подробного ответа."
        )
        self.assertIn("Не повторяй", continuation_messages[-1]["content"])

    async def test_openrouter_marks_response_after_continuations_are_exhausted(
        self,
    ) -> None:
        service = make_openrouter_service("unused")
        service._settings.ai_max_continuations = 1
        service._openrouter_client = FakeOpenRouterClient(
            [
                FakeOpenRouterResponse(
                    {
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {"content": "Первая часть"},
                            }
                        ]
                    }
                ),
                FakeOpenRouterResponse(
                    {
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {"content": "Вторая часть"},
                            }
                        ]
                    }
                ),
            ]
        )

        result = await service.generate(PromptBundle(prompt="длинный ответ"), None)

        self.assertTrue(result.truncated)
        self.assertIn("Первая часть\n\nВторая часть", result.text)
        self.assertIn("достиг лимита", result.text)
        self.assertEqual(len(service._openrouter_client.calls), 2)

    async def test_openrouter_continues_partial_provider_error(self) -> None:
        service = make_openrouter_service("unused")
        service._settings.ai_max_continuations = 1
        service._openrouter_client = FakeOpenRouterClient(
            [
                FakeOpenRouterResponse(
                    {
                        "choices": [
                            {
                                "finish_reason": "provider_error",
                                "message": {"content": "Уцелевшая часть"},
                            }
                        ]
                    }
                ),
                FakeOpenRouterResponse(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": "Продолжение"},
                            }
                        ]
                    }
                ),
            ]
        )

        result = await service.generate(PromptBundle(prompt="вопрос"), None)

        self.assertEqual(result.text, "Уцелевшая часть\n\nПродолжение")
        self.assertFalse(result.truncated)

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

    async def test_openrouter_includes_local_history(self) -> None:
        service = make_openrouter_service("новый ответ")
        history = (
            ConversationMessage(role="user", content="старый вопрос"),
            ConversationMessage(role="assistant", content="старый ответ"),
        )

        await service.generate(
            PromptBundle(prompt="новый вопрос"), None, history=history
        )

        messages = service._openrouter_client.calls[0][1]["messages"]
        self.assertEqual(
            [item["role"] for item in messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(messages[1]["content"], "старый вопрос")
        self.assertEqual(messages[2]["content"], "старый ответ")

    async def test_openrouter_limit_switches_text_request_to_groq(self) -> None:
        service = make_openrouter_service("unused")
        service._openrouter_client = FakeOpenRouterClient(
            [FakeOpenRouterResponse({}, error_code=429)]
        )
        service._groq_client = FakeOpenRouterClient(
            [
                FakeOpenRouterResponse(
                    {"choices": [{"message": {"content": "ответ Groq"}}]}
                )
            ]
        )
        switched_to: list[str] = []

        async def on_fallback(model: str) -> None:
            switched_to.append(model)

        result = await service.generate(
            PromptBundle(prompt="вопрос"), None, on_fallback=on_fallback
        )

        self.assertEqual(result.text, "ответ Groq")
        self.assertEqual(result.provider, "groq")
        self.assertEqual(switched_to, ["llama-3.1-8b-instant"])
        _, payload = service._groq_client.calls[0]
        self.assertEqual(payload["model"], "llama-3.1-8b-instant")
        self.assertEqual(payload["max_completion_tokens"], 256)

    async def test_owner_selection_can_use_groq_as_primary(self) -> None:
        service = make_openrouter_service("unused")
        service._groq_client = FakeOpenRouterClient(
            [
                FakeOpenRouterResponse(
                    {"choices": [{"message": {"content": "прямой ответ Groq"}}]}
                )
            ]
        )
        service._settings.groq_model = "llama-3.1-8b-instant"

        provider, model = service.select_model("groq")
        result = await service.generate(PromptBundle(prompt="вопрос"), None)

        self.assertEqual((provider, model), ("groq", "llama-3.1-8b-instant"))
        self.assertEqual(result.text, "прямой ответ Groq")
        self.assertEqual(result.provider, "groq")
        _, payload = service._groq_client.calls[0]
        self.assertEqual(payload["model"], "llama-3.1-8b-instant")

    async def test_groq_primary_continues_truncated_response(self) -> None:
        service = make_openrouter_service("unused")
        service._groq_client = FakeOpenRouterClient(
            [
                FakeOpenRouterResponse(
                    {
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {"content": "Часть Groq"},
                            }
                        ]
                    }
                ),
                FakeOpenRouterResponse(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": "Финал Groq"},
                            }
                        ]
                    }
                ),
            ]
        )
        service._settings.groq_model = "llama-3.1-8b-instant"
        service.select_model("groq")

        result = await service.generate(PromptBundle(prompt="вопрос"), None)

        self.assertEqual(result.text, "Часть Groq\n\nФинал Groq")
        self.assertFalse(result.truncated)
        self.assertEqual(len(service._groq_client.calls), 2)

    async def test_unconfigured_provider_cannot_be_selected(self) -> None:
        service = make_openrouter_service("unused")

        with self.assertRaises(ModelSelectionError):
            service.select_model("groq")

    async def test_multimodal_limit_does_not_use_text_only_groq(self) -> None:
        service = make_openrouter_service("unused")
        service._openrouter_client = FakeOpenRouterClient(
            [FakeOpenRouterResponse({}, error_code=429)]
        )
        service._groq_client = FakeOpenRouterClient(
            [
                FakeOpenRouterResponse(
                    {"choices": [{"message": {"content": "не должен вызываться"}}]}
                )
            ]
        )
        bundle = PromptBundle(
            prompt="опиши",
            media=(MediaPayload("фото", "image", "image/jpeg", b"image"),),
        )

        with self.assertRaises(GeminiRequestError) as raised:
            await service.generate(bundle, None)

        self.assertEqual(service._groq_client.calls, [])
        self.assertEqual(raised.exception.delete_after_seconds, 20.0)

    async def test_openrouter_limit_error_is_marked_for_auto_deletion(self) -> None:
        service = make_openrouter_service("unused")
        service._openrouter_client = FakeOpenRouterClient(
            [FakeOpenRouterResponse({}, error_code=429)]
        )

        with self.assertRaises(GeminiRequestError) as raised:
            await service.generate(PromptBundle(prompt="вопрос"), None)

        self.assertEqual(raised.exception.delete_after_seconds, 20.0)

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
        self.assertEqual(result.provider, "google")
        self.assertEqual(result.interaction_id, "new-id")
        call = service._client.aio.interactions.calls[0]
        self.assertEqual(call["previous_interaction_id"], "old-id")
        self.assertEqual(call["model"], "gemini-test")
        request_input = call["input"]
        self.assertEqual(request_input[1]["type"], "image")
        self.assertEqual(
            base64.b64decode(request_input[1]["data"]), b"image-bytes"
        )

    async def test_google_interaction_continues_incomplete_response(self) -> None:
        service = make_service(
            [
                SimpleNamespace(
                    id="partial-id",
                    output_text="Первая часть Gemini",
                    status="incomplete",
                ),
                SimpleNamespace(
                    id="final-id",
                    output_text="Финал Gemini",
                    status="completed",
                ),
            ]
        )

        result = await service.generate(PromptBundle(prompt="реши всё"), None)

        self.assertEqual(
            result.text, "Первая часть Gemini\n\nФинал Gemini"
        )
        self.assertEqual(result.interaction_id, "final-id")
        self.assertFalse(result.truncated)
        calls = service._client.aio.interactions.calls
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["previous_interaction_id"], "partial-id")
        self.assertIn("Не повторяй", calls[1]["input"])

    async def test_google_generate_content_continues_max_tokens(self) -> None:
        service = make_service(
            [],
            [
                SimpleNamespace(
                    text="Первая часть generateContent",
                    candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
                ),
                SimpleNamespace(
                    text="Финал generateContent",
                    candidates=[SimpleNamespace(finish_reason="STOP")],
                ),
            ],
        )
        service._interactions_available = False

        result = await service.generate(PromptBundle(prompt="реши всё"), None)

        self.assertEqual(
            result.text,
            "Первая часть generateContent\n\nФинал generateContent",
        )
        self.assertFalse(result.truncated)
        calls = service._client.aio.models.calls
        self.assertEqual(len(calls), 2)
        continuation_contents = calls[1]["contents"]
        self.assertEqual(
            [content.role for content in continuation_contents],
            ["user", "model", "user"],
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
