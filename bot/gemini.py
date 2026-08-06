from __future__ import annotations

import asyncio
import base64
import logging
import random
from typing import Any

from google import genai
from google.genai import types

from .config import Settings
from .gemini_response import extract_interaction_id, extract_interaction_text
from .models import GeminiResult, PromptBundle

logger = logging.getLogger(__name__)


class GeminiRequestError(RuntimeError):
    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class GeminiService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        client_kwargs: dict[str, Any] = {"api_key": settings.gemini_api_key}
        if settings.gemini_vertex_ai:
            client_kwargs["vertexai"] = True
        self._client = genai.Client(**client_kwargs)
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        self._interactions_available = not settings.gemini_vertex_ai

    async def close(self) -> None:
        await self._client.aio.aclose()

    async def generate(
        self, bundle: PromptBundle, previous_interaction_id: str | None
    ) -> GeminiResult:
        request_input = _build_input(bundle)
        current_previous_id = (
            previous_interaction_id
            if self._settings.gemini_store_interactions
            else None
        )
        context_was_reset = bool(
            previous_interaction_id and not self._settings.gemini_store_interactions
        )
        attempt = 0

        async with self._semaphore:
            while True:
                try:
                    kwargs: dict[str, Any] = {
                        "model": self._settings.gemini_model,
                        "input": request_input,
                        "system_instruction": self._settings.gemini_system_prompt,
                        "generation_config": {
                            "temperature": self._settings.gemini_temperature,
                            "max_output_tokens": self._settings.gemini_max_output_tokens,
                        },
                        "store": self._settings.gemini_store_interactions,
                    }
                    if current_previous_id:
                        kwargs["previous_interaction_id"] = current_previous_id

                    async with asyncio.timeout(
                        self._settings.gemini_timeout_seconds
                    ):
                        if self._interactions_available:
                            interaction = await self._client.aio.interactions.create(
                                **kwargs
                            )
                            text = extract_interaction_text(interaction)
                            interaction_id = (
                                extract_interaction_id(interaction)
                                if self._settings.gemini_store_interactions
                                else None
                            )
                        else:
                            response = await self._client.aio.models.generate_content(
                                model=self._settings.gemini_model,
                                contents=_build_generate_content(bundle),
                                config=types.GenerateContentConfig(
                                    system_instruction=(
                                        self._settings.gemini_system_prompt
                                    ),
                                    temperature=self._settings.gemini_temperature,
                                    max_output_tokens=(
                                        self._settings.gemini_max_output_tokens
                                    ),
                                ),
                            )
                            text = str(getattr(response, "text", "") or "").strip()
                            interaction_id = None

                    if not text:
                        raise GeminiRequestError(
                            "Gemini обработал запрос, но не вернул текстовый ответ. "
                            "Попробуйте переформулировать сообщение."
                        )
                    return GeminiResult(
                        text=text,
                        interaction_id=interaction_id,
                        context_was_reset=context_was_reset,
                    )
                except GeminiRequestError:
                    raise
                except TimeoutError as exc:
                    raise GeminiRequestError(
                        "Gemini не успел ответить за отведённое время. "
                        "Попробуйте ещё раз или отправьте файл меньшего размера."
                    ) from exc
                except Exception as exc:
                    code = _error_code(exc)
                    if self._interactions_available and _is_location_error(exc):
                        logger.warning(
                            "Gemini Interactions API is unavailable in this region; "
                            "switching to generateContent"
                        )
                        self._interactions_available = False
                        current_previous_id = None
                        context_was_reset = context_was_reset or bool(
                            previous_interaction_id
                        )
                        attempt = 0
                        continue
                    if current_previous_id and _is_expired_context_error(exc, code):
                        logger.info("Gemini conversation context expired; starting a new one")
                        current_previous_id = None
                        context_was_reset = True
                        continue

                    attempt += 1
                    if (
                        _is_retryable(exc, code)
                        and attempt < self._settings.gemini_retry_attempts
                    ):
                        delay = min(8.0, (2 ** (attempt - 1)) + random.random())
                        logger.warning(
                            "Retryable Gemini error code=%s attempt=%s type=%s",
                            code,
                            attempt,
                            type(exc).__name__,
                        )
                        await asyncio.sleep(delay)
                        continue

                    logger.exception(
                        "Gemini request failed code=%s type=%s",
                        code,
                        type(exc).__name__,
                    )
                    raise GeminiRequestError(_friendly_error(code)) from exc


def _build_input(bundle: PromptBundle) -> str | list[dict[str, str]]:
    if not bundle.media:
        return bundle.prompt

    content: list[dict[str, str]] = []
    for item in bundle.media:
        content.append({"type": "text", "text": item.label})
        content.append(
            {
                "type": item.kind,
                "data": base64.b64encode(item.data).decode("ascii"),
                "mime_type": item.mime_type,
            }
        )
    content.append({"type": "text", "text": bundle.prompt})
    return content


def _build_generate_content(bundle: PromptBundle) -> str | list[types.Part]:
    if not bundle.media:
        return bundle.prompt

    parts: list[types.Part] = []
    for item in bundle.media:
        parts.append(types.Part.from_text(text=item.label))
        parts.append(types.Part.from_bytes(data=item.data, mime_type=item.mime_type))
    parts.append(types.Part.from_text(text=bundle.prompt))
    return parts


def _error_code(exc: Exception) -> int | None:
    for name in ("code", "status_code"):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            return value
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _is_expired_context_error(exc: Exception, code: int | None) -> bool:
    if code not in {400, 404}:
        return False
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "previous_interaction",
            "previous interaction",
            "interaction not found",
            "interaction expired",
            "not found",
        )
    )


def _is_location_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "api is not available in your current location" in text
        or "interactions api is not available" in text
    )


def _is_retryable(exc: Exception, code: int | None) -> bool:
    if code in {408, 429, 500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "temporarily unavailable",
            "connection reset",
            "connection error",
            "rate limit",
            "resource exhausted",
        )
    )


def _friendly_error(code: int | None) -> str:
    if code == 429:
        return "Лимит запросов Gemini временно исчерпан. Попробуйте немного позже."
    if code in {401, 403}:
        return (
            "Gemini отклонил ключ API. Владельцу бота нужно проверить "
            "GEMINI_API_KEY и доступ выбранной модели."
        )
    if code == 400:
        return (
            "Gemini не смог обработать этот запрос. Проверьте формат вложения "
            "или попробуйте отправить его отдельно с пояснением."
        )
    if code in {500, 502, 503, 504}:
        return "Сервис Gemini временно недоступен. Попробуйте ещё раз позднее."
    return "Не удалось получить ответ от Gemini. Попробуйте ещё раз позднее."
