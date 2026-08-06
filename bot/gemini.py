from __future__ import annotations

import asyncio
import base64
import logging
import random
from typing import Any

import httpx
from google import genai
from google.genai import types

from .config import Settings
from .gemini_response import extract_interaction_id, extract_interaction_text
from .models import ConversationMessage, GeminiResult, PromptBundle

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/"
OPENROUTER_APP_URL = "https://github.com/EnroXD1/gemini_telegram_bot"


class GeminiRequestError(RuntimeError):
    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class GeminiService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        self._client: Any | None = None
        self._openrouter_client: httpx.AsyncClient | None = None

        if settings.ai_provider == "openrouter":
            self._openrouter_client = httpx.AsyncClient(
                base_url=OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "HTTP-Referer": OPENROUTER_APP_URL,
                    "X-Title": "Gemini Telegram Business Bot",
                },
                timeout=settings.gemini_timeout_seconds,
            )
            self._interactions_available = False
        else:
            client_kwargs: dict[str, Any] = {"api_key": settings.gemini_api_key}
            if settings.gemini_vertex_ai:
                client_kwargs["vertexai"] = True
            self._client = genai.Client(**client_kwargs)
            self._interactions_available = not settings.gemini_vertex_ai

    async def close(self) -> None:
        if self._openrouter_client is not None:
            await self._openrouter_client.aclose()
        if self._client is not None:
            await self._client.aio.aclose()

    async def generate(
        self,
        bundle: PromptBundle,
        previous_interaction_id: str | None,
        history: tuple[ConversationMessage, ...] = (),
    ) -> GeminiResult:
        if self._settings.ai_provider == "openrouter":
            return await self._generate_openrouter(
                bundle, previous_interaction_id, history
            )

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

    async def _generate_openrouter(
        self,
        bundle: PromptBundle,
        previous_interaction_id: str | None,
        history: tuple[ConversationMessage, ...],
    ) -> GeminiResult:
        client = self._openrouter_client
        if client is None:
            raise RuntimeError("OpenRouter client is not initialized")

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self._settings.gemini_system_prompt,
            }
        ]
        messages.extend(
            {"role": item.role, "content": item.content} for item in history
        )
        messages.append(
            {
                "role": "user",
                "content": _build_openrouter_content(bundle),
            }
        )

        payload = {
            "model": self._settings.openrouter_model,
            "messages": messages,
            "temperature": self._settings.gemini_temperature,
            "max_tokens": self._settings.gemini_max_output_tokens,
            "stream": False,
        }
        attempt = 0

        async with self._semaphore:
            while True:
                try:
                    async with asyncio.timeout(
                        self._settings.gemini_timeout_seconds
                    ):
                        response = await client.post(
                            "chat/completions", json=payload
                        )
                        response.raise_for_status()
                    text = _extract_openrouter_text(response.json())
                    if not text:
                        raise GeminiRequestError(
                            "Модель обработала запрос через OpenRouter, но не вернула "
                            "текстовый ответ. Попробуйте переформулировать сообщение."
                        )
                    return GeminiResult(
                        text=text,
                        interaction_id=None,
                        context_was_reset=bool(previous_interaction_id),
                    )
                except GeminiRequestError:
                    raise
                except TimeoutError as exc:
                    raise GeminiRequestError(
                        "OpenRouter не успел получить ответ модели за отведённое "
                        "время. Попробуйте ещё раз или отправьте файл меньшего размера."
                    ) from exc
                except Exception as exc:
                    code = _error_code(exc)
                    attempt += 1
                    if (
                        _is_retryable(exc, code)
                        and attempt < self._settings.gemini_retry_attempts
                    ):
                        delay = min(8.0, (2 ** (attempt - 1)) + random.random())
                        logger.warning(
                            "Retryable OpenRouter error code=%s attempt=%s type=%s",
                            code,
                            attempt,
                            type(exc).__name__,
                        )
                        await asyncio.sleep(delay)
                        continue

                    logger.exception(
                        "OpenRouter request failed code=%s type=%s",
                        code,
                        type(exc).__name__,
                    )
                    raise GeminiRequestError(
                        _friendly_openrouter_error(code)
                    ) from exc


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


def _build_openrouter_content(bundle: PromptBundle) -> str | list[dict[str, Any]]:
    if not bundle.media:
        return bundle.prompt

    content: list[dict[str, Any]] = []
    for item in bundle.media:
        content.append({"type": "text", "text": item.label})
        encoded = base64.b64encode(item.data).decode("ascii")
        if item.kind == "image":
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{item.mime_type};base64,{encoded}"
                    },
                }
            )
        elif item.kind == "document":
            content.append(
                {
                    "type": "file",
                    "file": {
                        "filename": "document.pdf",
                        "file_data": f"data:{item.mime_type};base64,{encoded}",
                    },
                }
            )
        elif item.kind == "audio":
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded,
                        "format": _audio_format(item.mime_type),
                    },
                }
            )
        elif item.kind == "video":
            content.append(
                {
                    "type": "video_url",
                    "video_url": {
                        "url": f"data:{item.mime_type};base64,{encoded}"
                    },
                }
            )
        else:
            raise GeminiRequestError(
                f"OpenRouter не поддерживает тип вложения {item.kind!r}."
            )
    content.append({"type": "text", "text": bundle.prompt})
    return content


def _audio_format(mime_type: str) -> str:
    normalized = mime_type.lower().split(";", 1)[0].strip()
    formats = {
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/aiff": "aiff",
        "audio/x-aiff": "aiff",
        "audio/aac": "aac",
        "audio/ogg": "ogg",
        "audio/opus": "opus",
        "audio/flac": "flac",
        "audio/mp4": "m4a",
        "audio/x-m4a": "m4a",
        "audio/webm": "webm",
    }
    return formats.get(normalized, normalized.partition("/")[2] or "wav")


def _extract_openrouter_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = [
        str(part.get("text", "")).strip()
        for part in content
        if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
    ]
    return "\n".join(part for part in parts if part).strip()


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


def _friendly_openrouter_error(code: int | None) -> str:
    if code == 400:
        return (
            "OpenRouter не смог обработать запрос. Проверьте формат вложения "
            "или попробуйте отправить его отдельно с пояснением."
        )
    if code in {401, 403}:
        return (
            "OpenRouter отклонил ключ или доступ к модели. Владельцу бота нужно "
            "проверить OPENROUTER_API_KEY и настройки ключа."
        )
    if code == 402:
        return (
            "На балансе OpenRouter недостаточно средств для этого запроса. "
            "Владельцу бота нужно пополнить баланс."
        )
    if code == 429:
        return "Лимит запросов OpenRouter временно исчерпан. Попробуйте немного позже."
    if code in {500, 502, 503, 504}:
        return "OpenRouter или выбранный поставщик временно недоступен. Попробуйте позже."
    return "Не удалось получить ответ через OpenRouter. Попробуйте ещё раз позднее."
