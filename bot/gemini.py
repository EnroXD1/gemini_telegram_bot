from __future__ import annotations

import asyncio
import base64
import gc
import json
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
from google import genai
from google.genai import types

from .config import (
    DEFAULT_OPENROUTER_FALLBACK_MODELS,
    Settings,
)
from .gemini_response import (
    extract_interaction_id,
    extract_interaction_status,
    extract_interaction_text,
)
from .models import ConversationMessage, GeminiResult, PromptBundle

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/"
OPENROUTER_APP_URL = "https://github.com/EnroXD1/gemini_telegram_bot"
GROQ_BASE_URL = "https://api.groq.com/openai/v1/"
_OPENROUTER_MEDIA_UNSUPPORTED_MODELS = frozenset(
    DEFAULT_OPENROUTER_FALLBACK_MODELS[:2]
)


@dataclass(frozen=True, slots=True)
class ModelSwitchNotice:
    source_provider: str
    source_model: str
    target_provider: str
    target_model: str
    reason: str


@dataclass(frozen=True, slots=True)
class _ModelRoute:
    provider: str
    model: str


FallbackNotifier = Callable[[ModelSwitchNotice], Awaitable[None]]

_INCOMPLETE_FINISH_REASONS = {
    "length",
    "max_tokens",
    "max_output_tokens",
    "token_limit",
    "error",
    "provider_error",
}
_INCOMPLETE_INTERACTION_STATUSES = {"incomplete", "budget_exceeded"}
_BLOCKED_FINISH_REASONS = {
    "blocked",
    "blocklist",
    "content_filter",
    "content-filter",
    "content_filtered",
    "image_prohibited_content",
    "image_recitation",
    "image_safety",
    "prohibited_content",
    "recitation",
    "safety",
    "safety_blocked",
    "spii",
}
_NON_CHAT_MODEL_MARKERS = (
    "content-safety",
    "content_safety",
    "moderation",
    "guardrail",
    "guardrails",
    "classifier",
)
_SAFETY_REPORT_MARKERS = (
    "user safety:",
    "response safety:",
    "safety categories:",
)
_REQUEST_BLOCK_MARKERS = (
    "request blocked",
    "blocked request",
    "prompt blocked",
    "content filter",
    "content_filter",
    "safety policy",
    "policy violation",
)
_BLOCKED_RESPONSE_MESSAGES = frozenset(
    {
        "blocked request",
        "content filter",
        "content filtered",
        "content filtered by provider",
        "content filtered by safety policy",
        "content_filter",
        "policy violation",
        "prompt blocked",
        "prompt blocked by provider",
        "prompt blocked by safety policy",
        "request blocked",
        "request blocked by provider",
        "request blocked by provider safety policy",
        "safety policy violation",
    }
)
_CONTINUATION_PROMPT = (
    "Продолжи предыдущий ответ точно с места остановки. Не повторяй уже "
    "написанное и не начинай решение заново. Верни только продолжение. "
    "Заверши все начатые списки, формулы, блоки кода и Markdown-разметку."
)
_TRUNCATED_NOTICE = (
    "⚠️ Ответ всё ещё достиг лимита выбранной модели после автоматических "
    "продолжений. Отправьте «продолжи с места остановки», если требуется ещё."
)


@dataclass(frozen=True, slots=True)
class _ChatCompletion:
    text: str
    finish_reason: str | None
    model: str | None


@dataclass(frozen=True, slots=True)
class _Base64Data:
    data: bytes


class _OpenRouterMediaBody:
    """Re-iterable JSON body that encodes raw media in bounded chunks."""

    _RAW_CHUNK_BYTES = 48 * 1024

    def __init__(self, segments: list[bytes | _Base64Data]) -> None:
        self._segments = tuple(segments)
        self.content_length = sum(
            len(segment)
            if isinstance(segment, bytes)
            else 4 * ((len(segment.data) + 2) // 3)
            for segment in self._segments
        )

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[bytes]:
        for segment in self._segments:
            if isinstance(segment, bytes):
                yield segment
                continue
            view = memoryview(segment.data)
            for offset in range(0, len(view), self._RAW_CHUNK_BYTES):
                yield base64.b64encode(
                    view[offset : offset + self._RAW_CHUNK_BYTES]
                )


class GeminiRequestError(RuntimeError):
    def __init__(
        self,
        user_message: str,
        *,
        delete_after_seconds: float | None = None,
        provider: str | None = None,
        status_code: int | None = None,
        fallback_allowed: bool = False,
        quota_exhausted: bool = False,
        fallback_reason: str | None = None,
    ) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.delete_after_seconds = delete_after_seconds
        self.provider = provider
        self.status_code = status_code
        self.fallback_allowed = fallback_allowed
        self.quota_exhausted = quota_exhausted
        self.fallback_reason = fallback_reason


class ModelSelectionError(ValueError):
    """Raised when an owner selects an unavailable provider or invalid model."""


class GeminiService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        self._media_semaphore = asyncio.Semaphore(1)
        self._client: Any | None = None
        self._openrouter_client: httpx.AsyncClient | None = None
        self._groq_client: httpx.AsyncClient | None = None
        self._provider = settings.ai_provider
        if self._provider == "openrouter":
            self._model = getattr(settings, "openrouter_model", "openrouter/free")
        elif self._provider == "groq":
            self._model = getattr(settings, "groq_model", "llama-3.1-8b-instant")
        else:
            self._model = getattr(settings, "gemini_model", "gemini-3.6-flash")
        self._interactions_available = False
        self._route_cooldowns: dict[tuple[str, str], float] = {}

        if getattr(settings, "openrouter_api_key", ""):
            self._openrouter_client = httpx.AsyncClient(
                base_url=OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "HTTP-Referer": OPENROUTER_APP_URL,
                    "X-Title": "Gemini Telegram Business Bot",
                },
                timeout=settings.gemini_timeout_seconds,
            )
        if getattr(settings, "groq_api_key", ""):
            self._groq_client = httpx.AsyncClient(
                base_url=GROQ_BASE_URL,
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                timeout=settings.gemini_timeout_seconds,
            )
        if getattr(settings, "gemini_api_key", ""):
            client_kwargs: dict[str, Any] = {
                "api_key": settings.gemini_api_key,
                "http_options": types.HttpOptions(
                    # The GAOS transport mutates attempts=0 into one retry.
                    # One attempt plus an unused status code disables both its
                    # retry layers; explicit bot retries remain in this module.
                    retry_options=types.HttpRetryOptions(
                        attempts=1,
                        http_status_codes=[418],
                    )
                ),
            }
            if settings.gemini_vertex_ai:
                client_kwargs["vertexai"] = True
            self._client = genai.Client(**client_kwargs)
            self._interactions_available = not settings.gemini_vertex_ai

    @property
    def current_provider(self) -> str:
        return self._provider

    @property
    def current_model(self) -> str:
        return self._model

    @property
    def uses_local_history(self) -> bool:
        return self._provider in {"openrouter", "groq"}

    def configured_providers(self) -> dict[str, str]:
        providers: dict[str, str] = {}
        if self._client is not None:
            providers["google"] = self._settings.gemini_model
        if self._openrouter_client is not None:
            providers["openrouter"] = self._settings.openrouter_model
        if self._groq_client is not None:
            providers["groq"] = self._settings.groq_model
        return providers

    def select_model(self, provider: str, model: str | None = None) -> tuple[str, str]:
        aliases = {
            "gemini": "google",
            "google": "google",
            "or": "openrouter",
            "openrouter": "openrouter",
            "groq": "groq",
        }
        normalized = aliases.get(provider.strip().lower())
        available = self.configured_providers()
        if normalized is None:
            raise ModelSelectionError(
                "Неизвестный провайдер. Используйте google, openrouter или groq."
            )
        if normalized not in available:
            raise ModelSelectionError(
                f"Провайдер {normalized} не настроен: для него нет API-ключа."
            )
        selected_model = (model or available[normalized]).strip()
        if not selected_model or len(selected_model) > 200:
            raise ModelSelectionError("ID модели должен содержать от 1 до 200 символов.")
        if any(ord(character) < 32 for character in selected_model):
            raise ModelSelectionError("ID модели содержит недопустимые символы.")
        self._provider = normalized
        self._model = selected_model
        cooldowns = getattr(self, "_route_cooldowns", None)
        if cooldowns is not None:
            cooldowns.pop((normalized, selected_model), None)
        return normalized, selected_model

    async def close(self) -> None:
        if self._openrouter_client is not None:
            await self._openrouter_client.aclose()
        if self._groq_client is not None:
            await self._groq_client.aclose()
        if self._client is not None:
            await self._client.aio.aclose()

    async def generate(
        self,
        bundle: PromptBundle,
        previous_interaction_id: str | None,
        history: tuple[ConversationMessage, ...] = (),
        on_fallback: FallbackNotifier | None = None,
    ) -> GeminiResult:
        routes = self._candidate_routes(bundle)
        automatic_fallback = getattr(
            self._settings, "ai_auto_fallback_enabled", True
        )
        last_error: GeminiRequestError | None = None
        failed_route: _ModelRoute | None = None
        notify_switch = False
        attempted = 0

        for route in routes:
            if self._route_is_cooling_down(route):
                if failed_route is None:
                    failed_route = route
                    last_error = GeminiRequestError(
                        "Лимит модели недавно был исчерпан.",
                        provider=route.provider,
                        fallback_allowed=True,
                        quota_exhausted=True,
                    )
                continue
            if failed_route is not None and last_error is not None:
                if notify_switch:
                    await self._notify_model_switch(
                        on_fallback,
                        failed_route,
                        route,
                        reason=_fallback_reason(last_error),
                    )
                else:
                    logger.info(
                        "Using AI fallback silently while route cools down "
                        "%s/%s -> %s/%s",
                        failed_route.provider,
                        failed_route.model,
                        route.provider,
                        route.model,
                    )
            try:
                attempted += 1
                logger.info(
                    "Starting AI route provider=%s model=%s media_items=%s "
                    "media_bytes=%s",
                    route.provider,
                    route.model,
                    len(bundle.media),
                    _bundle_media_bytes(bundle),
                )
                if bundle.media:
                    media_semaphore = getattr(self, "_media_semaphore", None)
                    if media_semaphore is None:
                        media_semaphore = asyncio.Semaphore(1)
                        self._media_semaphore = media_semaphore
                    async with media_semaphore:
                        return await self._generate_route(
                            route,
                            bundle,
                            previous_interaction_id,
                            history,
                            is_fallback=failed_route is not None,
                        )
                return await self._generate_route(
                    route,
                    bundle,
                    previous_interaction_id,
                    history,
                    is_fallback=failed_route is not None,
                )
            except GeminiRequestError as exc:
                failed_route = route
                notify_switch = True
                if not automatic_fallback or not exc.fallback_allowed:
                    raise
                last_error = _detached_request_error(exc)
                self._start_route_cooldown(route, last_error)
                if bundle.media:
                    _release_exception_chain(exc)
                    gc.collect()

        if last_error is not None and attempted > 1:
            raise GeminiRequestError(
                "Все доступные модели временно недоступны, исчерпали лимит или "
                "не смогли обработать этот запрос. "
                "Попробуйте немного позже.",
                delete_after_seconds=20.0,
                fallback_allowed=False,
                quota_exhausted=last_error.quota_exhausted,
            ) from last_error
        if last_error is not None:
            raise last_error
        raise GeminiRequestError(
            "Для этого запроса нет доступной AI-модели. Владельцу бота нужно "
            "проверить настроенные API-ключи."
        )

    def fallback_routes(self) -> tuple[tuple[str, str], ...]:
        routes = self._candidate_routes(PromptBundle(prompt=""))
        return tuple((route.provider, route.model) for route in routes[1:])

    def _candidate_routes(self, bundle: PromptBundle) -> list[_ModelRoute]:
        primary = _ModelRoute(self._provider, self._model)
        routes = [primary]
        if not getattr(self._settings, "ai_auto_fallback_enabled", True):
            return routes

        available = self.configured_providers()

        def add(provider: str, model: str) -> None:
            route = _ModelRoute(provider, model)
            if route in routes:
                return
            if provider == "groq" and bundle.media:
                return
            if (
                provider == "openrouter"
                and bundle.media
                and not _openrouter_model_supports_media(model)
            ):
                return
            if (
                provider == "groq"
                and primary.provider != "groq"
                and not getattr(self._settings, "groq_fallback_enabled", True)
            ):
                return
            routes.append(route)

        default_model = available.get(primary.provider)
        if default_model:
            add(primary.provider, default_model)
        for model in self._fallback_models_for(primary.provider):
            add(primary.provider, model)

        provider_order = {
            "google": ("openrouter", "groq"),
            "openrouter": ("groq", "google"),
            "groq": ("openrouter", "google"),
        }.get(primary.provider, ("openrouter", "google", "groq"))
        for provider in provider_order:
            model = available.get(provider)
            if model:
                add(provider, model)
                for fallback_model in self._fallback_models_for(provider):
                    add(provider, fallback_model)
        return routes

    def _fallback_models_for(self, provider: str) -> tuple[str, ...]:
        if provider == "openrouter" and self._openrouter_client is not None:
            return tuple(
                getattr(self._settings, "openrouter_fallback_models", ())
            )
        if provider == "groq" and self._groq_client is not None:
            return tuple(getattr(self._settings, "groq_fallback_models", ()))
        return ()

    async def _generate_route(
        self,
        route: _ModelRoute,
        bundle: PromptBundle,
        previous_interaction_id: str | None,
        history: tuple[ConversationMessage, ...],
        *,
        is_fallback: bool,
    ) -> GeminiResult:
        if route.provider == "openrouter":
            if bundle.media and not _openrouter_model_supports_media(route.model):
                raise GeminiRequestError(
                    "Динамическая или текстовая модель OpenRouter не подходит "
                    "для обработки медиа. Пробую другой маршрут.",
                    provider="openrouter",
                    fallback_allowed=True,
                )
            return await self._generate_openrouter(
                bundle,
                previous_interaction_id,
                history,
                model=route.model,
            )
        if route.provider == "groq":
            if bundle.media:
                raise GeminiRequestError(
                    "Модель Groq работает только с текстом.",
                    provider="groq",
                    fallback_allowed=True,
                )
            messages = self._build_chat_messages(bundle, history)
            async with self._semaphore:
                return await self._generate_groq(
                    messages,
                    previous_interaction_id,
                    model=route.model,
                    is_fallback=is_fallback,
                )
        return await self._generate_google(
            bundle,
            previous_interaction_id,
            model=route.model,
        )

    def _route_is_cooling_down(self, route: _ModelRoute) -> bool:
        cooldowns = getattr(self, "_route_cooldowns", None)
        if cooldowns is None:
            cooldowns = {}
            self._route_cooldowns = cooldowns
        now = time.monotonic()
        for key in ((route.provider, route.model), (route.provider, "*")):
            until = cooldowns.get(key, 0.0)
            if until > now:
                return True
            cooldowns.pop(key, None)
        return False

    def _start_route_cooldown(
        self, route: _ModelRoute, error: GeminiRequestError
    ) -> None:
        if not error.quota_exhausted:
            return
        seconds = float(
            getattr(self._settings, "ai_fallback_cooldown_seconds", 600.0)
        )
        provider_wide = error.status_code == 402 and route.provider != "openrouter"
        key = (
            (route.provider, "*")
            if provider_wide
            else (route.provider, route.model)
        )
        self._route_cooldowns[key] = time.monotonic() + max(1.0, seconds)

    async def _notify_model_switch(
        self,
        notifier: FallbackNotifier | None,
        source: _ModelRoute,
        target: _ModelRoute,
        *,
        reason: str,
    ) -> None:
        logger.warning(
            "Switching AI route %s/%s -> %s/%s reason=%s",
            source.provider,
            source.model,
            target.provider,
            target.model,
            reason,
        )
        if notifier is None:
            return
        try:
            await notifier(
                ModelSwitchNotice(
                    source_provider=source.provider,
                    source_model=source.model,
                    target_provider=target.provider,
                    target_model=target.model,
                    reason=reason,
                )
            )
        except Exception:
            logger.exception("Could not show automatic model switch status")

    async def _generate_google(
        self,
        bundle: PromptBundle,
        previous_interaction_id: str | None,
        *,
        model: str,
    ) -> GeminiResult:

        base_request_input: Any | None = None
        request_input: Any | None = None
        generate_contents: Any | None = None
        if self._interactions_available:
            base_request_input = _build_input(bundle)
            request_input = base_request_input
        else:
            generate_contents = _build_generate_content(bundle)
        current_previous_id = (
            previous_interaction_id
            if self._settings.gemini_store_interactions
            else None
        )
        context_was_reset = bool(
            previous_interaction_id and not self._settings.gemini_store_interactions
        )
        combined_text = ""
        continuation_count = 0
        interaction_id: str | None = None
        attempt = 0

        async with self._semaphore:
            while True:
                try:
                    async with asyncio.timeout(
                        self._settings.gemini_timeout_seconds
                    ):
                        if self._interactions_available:
                            kwargs: dict[str, Any] = {
                                "model": model,
                                "input": request_input,
                                "system_instruction": (
                                    self._settings.gemini_system_prompt
                                ),
                                "generation_config": {
                                    "temperature": (
                                        self._settings.gemini_temperature
                                    ),
                                    "max_output_tokens": (
                                        self._settings.gemini_max_output_tokens
                                    ),
                                },
                                "store": self._settings.gemini_store_interactions,
                            }
                            if current_previous_id:
                                kwargs["previous_interaction_id"] = (
                                    current_previous_id
                                )
                            interaction = await self._client.aio.interactions.create(
                                **kwargs
                            )
                            text = extract_interaction_text(interaction)
                            interaction_id = (
                                extract_interaction_id(interaction)
                                if self._settings.gemini_store_interactions
                                else None
                            )
                            finish_reason = extract_interaction_status(interaction)
                            truncated = (
                                finish_reason in _INCOMPLETE_INTERACTION_STATUSES
                            )
                        else:
                            if generate_contents is None:
                                generate_contents = _build_generate_content(bundle)
                            response = await self._client.aio.models.generate_content(
                                model=model,
                                contents=generate_contents,
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
                            finish_reason = _extract_google_finish_reason(response)
                            truncated = _needs_continuation(finish_reason)

                    if _is_blocked_finish_reason(finish_reason):
                        raise GeminiRequestError(
                            "Gemini заблокировал этот запрос правилами безопасности. "
                            "Пробую другую модель.",
                            provider="google",
                            fallback_allowed=True,
                            fallback_reason="blocked",
                        )
                    if not text:
                        raise GeminiRequestError(
                            "Gemini обработал запрос, но не вернул текстовый ответ. "
                            "Попробуйте переформулировать сообщение.",
                            provider="google",
                            fallback_allowed=True,
                        )
                    combined_text = _merge_continuation(combined_text, text)
                    logger.info(
                        "Gemini completion model=%s finish_reason=%s "
                        "continuation=%s",
                        model,
                        finish_reason,
                        continuation_count,
                    )
                    if truncated and continuation_count < _max_continuations(
                        self._settings
                    ):
                        continuation_count += 1
                        if self._interactions_available:
                            if interaction_id:
                                current_previous_id = interaction_id
                                request_input = _CONTINUATION_PROMPT
                            else:
                                current_previous_id = None
                                request_input = _standalone_continuation_input(
                                    combined_text
                                )
                        else:
                            generate_contents = _build_generate_content_continuation(
                                bundle, combined_text
                            )
                        attempt = 0
                        logger.info(
                            "Continuing truncated Gemini response part=%s",
                            continuation_count + 1,
                        )
                        continue
                    return GeminiResult(
                        text=_finish_text(combined_text, truncated),
                        interaction_id=interaction_id,
                        context_was_reset=context_was_reset,
                        provider="google",
                        truncated=truncated,
                    )
                except GeminiRequestError:
                    if combined_text:
                        return GeminiResult(
                            text=_finish_text(combined_text, True),
                            interaction_id=interaction_id,
                            context_was_reset=context_was_reset,
                            provider="google",
                            truncated=True,
                        )
                    raise
                except TimeoutError as exc:
                    if combined_text:
                        logger.warning(
                            "Gemini continuation timed out after partial response"
                        )
                        return GeminiResult(
                            text=_finish_text(combined_text, True),
                            interaction_id=interaction_id,
                            context_was_reset=context_was_reset,
                            provider="google",
                            truncated=True,
                        )
                    raise GeminiRequestError(
                        "Gemini не успел ответить за отведённое время. "
                        "Попробуйте ещё раз или отправьте файл меньшего размера.",
                        provider="google",
                        status_code=408,
                        fallback_allowed=True,
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
                        base_request_input = None
                        request_input = None
                        context_was_reset = context_was_reset or bool(
                            previous_interaction_id
                        )
                        if combined_text:
                            generate_contents = _build_generate_content_continuation(
                                bundle, combined_text
                            )
                        else:
                            generate_contents = _build_generate_content(bundle)
                        attempt = 0
                        continue
                    if current_previous_id and _is_expired_context_error(exc, code):
                        logger.info("Gemini conversation context expired; starting a new one")
                        current_previous_id = None
                        context_was_reset = True
                        request_input = (
                            _standalone_continuation_input(combined_text)
                            if combined_text
                            else base_request_input
                        )
                        continue

                    attempt += 1
                    if (
                        _is_retryable(exc, code)
                        and not _is_quota_error(exc, code)
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
                    if combined_text:
                        return GeminiResult(
                            text=_finish_text(combined_text, True),
                            interaction_id=interaction_id,
                            context_was_reset=context_was_reset,
                            provider="google",
                            truncated=True,
                        )
                    raise GeminiRequestError(
                        _friendly_error(code),
                        delete_after_seconds=20.0 if code == 429 else None,
                        provider="google",
                        status_code=code,
                        fallback_allowed=_allows_automatic_fallback(exc, code),
                        quota_exhausted=_is_quota_error(exc, code),
                        fallback_reason=(
                            "blocked" if _is_safety_block_error(exc) else None
                        ),
                    ) from exc

    async def _generate_openrouter(
        self,
        bundle: PromptBundle,
        previous_interaction_id: str | None,
        history: tuple[ConversationMessage, ...],
        *,
        model: str,
    ) -> GeminiResult:
        client = self._openrouter_client
        if client is None:
            raise RuntimeError("OpenRouter client is not initialized")

        has_media = bool(bundle.media)
        base_messages = (
            None if has_media else self._build_chat_messages(bundle, history)
        )
        request_messages = base_messages
        combined_text = ""
        continuation_count = 0
        attempt = 0

        async with self._semaphore:
            while True:
                try:
                    async with asyncio.timeout(
                        self._settings.gemini_timeout_seconds
                    ):
                        if has_media:
                            body = _build_openrouter_media_body(
                                bundle,
                                history,
                                system_instruction=(
                                    self._settings.gemini_system_prompt
                                ),
                                model=model,
                                temperature=self._settings.gemini_temperature,
                                max_tokens=(
                                    self._settings.gemini_max_output_tokens
                                ),
                                combined_text=combined_text,
                            )
                            response = await client.post(
                                "chat/completions",
                                content=body,
                                headers={
                                    "Content-Type": "application/json",
                                    "Content-Length": str(body.content_length),
                                },
                            )
                        else:
                            payload = {
                                "model": model,
                                "messages": request_messages,
                                "temperature": self._settings.gemini_temperature,
                                "max_tokens": (
                                    self._settings.gemini_max_output_tokens
                                ),
                                "stream": False,
                            }
                            response = await client.post(
                                "chat/completions", json=payload
                            )
                        response.raise_for_status()
                    completion = _extract_chat_completion(response.json())
                    _raise_if_unusable_chat_completion(
                        completion,
                        provider="openrouter",
                        requested_model=model,
                    )
                    if not completion.text:
                        raise GeminiRequestError(
                            "Модель обработала запрос через OpenRouter, но не вернула "
                            "текстовый ответ. Попробуйте переформулировать сообщение.",
                            provider="openrouter",
                            fallback_allowed=True,
                        )
                    combined_text = _merge_continuation(
                        combined_text, completion.text
                    )
                    truncated = _needs_continuation(completion.finish_reason)
                    logger.info(
                        "OpenRouter completion model=%s finish_reason=%s "
                        "continuation=%s",
                        completion.model or model,
                        completion.finish_reason,
                        continuation_count,
                    )
                    if truncated and continuation_count < _max_continuations(
                        self._settings
                    ):
                        continuation_count += 1
                        if base_messages is not None:
                            request_messages = _continuation_messages(
                                base_messages, combined_text
                            )
                        attempt = 0
                        logger.info(
                            "Continuing truncated OpenRouter response part=%s",
                            continuation_count + 1,
                        )
                        continue
                    return GeminiResult(
                        text=_finish_text(combined_text, truncated),
                        interaction_id=None,
                        context_was_reset=bool(previous_interaction_id),
                        provider="openrouter",
                        truncated=truncated,
                    )
                except GeminiRequestError:
                    if combined_text:
                        return GeminiResult(
                            text=_finish_text(combined_text, True),
                            interaction_id=None,
                            context_was_reset=bool(previous_interaction_id),
                            provider="openrouter",
                            truncated=True,
                        )
                    raise
                except TimeoutError as exc:
                    if combined_text:
                        logger.warning(
                            "OpenRouter continuation timed out after partial response"
                        )
                        return GeminiResult(
                            text=_finish_text(combined_text, True),
                            interaction_id=None,
                            context_was_reset=bool(previous_interaction_id),
                            provider="openrouter",
                            truncated=True,
                        )
                    raise GeminiRequestError(
                        "OpenRouter не успел получить ответ модели за отведённое "
                        "время. Попробуйте ещё раз или отправьте файл меньшего размера.",
                        provider="openrouter",
                        status_code=408,
                        fallback_allowed=True,
                    ) from exc
                except Exception as exc:
                    code = _error_code(exc)
                    attempt += 1
                    if (
                        _is_retryable(exc, code)
                        and not _is_quota_error(exc, code)
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
                    if combined_text:
                        return GeminiResult(
                            text=_finish_text(combined_text, True),
                            interaction_id=None,
                            context_was_reset=bool(previous_interaction_id),
                            provider="openrouter",
                            truncated=True,
                        )
                    raise GeminiRequestError(
                        _friendly_openrouter_error(code),
                        delete_after_seconds=20.0 if code == 429 else None,
                        provider="openrouter",
                        status_code=code,
                        fallback_allowed=_allows_automatic_fallback(exc, code),
                        quota_exhausted=_is_quota_error(exc, code),
                        fallback_reason=(
                            "blocked" if _is_safety_block_error(exc) else None
                        ),
                    ) from exc

    async def _generate_groq(
        self,
        messages: list[dict[str, Any]],
        previous_interaction_id: str | None,
        *,
        model: str,
        is_fallback: bool,
    ) -> GeminiResult:
        client = self._groq_client
        if client is None:
            raise RuntimeError("Groq fallback client is not initialized")
        base_messages = messages
        request_messages = base_messages
        combined_text = ""
        continuation_count = 0
        attempt = 0
        while True:
            try:
                payload = {
                    "model": model,
                    "messages": request_messages,
                    "temperature": self._settings.gemini_temperature,
                    "max_completion_tokens": self._settings.groq_max_output_tokens,
                    "stream": False,
                }
                async with asyncio.timeout(self._settings.gemini_timeout_seconds):
                    response = await client.post("chat/completions", json=payload)
                    response.raise_for_status()
                completion = _extract_chat_completion(response.json())
                _raise_if_unusable_chat_completion(
                    completion,
                    provider="groq",
                    requested_model=model,
                )
                if not completion.text:
                    label = "Резервная модель Groq" if is_fallback else "Модель Groq"
                    raise GeminiRequestError(
                        f"{label} не вернула текстовый ответ. "
                        "Попробуйте переформулировать сообщение.",
                        provider="groq",
                        fallback_allowed=True,
                    )
                combined_text = _merge_continuation(
                    combined_text, completion.text
                )
                truncated = _needs_continuation(completion.finish_reason)
                logger.info(
                    "Groq completion model=%s finish_reason=%s continuation=%s",
                    completion.model or model,
                    completion.finish_reason,
                    continuation_count,
                )
                if truncated and continuation_count < _max_continuations(
                    self._settings
                ):
                    continuation_count += 1
                    request_messages = _continuation_messages(
                        base_messages, combined_text
                    )
                    attempt = 0
                    logger.info(
                        "Continuing truncated Groq response part=%s",
                        continuation_count + 1,
                    )
                    continue
                return GeminiResult(
                    text=_finish_text(combined_text, truncated),
                    interaction_id=None,
                    context_was_reset=bool(previous_interaction_id),
                    provider="groq",
                    truncated=truncated,
                )
            except GeminiRequestError:
                if combined_text:
                    return GeminiResult(
                        text=_finish_text(combined_text, True),
                        interaction_id=None,
                        context_was_reset=bool(previous_interaction_id),
                        provider="groq",
                        truncated=True,
                    )
                raise
            except TimeoutError as exc:
                if combined_text:
                    logger.warning(
                        "Groq continuation timed out after partial response"
                    )
                    return GeminiResult(
                        text=_finish_text(combined_text, True),
                        interaction_id=None,
                        context_was_reset=bool(previous_interaction_id),
                        provider="groq",
                        truncated=True,
                    )
                label = "Резервная модель Groq" if is_fallback else "Модель Groq"
                raise GeminiRequestError(
                    f"{label} не успела ответить. "
                    "Попробуйте немного позже.",
                    provider="groq",
                    status_code=408,
                    fallback_allowed=True,
                ) from exc
            except Exception as exc:
                code = _error_code(exc)
                attempt += 1
                if (
                    _is_retryable(exc, code)
                    and not _is_quota_error(exc, code)
                    and attempt < self._settings.gemini_retry_attempts
                ):
                    delay = min(4.0, (2 ** (attempt - 1)) + random.random())
                    logger.warning(
                        "Retryable Groq error code=%s attempt=%s type=%s",
                        code,
                        attempt,
                        type(exc).__name__,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.exception(
                    "Groq fallback failed code=%s type=%s",
                    code,
                    type(exc).__name__,
                )
                if combined_text:
                    return GeminiResult(
                        text=_finish_text(combined_text, True),
                        interaction_id=None,
                        context_was_reset=bool(previous_interaction_id),
                        provider="groq",
                        truncated=True,
                    )
                raise GeminiRequestError(
                    _friendly_groq_error(code, is_fallback=is_fallback),
                    delete_after_seconds=20.0 if code == 429 else None,
                    provider="groq",
                    status_code=code,
                    fallback_allowed=_allows_automatic_fallback(exc, code),
                    quota_exhausted=_is_quota_error(exc, code),
                    fallback_reason=(
                        "blocked" if _is_safety_block_error(exc) else None
                    ),
                ) from exc

    def _build_chat_messages(
        self,
        bundle: PromptBundle,
        history: tuple[ConversationMessage, ...],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._settings.gemini_system_prompt}
        ]
        messages.extend(
            {"role": item.role, "content": item.content} for item in history
        )
        messages.append(
            {"role": "user", "content": _build_openrouter_content(bundle)}
        )
        return messages


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


def _build_generate_content_continuation(
    bundle: PromptBundle, combined_text: str
) -> list[types.Content]:
    original = _build_generate_content(bundle)
    user_parts = (
        [types.Part.from_text(text=original)]
        if isinstance(original, str)
        else original
    )
    return [
        types.Content(role="user", parts=user_parts),
        types.Content(
            role="model",
            parts=[types.Part.from_text(text=combined_text)],
        ),
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=_CONTINUATION_PROMPT)],
        ),
    ]


def _standalone_continuation_input(combined_text: str) -> str:
    return (
        "Ниже приведена уже полученная часть ответа. Продолжи её строго с места "
        "остановки.\n\n"
        f"{combined_text}\n\n{_CONTINUATION_PROMPT}"
    )


def _extract_google_finish_reason(response: object) -> str | None:
    candidates = _field(response, "candidates", []) or []
    if not isinstance(candidates, (list, tuple)) or not candidates:
        return None
    return _optional_string(_field(candidates[0], "finish_reason"))


def _field(obj: object, name: str, default: object = None) -> object:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


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


def _build_openrouter_media_body(
    bundle: PromptBundle,
    history: tuple[ConversationMessage, ...],
    *,
    system_instruction: str,
    model: str,
    temperature: float,
    max_tokens: int,
    combined_text: str,
) -> _OpenRouterMediaBody:
    if not bundle.media:
        raise ValueError("Streaming OpenRouter body requires media")

    segments: list[bytes | _Base64Data] = [
        b'{"model":',
        _json_bytes(model),
        b',"messages":[',
    ]
    first_message = True

    def add_message(role: str, content: str) -> None:
        nonlocal first_message
        if not first_message:
            segments.append(b",")
        first_message = False
        segments.append(_json_bytes({"role": role, "content": content}))

    add_message("system", system_instruction)
    for item in history:
        add_message(item.role, item.content)

    if not first_message:
        segments.append(b",")
    first_message = False
    segments.append(b'{"role":"user","content":[')
    first_part = True

    def start_part() -> None:
        nonlocal first_part
        if not first_part:
            segments.append(b",")
        first_part = False

    for item in bundle.media:
        start_part()
        segments.append(_json_bytes({"type": "text", "text": item.label}))
        start_part()
        if item.kind == "image":
            segments.extend(
                (
                    b'{"type":"image_url","image_url":{"url":',
                    _json_open_string(f"data:{item.mime_type};base64,"),
                    _Base64Data(item.data),
                    b'"}}',
                )
            )
        elif item.kind == "document":
            segments.extend(
                (
                    b'{"type":"file","file":{"filename":"document.pdf",'
                    b'"file_data":',
                    _json_open_string(f"data:{item.mime_type};base64,"),
                    _Base64Data(item.data),
                    b'"}}',
                )
            )
        elif item.kind == "audio":
            segments.extend(
                (
                    b'{"type":"input_audio","input_audio":{"data":',
                    _json_open_string(""),
                    _Base64Data(item.data),
                    b'","format":',
                    _json_bytes(_audio_format(item.mime_type)),
                    b"}}",
                )
            )
        elif item.kind == "video":
            segments.extend(
                (
                    b'{"type":"video_url","video_url":{"url":',
                    _json_open_string(f"data:{item.mime_type};base64,"),
                    _Base64Data(item.data),
                    b'"}}',
                )
            )
        else:
            raise GeminiRequestError(
                f"OpenRouter не поддерживает тип вложения {item.kind!r}."
            )

    start_part()
    segments.append(_json_bytes({"type": "text", "text": bundle.prompt}))
    segments.append(b"]}")
    if combined_text:
        add_message("assistant", combined_text)
        add_message("user", _CONTINUATION_PROMPT)
    segments.extend(
        (
            b'],"temperature":',
            _json_bytes(temperature),
            b',"max_tokens":',
            _json_bytes(max_tokens),
            b',"stream":false}',
        )
    )
    return _OpenRouterMediaBody(segments)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_open_string(value: str) -> bytes:
    encoded = _json_bytes(value)
    return encoded[:-1]


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


def _extract_chat_completion(payload: dict[str, Any]) -> _ChatCompletion:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return _ChatCompletion("", None, _optional_string(payload.get("model")))
    first = choices[0]
    if not isinstance(first, dict):
        return _ChatCompletion("", None, _optional_string(payload.get("model")))
    finish_reason = _optional_string(
        first.get("finish_reason") or first.get("native_finish_reason")
    )
    message = first.get("message")
    if not isinstance(message, dict):
        return _ChatCompletion(
            "", finish_reason, _optional_string(payload.get("model"))
        )
    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts = [
            str(part.get("text", "")).strip()
            for part in content
            if isinstance(part, dict)
            and part.get("type") in {"text", "output_text"}
        ]
        text = "\n".join(part for part in parts if part).strip()
    else:
        text = ""
    return _ChatCompletion(
        text=text,
        finish_reason=finish_reason,
        model=_optional_string(payload.get("model")),
    )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", value)
    normalized = str(enum_value).strip()
    return normalized or None


def _needs_continuation(finish_reason: str | None) -> bool:
    if finish_reason is None:
        return False
    normalized = finish_reason.strip().lower().rsplit(".", 1)[-1]
    return normalized in _INCOMPLETE_FINISH_REASONS


def _is_blocked_finish_reason(finish_reason: str | None) -> bool:
    if finish_reason is None:
        return False
    normalized = finish_reason.strip().lower().rsplit(".", 1)[-1]
    return normalized in _BLOCKED_FINISH_REASONS


def _fallback_reason(error: GeminiRequestError) -> str:
    if error.fallback_reason:
        return error.fallback_reason
    if error.quota_exhausted:
        return "limit"
    return "unavailable"


def _bundle_media_bytes(bundle: PromptBundle) -> int:
    return sum(len(item.data) for item in bundle.media)


def _openrouter_model_supports_media(model: str) -> bool:
    return model.strip().lower() not in _OPENROUTER_MEDIA_UNSUPPORTED_MODELS


def _detached_request_error(error: GeminiRequestError) -> GeminiRequestError:
    """Copy user-facing metadata without retaining provider traceback buffers."""
    return GeminiRequestError(
        error.user_message,
        delete_after_seconds=error.delete_after_seconds,
        provider=error.provider,
        status_code=error.status_code,
        fallback_allowed=error.fallback_allowed,
        quota_exhausted=error.quota_exhausted,
        fallback_reason=error.fallback_reason,
    )


def _release_exception_chain(error: BaseException) -> None:
    error.__traceback__ = None
    error.__cause__ = None
    error.__context__ = None


def _raise_if_unusable_chat_completion(
    completion: _ChatCompletion,
    *,
    provider: str,
    requested_model: str,
) -> None:
    actual_model = completion.model or requested_model
    label = {"openrouter": "OpenRouter", "groq": "Groq"}.get(provider, provider)
    if provider == "openrouter" and _is_non_chat_model(actual_model):
        raise GeminiRequestError(
            f"{label} выбрал служебную safety-модель вместо чат-модели. "
            "Пробую другой маршрут.",
            provider=provider,
            fallback_allowed=True,
            fallback_reason="blocked",
        )
    if _is_blocked_finish_reason(completion.finish_reason):
        raise GeminiRequestError(
            f"{label} заблокировал этот запрос правилами безопасности. "
            "Пробую другую модель.",
            provider=provider,
            fallback_allowed=True,
            fallback_reason="blocked",
        )
    if _looks_like_blocked_response(completion.text):
        raise GeminiRequestError(
            f"{label} вернул служебное сообщение о блокировке запроса. "
            "Пробую другую модель.",
            provider=provider,
            fallback_allowed=True,
            fallback_reason="blocked",
        )


def _is_non_chat_model(model: str) -> bool:
    normalized = model.lower()
    return any(marker in normalized for marker in _NON_CHAT_MODEL_MARKERS)


def _looks_like_blocked_response(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    safety_lines = sum(
        1 for marker in _SAFETY_REPORT_MARKERS if marker in normalized
    )
    if safety_lines >= 2:
        return True
    compact = " ".join(normalized.split()).rstrip(" .!;:")
    return compact in _BLOCKED_RESPONSE_MESSAGES


def _max_continuations(settings: object) -> int:
    try:
        value = int(getattr(settings, "ai_max_continuations", 2))
    except (TypeError, ValueError):
        return 2
    return max(0, min(5, value))


def _continuation_messages(
    base_messages: list[dict[str, Any]], combined_text: str
) -> list[dict[str, Any]]:
    return [
        *base_messages,
        {"role": "assistant", "content": combined_text},
        {"role": "user", "content": _CONTINUATION_PROMPT},
    ]


def _merge_continuation(current: str, continuation: str) -> str:
    current = current.strip()
    continuation = continuation.strip()
    if not current:
        return continuation
    if not continuation:
        return current

    max_overlap = min(len(current), len(continuation), 2000)
    for overlap in range(max_overlap, 19, -1):
        if current[-overlap:] == continuation[:overlap]:
            continuation = continuation[overlap:].lstrip()
            break
    if not continuation:
        return current
    return f"{current}\n\n{continuation}"


def _finish_text(text: str, truncated: bool) -> str:
    normalized = text.strip()
    if not truncated:
        return normalized
    return f"{normalized}\n\n{_TRUNCATED_NOTICE}"


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


def _is_quota_error(exc: Exception, code: int | None) -> bool:
    if code in {402, 429}:
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "rate limit",
            "resource exhausted",
            "quota exceeded",
            "insufficient credits",
        )
    )


def _allows_automatic_fallback(exc: Exception, code: int | None) -> bool:
    return (
        code in {402, 408, 429, 500, 502, 503, 504}
        or _is_retryable(exc, code)
        or _is_safety_block_error(exc)
    )


def _is_safety_block_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    body = getattr(response, "text", "") if response is not None else ""
    text = f"{exc} {body}".lower()
    return any(marker in text for marker in _REQUEST_BLOCK_MARKERS)


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


def _friendly_groq_error(code: int | None, *, is_fallback: bool = True) -> str:
    if not is_fallback:
        if code in {401, 403}:
            return (
                "Groq отклонил ключ API или доступ к модели. Владельцу бота "
                "нужно проверить GROQ_API_KEY и ID модели."
            )
        if code == 429:
            return "Лимит запросов Groq временно исчерпан. Попробуйте немного позже."
        if code == 400:
            return "Groq не смог обработать запрос. Попробуйте выбрать другую модель."
        if code in {500, 502, 503, 504}:
            return "Groq временно недоступен. Попробуйте позже."
        return "Не удалось получить ответ от Groq. Попробуйте позже."
    if code in {401, 403}:
        return (
            "Резервная модель Groq отклонила ключ API. Владельцу бота нужно "
            "проверить GROQ_API_KEY."
        )
    if code == 429:
        return (
            "Лимиты основной и резервной моделей временно исчерпаны. "
            "Попробуйте немного позже."
        )
    if code == 400:
        return (
            "Резервная модель Groq не смогла обработать текст запроса. "
            "Попробуйте переформулировать сообщение."
        )
    if code in {500, 502, 503, 504}:
        return "Основная и резервная модели временно недоступны. Попробуйте позже."
    return "Не удалось получить ответ от резервной модели Groq. Попробуйте позже."
