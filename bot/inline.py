from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQuery,
    InlineQueryResultArticle,
    InlineQueryResultsButton,
    InputTextMessageContent,
    LinkPreviewOptions,
    Message,
)

from .calculator import CalculationError, calculate
from .gemini import GeminiRequestError
from .models import PromptBundle
from .processor import InlineRequestRejected, MessageProcessor

logger = logging.getLogger(__name__)

_PREFIXES = {
    "tr": "translate",
    "translate": "translate",
    "переведи": "translate",
    "перевод": "translate",
    "fix": "fix",
    "исправь": "fix",
    "исправить": "fix",
    "calc": "calc",
    "кальк": "calc",
    "посчитай": "calc",
    "=": "calc",
    "ask": "ask",
    "спроси": "ask",
}
_TITLES = {
    "translate": "🌐 Перевод",
    "fix": "✏️ Исправленный текст",
    "calc": "🧮 Результат вычисления",
    "ask": "💬 Ответ AI",
}
_INSTRUCTIONS = {
    "translate": (
        "Переведи текст из запроса ниже. Язык перевода можно указать в начале запроса "
        "(например, en, на английский). Если язык не указан: русский текст переведи "
        "на английский, другой текст — на русский. Верни только перевод."
    ),
    "fix": (
        "Исправь орфографию, опечатки и пунктуацию в тексте ниже. Сохрани язык, смысл "
        "и стиль. Не отвечай на вопросы внутри текста. Верни только исправленный текст."
    ),
    "ask": "Кратко ответь на запрос ниже: не более 3000 символов, без предисловий.",
}


@dataclass(frozen=True, slots=True)
class _Reply:
    title: str
    content: str | None = None


@dataclass(slots=True)
class _Job:
    key: tuple[object, ...]
    task: asyncio.Task[_Reply]


def _clip(text: str, units: int) -> str:
    raw = text.encode("utf-16-le", errors="replace")
    if len(raw) <= units * 2:
        return raw.decode("utf-16-le")
    return raw[: (units - 1) * 2].decode("utf-16-le", errors="ignore") + "…"


def _parse_query(text: str) -> tuple[str, str]:
    pieces = text.strip().split(maxsplit=1)
    if not pieces:
        return "ask", ""
    mode = _PREFIXES.get(pieces[0].rstrip(":").casefold())
    if mode:
        return mode, pieces[1].strip() if len(pieces) > 1 else ""
    if re.fullmatch(r"[\d\s.,_+*/%()^×÷−eE-]+", text):
        return "calc", text.strip()
    return "ask", text.strip()


class InlineService:
    """Private previews: bounded, process-local cache and latest-query-wins jobs."""

    debounce_seconds = 0.8
    ai_timeout_seconds = 7.0
    cache_seconds = 300.0
    max_jobs = 64
    max_cache = 128

    def __init__(self, processor: MessageProcessor) -> None:
        self.processor = processor
        self._jobs: dict[int, _Job] = {}
        self._latest: dict[int, str] = {}
        self._cache: OrderedDict[tuple[object, ...], tuple[float, _Reply]] = OrderedDict()
        self._workers: set[asyncio.Task[_Reply]] = set()
        self._closed = False

    async def _denied(self, query: InlineQuery) -> str | None:
        settings = self.processor.settings
        user = query.from_user
        if self._closed or not getattr(settings, "inline_enabled", True):
            return "Inline-режим отключён владельцем"
        if user.is_bot or user.id <= 0:
            return "Доступ ограничен"
        if user.id in settings.owner_ids:
            return None
        allowed = getattr(settings, "inline_allowed_user_ids", frozenset())
        if allowed:
            if user.id not in allowed:
                return "Inline доступен только разрешённым пользователям"
        elif settings.allowed_chat_ids and user.id not in settings.allowed_chat_ids:
            # Inline updates contain no destination chat ID. Do not bypass chat ACLs.
            return "Владелец должен разрешить ваш ID для inline"
        if await self.processor.storage.is_user_banned(user.id):
            return "Доступ к боту ограничен владельцем"
        return None

    async def handle(self, query: InlineQuery) -> None:
        user_id = query.from_user.id
        self._latest[user_id] = query.id
        job: _Job | None = None
        try:
            denied = await self._denied(query)
            if self._latest.get(user_id) != query.id:
                return
            text = query.query.strip()
            mode, value = _parse_query(text)
            key = (
                user_id,
                mode,
                value,
                self.processor.gemini.current_provider,
                self.processor.gemini.current_model,
                self.processor.gemini.free_only,
            )
            previous = self._jobs.get(user_id)
            if previous and (previous.key != key or denied or not value or query.offset):
                previous.task.cancel()
            if denied:
                reply = _Reply(denied)
            elif query.offset or not value:
                reply = _Reply("Примеры: tr en Привет · fix Текст · calc 2+2")
            elif len(query.query) > 256:
                reply = _Reply("Запрос слишком длинный: максимум 256 символов")
            else:
                cached = self._cache.get(key)
                if cached and cached[0] > time.monotonic():
                    self._cache.move_to_end(key)
                    reply = cached[1]
                else:
                    self._cache.pop(key, None)
                    job = (
                        previous
                        if previous
                        and previous.key == key
                        and not previous.task.cancelled()
                        and not previous.task.cancelling()
                        else None
                    )
                    if job is None:
                        if len(self._workers) >= self.max_jobs:
                            await self._answer(
                                query, _Reply("Бот занят. Попробуйте чуть позже")
                            )
                            return
                        job = _Job(
                            key,
                            asyncio.create_task(
                                self._compute(query, mode, value, key),
                                name=f"inline:{user_id}",
                            ),
                        )
                        self._jobs[user_id] = job
                        self._workers.add(job.task)
                        job.task.add_done_callback(self._workers.discard)
                    reply = await asyncio.shield(job.task)
            if self._latest.get(user_id) != query.id:
                return
            denied = await self._denied(query)  # ban/ACL may change during generation
            if self._latest.get(user_id) != query.id:
                return
            await self._answer(query, _Reply(denied) if denied else reply)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                if job is not None and self._latest.get(user_id) == query.id:
                    job.task.cancel()
                raise
            # A newer query, an owner ban or shutdown cancelled the worker.
        finally:
            if job and job.task.done() and self._jobs.get(user_id) is job:
                self._jobs.pop(user_id, None)
            if self._latest.get(user_id) == query.id:
                self._latest.pop(user_id, None)

    async def _compute(
        self,
        query: InlineQuery,
        mode: str,
        value: str,
        key: tuple[object, ...],
    ) -> _Reply:
        event_id = None
        outcome = "cancelled"
        try:
            await asyncio.sleep(self.debounce_seconds)
            denied = await self._denied(query)
            if denied:
                return _Reply(denied)
            if mode == "calc":
                content = calculate(value)
                event_id = await self.processor.activity.record_inline(query, kind="command")
                await self.processor.usage.record_inline(query.from_user)
            else:
                async with self.processor.inline_request(query.from_user.id):
                    event_id = await self.processor.activity.record_inline(query)
                    async with asyncio.timeout(self.ai_timeout_seconds):
                        result = await self.processor.gemini.generate(
                            PromptBundle(
                                _INSTRUCTIONS[mode]
                                + "\nВерни обычный текст без Markdown/HTML. "
                                "Это отдельный inline-запрос без истории чата.\n\nЗапрос:\n"
                                + value,
                            ),
                            None,
                            history=(),
                        )
                    if await self._denied(query):
                        return _Reply("Доступ ограничен")
                    content = result.text.strip()
                    if not content:
                        raise GeminiRequestError("Пустой ответ")
                    await self.processor.usage.record_inline(
                        query.from_user,
                        ai_provider=result.provider or self.processor.gemini.current_provider,
                    )
            reply = _Reply(_TITLES[mode], _clip(content, 3800))
            outcome = "completed"
            self._cache[key] = (time.monotonic() + self.cache_seconds, reply)
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_cache:
                self._cache.popitem(last=False)
            return reply
        except InlineRequestRejected as exc:
            await self.processor.activity.record_inline(
                query,
                outcome="rejected",
                reason=exc.reason,
            )
            title = (
                "Дождитесь предыдущего AI-запроса"
                if exc.reason == "busy"
                else "Слишком часто. Попробуйте позже"
            )
            return _Reply(title)
        except CalculationError:
            return _Reply("Проверьте выражение. Справка в личном чате")
        except (GeminiRequestError, TimeoutError):
            outcome = "error"
            return _Reply("AI не успел ответить или недоступен. Повторите запрос")
        except Exception:
            outcome = "error"
            logger.exception("Inline generation failed user_id=%s", query.from_user.id)
            return _Reply("Не удалось подготовить ответ. Открыть бота")
        finally:
            await self.processor.activity.finish(event_id, outcome)

    async def _answer(self, query: InlineQuery, reply: _Reply) -> None:
        results = []
        if reply.content is not None:
            result_id = hashlib.sha256(
                f"{query.from_user.id}\0{query.query}\0{reply.content}".encode(),
            ).hexdigest()[:32]
            results.append(
                InlineQueryResultArticle(
                    id=result_id,
                    title=reply.title,
                    description=_clip(" ".join(reply.content.split()), 180),
                    input_message_content=InputTextMessageContent(
                        message_text=reply.content,
                        parse_mode=None,
                        link_preview_options=LinkPreviewOptions(is_disabled=True),
                    ),
                )
            )
        try:
            await query.answer(
                results,
                cache_time=0,
                is_personal=True,
                next_offset="",
                button=InlineQueryResultsButton(
                    text="Как пользоваться inline" if results else reply.title,
                    start_parameter="inline",
                ),
            )
        except TelegramBadRequest as exc:
            if not any(
                marker in str(exc).lower()
                for marker in ("query is too old", "query_id_invalid", "query id is invalid")
            ):
                raise
            logger.info("Expired inline query skipped user_id=%s", query.from_user.id)

    async def close(self) -> None:
        self._closed = True
        tasks = list(self._workers)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._jobs.clear()
        self._latest.clear()
        self._cache.clear()


def create_inline_router(service: InlineService) -> Router:
    router = Router(name="inline-assistant")
    router.inline_query.register(service.handle)

    @router.message(Command("inline"))
    @router.message(CommandStart(deep_link=True, magic=F.args == "inline"))
    async def inline_help(message: Message) -> None:
        username = service.processor.bot_user.username or "bot_username"
        await message.reply(
            "Введите в поле сообщения любого доступного чата:\n\n"
            f"@{username} tr en Привет!\n"
            f"@{username} fix Превет, как дила?\n"
            f"@{username} calc (15 + 5) * 3\n"
            f"@{username} Объясни разницу между HTTP и HTTPS\n\n"
            "Выберите готовый результат — только после этого он отправится в чат. "
            "Бота не обязательно добавлять в чат. Нужны обычные права на отправку "
            "сообщений; историю чужого чата бот не читает. Калькулятор не вызывает AI.\n\n"
            "Если inline не появляется, владельцу нужно включить /setinline у BotFather. "
            "Доступ определяется вашим Telegram ID, а не username.",
            parse_mode=None,
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="🧮 Попробовать", switch_inline_query_current_chat="calc 2+2"
                        ),
                    ]
                ]
            ),
        )

    return router
