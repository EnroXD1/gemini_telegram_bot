from __future__ import annotations

import asyncio
import hashlib
import math
from contextlib import suppress
from datetime import UTC

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    ReplyKeyboardRemove,
)

from . import __version__
from .gemini import GeminiService, ModelSelectionError, _is_free_openrouter_route
from .keyboards import (
    BACK_BUTTON,
    DIAGNOSTICS_BUTTON,
    HIDE_KEYBOARD_BUTTON,
    MORE_BUTTON,
    SETTINGS_BUTTON,
    main_keyboard,
)
from .processor import MessageProcessor
from .storage import Storage


async def restore_ai_settings(storage: Storage, service: GeminiService) -> None:
    saved_free = await storage.get_runtime_setting("ai_free_only")
    if saved_free in {"true", "false"}:
        # A removed key must never disable a previously saved spending guard.
        service.set_free_only(saved_free == "true", allow_unconfigured=True)
    saved_selection = await storage.get_ai_selection()
    if saved_selection is not None:
        with suppress(ModelSelectionError):
            service.select_model(*saved_selection)


def _button(text: str, action: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=f"cfg:{action}")


def _short(value: str, limit: int = 72) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _model_token(provider: str, model: str) -> str:
    return hashlib.sha256(f"{provider}\n{model}".encode()).hexdigest()[:16]


def _model_choices(service: GeminiService) -> list[tuple[str, str]]:
    routes = [*service.configured_routes(), *service.configured_providers().items()]
    if "openrouter" in service.configured_providers():
        routes.insert(0, ("openrouter", "openrouter/free"))
    return list(dict.fromkeys(route for route in routes if service.route_allowed(*route)))


async def settings_view(
    processor: MessageProcessor,
    owner_id: int,
) -> tuple[str, InlineKeyboardMarkup]:
    service = processor.gemini
    enabled = await processor.storage.get_business_auto_reply_enabled(
        owner_id, processor.settings.business_auto_reply_enabled
    )
    mode = "включён" if service.free_only else "выключен"
    effective = service.effective_routes()
    first = f"{effective[0][0]} / {_short(effective[0][1])}" if effective else "нет маршрута"
    last = service.health.last_response
    last_text = "ответов после запуска ещё нет"
    if last:
        model = last.actual_model or f"маршрут {last.route_model} (ID не сообщён API)"
        last_text = f"{last.provider} / {_short(model)}, {last.seconds:.1f} сек."
    text = (
        f"⚙️ Настройки бота v{__version__}\n"
        "AI-настройки общие для всех чатов.\n\n"
        f"Только бесплатно: {mode}\n"
        f"Выбрано: {service.current_provider} / {_short(service.current_model)}\n"
        f"Первый разрешённый маршрут: {first}\n"
        f"Последний AI-ответ: {last_text}\n"
        f"Ваши Business-автоответы: {'включены' if enabled else 'выключены'}\n\n"
        "В бесплатном режиме работают только OpenRouter Free и :free. "
        "Google/Groq исключены: их тариф нельзя определить по наличию ключа. "
        "У бесплатных моделей остаются лимиты запросов."
    )
    return text, InlineKeyboardMarkup(
        inline_keyboard=[
            [_button("🤖 Выбрать модель", "models")],
            [
                _button(
                    "🔒 Только бесплатно: ВКЛ"
                    if service.free_only
                    else "💳 Только бесплатно: ВЫКЛ",
                    "free:off" if service.free_only else "free:on",
                )
            ],
            [_button("📊 Диагностика моделей", "diagnostics")],
            [
                _button(
                    "🔕 Выключить Business-ответы"
                    if enabled
                    else "🔔 Включить Business-ответы",
                    "business:off" if enabled else "business:on",
                )
            ],
            [_button("🔄 Обновить панель", "home")],
        ]
    )


def diagnostics_view(service: GeminiService) -> tuple[str, InlineKeyboardMarkup]:
    health = service.health
    started = health.started_at.astimezone(UTC).strftime("%d.%m %H:%M UTC")
    lines = [
        "📊 Диагностика AI-маршрутов",
        f"Наблюдения с запуска: {started}.",
        "Наличие ключа или модели в каталоге не означает успешную проверку.",
        "",
    ]
    last = health.last_response
    if last:
        lines.extend(
            [
                f"Последний ответ: {last.provider} / "
                f"{_short(last.actual_model or last.route_model)}",
                f"Маршрут: {_short(last.route_model)}; время попытки: {last.seconds:.1f} сек.",
                "ID фактической модели не сообщён API." if last.actual_model is None else "",
                "Ответ достиг лимита длины." if last.truncated else "",
                "",
            ]
        )
    else:
        lines.extend(["Успешных AI-ответов после запуска ещё нет.", ""])
    effective = service.effective_routes()
    routes = list(dict.fromkeys([*effective, *service.configured_routes(), *health.routes]))
    media = service.effective_routes(media=True)
    for provider, model in routes[:10]:
        key = (provider, model)
        stats = health.routes.get(key)
        catalog_status = service.route_catalog_status(provider, model)
        cooldown = math.ceil(service.cooldown_remaining(provider, model))
        if not service.route_allowed(provider, model):
            state = "🔒 исключён бесплатным режимом"
        elif catalog_status:
            state = f"⚠️ {catalog_status}"
        elif key not in effective:
            state = "⏸ не участвует в текущей цепочке"
        elif cooldown:
            state = f"⏳ пауза ещё {cooldown} сек."
        elif service.route_is_probing(provider, model):
            state = "⏳ проверка восстановления после паузы"
        elif stats is None:
            state = "⚪ запросов ещё не было"
        else:
            state = (
                "✅ последний запрос успешен"
                if stats.last_error is None
                else "❌ последняя попытка неуспешна"
            )
        lines.append(f"{provider} / {_short(model, 60)}\n{state}")
        key_status = service.key_pool_status(provider, model)
        if key_status is not None:
            ready, total = key_status
            lines.append(f"Пул ключей: доступно {ready}/{total}; значения скрыты.")
        if stats:
            lines.append(
                f"Успехов {stats.successes}/{stats.attempts}; ошибок {stats.failures}; "
                f"среднее время {stats.total_seconds / stats.attempts:.1f} сек."
            )
            if stats.last_error:
                code = f"HTTP {stats.last_code}: " if stats.last_code else ""
                lines.append(f"Причина: {code}{stats.last_error}.")
        if key in effective and key not in media:
            lines.append("Только текст; вложения идут другим маршрутом.")
        lines.append("")
    if len(routes) > 10:
        lines.append(f"Ещё маршрутов: {len(routes) - 10}. Полная цепочка: /model.")
    if service.catalog_state == "error":
        lines.append("Каталог сейчас недоступен; используются настройки и последний снимок.")
    elif service.catalog_state == "loaded":
        lines.append("Каталог OpenRouter загружен; обновление не чаще раза в час.")
    else:
        lines.append("Каталог OpenRouter ещё не загружен.")
    lines.append("Показатели не содержат переписку и ключи; обнуляются при перезапуске.")
    return "\n".join(line for line in lines if line is not None)[:4000], InlineKeyboardMarkup(
        inline_keyboard=[
            [_button("🔄 Обновить показатели", "diagnostics")],
            [_button("📚 Обновить каталог моделей", "catalog")],
            [_button("⬅️ Настройки", "home")],
        ]
    )


def create_control_router(processor: MessageProcessor) -> Router:
    router = Router(name="control-panel")
    mutation_lock = asyncio.Lock()

    async def authorize(message: Message, user_id: int) -> bool:
        return (
            message.chat.type == "private"
            and not message.business_connection_id
            and message.chat.id == user_id
            and user_id in processor.settings.owner_ids
        )

    async def edit(callback: CallbackQuery, text: str, markup: InlineKeyboardMarkup) -> None:
        if not isinstance(callback.message, Message):
            return
        try:
            await callback.message.edit_text(text, reply_markup=markup, parse_mode=None)
        except TelegramBadRequest as exc:
            if "message is not modified" not in str(exc).lower():
                raise

    @router.message(Command("menu"))
    @router.message(
        F.text.in_({MORE_BUTTON, BACK_BUTTON}),
        F.chat.type == "private",
        ~F.business_connection_id,
    )
    async def menu_handler(message: Message) -> None:
        if message.chat.type != "private" or message.business_connection_id:
            await message.reply("Меню доступно в личном чате с ботом.")
            return
        owner = bool(message.from_user and message.from_user.id in processor.settings.owner_ids)
        page = "other" if message.text == MORE_BUTTON else "main"
        text = (
            "Дополнительное меню. «Назад» — вернуться к основным действиям."
            if page == "other"
            else "⌨️ Главное меню. Вопрос можно написать обычным сообщением."
        )
        await message.reply(
            text,
            reply_markup=main_keyboard(owner=owner, page=page),
        )

    @router.message(Command("hidekeyboard"))
    @router.message(
        F.text == HIDE_KEYBOARD_BUTTON, F.chat.type == "private", ~F.business_connection_id
    )
    async def hide_keyboard_handler(message: Message) -> None:
        if message.chat.type == "private" and not message.business_connection_id:
            await message.reply(
                "Клавиатура скрыта. Вернуть: /menu.", reply_markup=ReplyKeyboardRemove()
            )

    @router.callback_query(F.data.startswith("ui:"))
    async def public_callback(callback: CallbackQuery) -> None:
        message = callback.message
        if (
            not isinstance(message, Message)
            or message.chat.type != "private"
            or message.business_connection_id
            or message.chat.id != callback.from_user.id
        ):
            await callback.answer("Откройте личный чат с ботом.", show_alert=True)
            return
        action = callback.data
        if action not in {"ui:menu", "ui:reset"}:
            await callback.answer("Эта кнопка устарела.", show_alert=True)
            return
        await callback.answer()
        if action == "ui:reset":
            source = message.model_copy(update={"from_user": callback.from_user})
            processor.cancel(source)
            await processor.reset(source)
            text = "🆕 Новый диалог начат. Отправьте вопрос."
        else:
            text = "⌨️ Меню открыто. Вопрос можно написать обычным сообщением."
        await message.answer(
            text,
            reply_markup=main_keyboard(
                owner=callback.from_user.id in processor.settings.owner_ids
            ),
        )

    @router.message(Command("settings", "diagnostics"))
    @router.message(F.text.in_({SETTINGS_BUTTON, DIAGNOSTICS_BUTTON}))
    async def settings_handler(message: Message) -> None:
        user = message.from_user
        if user is None or not await authorize(message, user.id):
            await message.reply("Панель доступна только владельцу в личном чате с ботом.")
            return
        if (
            message.text == DIAGNOSTICS_BUTTON
            or (message.text or "").split(maxsplit=1)[0].split("@")[0] == "/diagnostics"
        ):
            text, markup = diagnostics_view(processor.gemini)
        else:
            text, markup = await settings_view(processor, user.id)
        await message.reply(text, reply_markup=markup, parse_mode=None)

    @router.callback_query(F.data.startswith("cfg:"))
    async def settings_callback(callback: CallbackQuery) -> None:
        message = callback.message
        if not isinstance(message, Message) or not await authorize(
            message, callback.from_user.id
        ):
            await callback.answer("Настройки доступны только владельцу.", show_alert=True)
            return
        action = (callback.data or "").removeprefix("cfg:")
        await callback.answer()
        service = processor.gemini
        if action == "free:off":
            await edit(
                callback,
                "Отключить режим «Только бесплатно»?\n"
                "Google, Groq и платные модели OpenRouter снова смогут участвовать "
                "в запросах. Расходы зависят от тарифа ваших аккаунтов.",
                InlineKeyboardMarkup(
                    inline_keyboard=[
                        [_button("Разрешить остальные маршруты", "free:confirm")],
                        [_button("Оставить только бесплатные", "home")],
                    ]
                ),
            )
            return
        if action == "models":
            choices = _model_choices(service)
            rows = [
                [
                    _button(
                        f"{provider}: {_short(model, 38)}",
                        f"model:{_model_token(provider, model)}",
                    )
                ]
                for provider, model in choices[:20]
            ]
            rows.append([_button("⬅️ Настройки", "home")])
            await edit(
                callback,
                "Выберите общую модель бота.\nТексты диалогов будут сброшены, "
                "как при /model. Для точного ID также доступна команда /model.",
                InlineKeyboardMarkup(inline_keyboard=rows),
            )
            return
        if action == "catalog":
            await service.refresh_model_catalog()
            text, markup = diagnostics_view(service)
        elif action == "diagnostics":
            text, markup = diagnostics_view(service)
        else:
            async with mutation_lock:
                try:
                    if action in {"free:on", "free:confirm"}:
                        enabled = action == "free:on"
                        if enabled and "openrouter" not in service.configured_providers():
                            raise ModelSelectionError(
                                "Сначала добавьте OPENROUTER_API_KEY на хостинге."
                            )
                        target = (service.current_provider, service.current_model)
                        if enabled and not _is_free_openrouter_route(*target):
                            target = ("openrouter", "openrouter/free")
                        old = (service.current_provider, service.current_model)
                        await processor.storage.set_ai_preferences(
                            free_only=enabled, provider=target[0], model=target[1]
                        )
                        service.set_free_only(enabled)
                        service.select_model(*target)
                        if old != target:
                            await processor.storage.clear_conversation_contexts()
                    elif action.startswith("model:"):
                        token = action.removeprefix("model:")
                        target = next(
                            (
                                route
                                for route in _model_choices(service)
                                if _model_token(*route) == token
                            ),
                            None,
                        )
                        if target is None:
                            raise ModelSelectionError(
                                "Список изменился. Откройте выбор модели заново."
                            )
                        await processor.storage.set_ai_selection(*target)
                        old = (service.current_provider, service.current_model)
                        service.select_model(*target)
                        if old != target:
                            await processor.storage.clear_conversation_contexts()
                    elif action in {"business:on", "business:off"}:
                        await processor.storage.set_business_auto_reply_enabled(
                            callback.from_user.id, action == "business:on"
                        )
                    elif action != "home":
                        raise ModelSelectionError("Эта кнопка устарела. Откройте /settings.")
                except ModelSelectionError as exc:
                    await edit(
                        callback,
                        str(exc),
                        InlineKeyboardMarkup(
                            inline_keyboard=[[_button("⬅️ Настройки", "home")]]
                        ),
                    )
                    return
            text, markup = await settings_view(processor, callback.from_user.id)
        await edit(callback, text, markup)

    return router
