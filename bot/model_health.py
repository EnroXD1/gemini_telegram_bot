from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .storage import Storage

logger = logging.getLogger(__name__)

DIAGNOSTIC_PERIODS = {"24h": 24, "7d": 7 * 24, "30d": 30 * 24}
DIAGNOSTIC_RETENTION_HOURS = 30 * 24


@dataclass(frozen=True, slots=True)
class DiagnosticObservation:
    provider: str
    model: str
    seconds: float
    at: int
    failed: bool
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class RouteAggregate:
    provider: str
    model: str
    attempts: int
    successes: int
    failures: int
    total_seconds: float
    truncated: int
    last_at: int


@dataclass(frozen=True, slots=True)
class DiagnosticsSnapshot:
    period: str
    since: int
    until: int
    routes: tuple[RouteAggregate, ...]


@dataclass(slots=True)
class RouteHealth:
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    total_seconds: float = 0.0
    last_seconds: float = 0.0
    last_code: int | None = None
    last_error: str | None = None
    last_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class LastResponse:
    provider: str
    route_model: str
    actual_model: str | None
    seconds: float
    at: datetime
    truncated: bool


class ModelHealth:
    """Live state in RAM, optional durable aggregates without user content or keys."""

    def __init__(self) -> None:
        self.started_at = datetime.now(UTC)
        self.routes: OrderedDict[tuple[str, str], RouteHealth] = OrderedDict()
        self.last_response: LastResponse | None = None
        self._storage: Storage | None = None
        self.persistence_failures = 0

    def attach_storage(self, storage: Storage) -> None:
        self._storage = storage

    async def record_attempt(
        self,
        provider: str,
        model: str,
        started: float,
        *,
        error: str | None = None,
        code: int | None = None,
        actual_model: str | None = None,
        truncated: bool = False,
    ) -> None:
        observation = self.record(
            provider,
            model,
            started,
            error=error,
            code=code,
            actual_model=actual_model,
            truncated=truncated,
        )
        if self._storage is not None:
            try:
                # Await one committed UPSERT, not a task/queue that can lose data at shutdown.
                await self._storage.record_model_diagnostic(observation)
            except Exception as exc:
                self.persistence_failures += 1
                logger.warning("Could not persist AI diagnostics type=%s", type(exc).__name__)

    def record(
        self,
        provider: str,
        model: str,
        started: float,
        *,
        error: str | None = None,
        code: int | None = None,
        actual_model: str | None = None,
        truncated: bool = False,
    ) -> DiagnosticObservation:
        key = (provider, model)
        health = self.routes.setdefault(key, RouteHealth())
        self.routes.move_to_end(key)
        if len(self.routes) > 64:
            self.routes.popitem(last=False)
        seconds = max(0.0, time.monotonic() - started)
        now = datetime.now(UTC)
        health.attempts += 1
        health.successes += error is None
        health.failures += error is not None
        health.total_seconds += seconds
        health.last_seconds = seconds
        health.last_error = error
        health.last_code = code
        health.last_at = now
        if error is None:
            self.last_response = LastResponse(
                provider, model, actual_model, seconds, now, truncated
            )
        return DiagnosticObservation(
            provider,
            model,
            seconds,
            int(now.timestamp()),
            error is not None,
            truncated and error is None,
        )


def error_label(code: int | None, *, blocked: bool = False) -> str:
    if blocked:
        return "запрос отклонён фильтром модели"
    return {
        400: "модель не приняла формат запроса",
        401: "ключ API отклонён",
        402: "недостаточно средств или квоты",
        403: "нет доступа к модели",
        404: "модель или endpoint не найдены",
        408: "истекло время ожидания",
        429: "исчерпан лимит запросов",
        500: "ошибка сервиса",
        502: "ошибка шлюза провайдера",
        503: "сервис временно недоступен",
        504: "шлюз не дождался ответа",
    }.get(code, "запрос не выполнен")
