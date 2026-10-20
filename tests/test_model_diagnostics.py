import asyncio
import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.model_health import DiagnosticObservation, ModelHealth
from bot.storage import Storage


@pytest.fixture
async def diagnostics_db(tmp_path, monkeypatch):
    # Freeze only the storage's clock; never patch asyncio's shared time module.
    clock = SimpleNamespace(now=int(time.time()) // 3600 * 3600 + 1800)
    monkeypatch.setattr("bot.storage.time", SimpleNamespace(time=lambda: clock.now))
    storage = Storage(tmp_path / "bot.sqlite3")
    await storage.open()
    yield storage, clock
    await storage.close()


def observation(clock, *, model="top-model", hours=0, failed=False, seconds=2, truncated=False):
    return DiagnosticObservation(
        "google", model, seconds, clock.now - hours * 3600, failed, truncated
    )


async def test_aggregates_survive_reopen_without_duplicating_counts(diagnostics_db):
    storage, clock = diagnostics_db
    await storage.set_runtime_setting("unrelated", "keep")
    await storage.record_model_diagnostic(observation(clock, seconds=2))
    await storage.record_model_diagnostic(observation(clock, seconds=10, failed=True))
    await storage.record_model_diagnostic(observation(clock, seconds=3, truncated=True))
    for _ in range(2):
        await storage.close()
        await storage.open()
        snapshot = await storage.get_model_diagnostics()
        assert snapshot.period == "24h" and len(snapshot.routes) == 1
        route = snapshot.routes[0]
        assert (route.attempts, route.successes, route.failures, route.truncated) == (
            3,
            2,
            1,
            1,
        )
        assert route.total_seconds == 15 and route.total_seconds / route.attempts == 5
        assert await storage.get_runtime_setting("unrelated") == "keep"


async def test_weighted_average_and_provider_identity_are_not_lost(diagnostics_db):
    storage, clock = diagnostics_db
    for _ in range(9):
        await storage.record_model_diagnostic(observation(clock, hours=1, seconds=1))
    await storage.record_model_diagnostic(observation(clock, seconds=21))
    await storage.record_model_diagnostic(
        DiagnosticObservation("groq", "top-model", 5, clock.now, False)
    )
    routes = (await storage.get_model_diagnostics()).routes
    assert len(routes) == 2
    assert routes[0].provider == "google" and routes[0].attempts == 10
    assert routes[0].total_seconds / routes[0].attempts == 3  # not average(hourly averages)
    assert routes[1].provider == "groq" and routes[1].attempts == 1


@pytest.mark.parametrize(
    "period,expected",
    [
        ("24h", {"age-0", "age-23"}),
        ("7d", {"age-0", "age-23", "age-24", "age-167"}),
        ("30d", {"age-0", "age-23", "age-24", "age-167", "age-168", "age-719"}),
    ],
)
async def test_periods_use_explicit_utc_hour_boundaries(diagnostics_db, period, expected):
    storage, clock = diagnostics_db
    for age in (-1, 0, 23, 24, 167, 168, 719, 720):
        await storage.record_model_diagnostic(observation(clock, hours=age, model=f"age-{age}"))
    snapshot = await storage.get_model_diagnostics(period)
    assert {route.model for route in snapshot.routes} == expected
    assert snapshot.since % 3600 == 0 and snapshot.until == clock.now


async def test_retention_prunes_only_old_aggregates_at_startup_and_on_hour_change(
    diagnostics_db,
):
    storage, clock = diagnostics_db
    await storage.record_model_diagnostic(observation(clock, hours=720, model="expired"))
    await storage.record_model_diagnostic(observation(clock, hours=719, model="oldest"))
    await storage.set_runtime_setting("unrelated", "keep")
    await storage.close()
    await storage.open()
    cursor = await storage._connection().execute("SELECT model FROM model_diagnostics")
    assert {row[0] for row in await cursor.fetchall()} == {"oldest"}
    await cursor.close()
    clock.now += 3600
    await storage.record_model_diagnostic(observation(clock, model="new"))
    snapshot = await storage.get_model_diagnostics("30d")
    assert {item.model for item in snapshot.routes} == {"new"}
    assert await storage.get_runtime_setting("unrelated") == "keep"


async def test_concurrent_observations_do_not_overwrite_each_other(diagnostics_db):
    storage, clock = diagnostics_db
    await asyncio.gather(
        *(
            storage.record_model_diagnostic(observation(clock, failed=bool(i % 2)))
            for i in range(100)
        )
    )
    (route,) = (await storage.get_model_diagnostics()).routes
    assert (route.attempts, route.successes, route.failures, route.total_seconds) == (
        100,
        50,
        50,
        200,
    )


async def test_new_table_can_be_added_to_existing_database_without_losing_context(tmp_path):
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE conversations "
            "(scope_key TEXT PRIMARY KEY, previous_interaction_id TEXT, updated_at INTEGER)"
        )
        db.execute("INSERT INTO conversations VALUES ('private:42', 'existing-context', 123)")
    storage = Storage(path)
    await storage.open()
    try:
        assert await storage.get_interaction_id("private:42") == "existing-context"
        assert (await storage.get_model_diagnostics()).routes == ()
    finally:
        await storage.close()


async def test_insert_is_rolled_back_after_write_failure(diagnostics_db, monkeypatch):
    storage, clock = diagnostics_db
    await storage.record_model_diagnostic(observation(clock))
    db = storage._connection()
    original = db.execute

    async def broken(sql, *args):
        result = await original(sql, *args)
        if "INSERT INTO model_diagnostics" in sql:
            raise sqlite3.OperationalError("failure after insert")
        return result

    monkeypatch.setattr(db, "execute", broken)
    with pytest.raises(sqlite3.OperationalError):
        await storage.record_model_diagnostic(observation(clock))
    monkeypatch.setattr(db, "execute", original)
    (route,) = (await storage.get_model_diagnostics()).routes
    assert route.attempts == 1


async def test_failed_commit_does_not_fail_ai_or_log_raw_error(
    diagnostics_db, monkeypatch, caplog
):
    storage, _ = diagnostics_db
    health = ModelHealth()
    health.attach_storage(storage)
    db = storage._connection()
    original = db.commit
    monkeypatch.setattr(
        db, "commit", AsyncMock(side_effect=sqlite3.OperationalError("FAKE-secret-prompt"))
    )
    await health.record_attempt("google", "model", time.monotonic())
    assert health.routes[("google", "model")].successes == 1
    assert health.persistence_failures == 1 and "FAKE-secret-prompt" not in caplog.text
    monkeypatch.setattr(db, "commit", original)
    assert (await storage.get_model_diagnostics()).routes == ()


async def test_cancelled_write_rolls_back_and_does_not_poison_next_attempt(
    diagnostics_db, monkeypatch
):
    storage, clock = diagnostics_db
    db = storage._connection()
    original = db.commit
    entered = asyncio.Event()

    async def blocked_commit():
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(db, "commit", blocked_commit)
    task = asyncio.create_task(storage.record_model_diagnostic(observation(clock)))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    monkeypatch.setattr(db, "commit", original)
    await storage.record_model_diagnostic(observation(clock))
    (route,) = (await storage.get_model_diagnostics()).routes
    assert route.attempts == 1


async def test_health_counters_and_current_state_are_separate_from_history(diagnostics_db):
    storage, clock = diagnostics_db
    await storage.record_model_diagnostic(observation(clock, failed=True))
    health = ModelHealth()
    health.attach_storage(storage)
    assert not health.routes and health.last_response is None
    await health.record_attempt("google", "top-model", time.monotonic(), truncated=True)
    assert health.routes[("google", "top-model")].attempts == 1
    (route,) = (await storage.get_model_diagnostics()).routes
    assert (route.attempts, route.successes, route.failures, route.truncated) == (2, 1, 1, 1)


async def test_schema_contains_no_message_identity_content_credentials_or_cooldowns(
    diagnostics_db,
):
    storage, clock = diagnostics_db
    await storage.record_model_diagnostic(observation(clock))
    cursor = await storage._connection().execute("PRAGMA table_info(model_diagnostics)")
    columns = {row[1] for row in await cursor.fetchall()}
    await cursor.close()
    assert columns == {
        "bucket_start",
        "provider",
        "model",
        "attempts",
        "successes",
        "failures",
        "total_seconds",
        "truncated",
        "last_at",
    }


@pytest.mark.parametrize("period", ["", "1d", "all", "24h OR 1=1"])
async def test_invalid_periods_are_rejected(diagnostics_db, period):
    storage, _ = diagnostics_db
    with pytest.raises(ValueError):
        await storage.get_model_diagnostics(period)
