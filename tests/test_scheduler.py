from datetime import datetime

import pytest

from neo.core.scheduler import (
    InvalidScheduleError,
    Scheduler,
    compute_next_run,
    next_run_after_firing,
)
from neo.memory.scheduled_job_store import ScheduledJob, ScheduledJobStore


class FakeSubtaskResult:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tool_calls: list = []


class FakeAgent:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.raise_on: str | None = None

    async def run_subtask(self, goal_context: str, instruction: str, **kwargs):
        self.calls.append(instruction)
        if self.raise_on is not None and instruction == self.raise_on:
            raise RuntimeError("boom")
        return FakeSubtaskResult(f"yanıt: {instruction}")


def _store(tmp_path) -> ScheduledJobStore:
    return ScheduledJobStore(tmp_path / "jobs.db")


# --- compute_next_run -------------------------------------------------

def test_daily_schedule_rolls_to_tomorrow_when_time_already_passed():
    now = datetime(2026, 9, 7, 9, 0, 0)
    result = compute_next_run("daily", "08:00", now)
    assert result == datetime(2026, 9, 8, 8, 0, 0)


def test_daily_schedule_stays_today_when_time_still_upcoming():
    now = datetime(2026, 9, 7, 7, 0, 0)
    result = compute_next_run("daily", "08:00", now)
    assert result == datetime(2026, 9, 7, 8, 0, 0)


def test_daily_schedule_rejects_bad_value():
    with pytest.raises(InvalidScheduleError):
        compute_next_run("daily", "sabah", datetime(2026, 9, 7, 9, 0, 0))


def test_once_schedule_parses_iso_datetime():
    result = compute_next_run("once", "2026-09-08T09:00:00", datetime(2026, 9, 7, 9, 0, 0))
    assert result == datetime(2026, 9, 8, 9, 0, 0)


def test_once_schedule_rejects_bad_value():
    with pytest.raises(InvalidScheduleError):
        compute_next_run("once", "yarın", datetime(2026, 9, 7, 9, 0, 0))


def test_interval_schedule_adds_seconds():
    now = datetime(2026, 9, 7, 9, 0, 0)
    result = compute_next_run("interval", "3600", now)
    assert result == datetime(2026, 9, 7, 10, 0, 0)


def test_interval_schedule_rejects_non_positive():
    with pytest.raises(InvalidScheduleError):
        compute_next_run("interval", "0", datetime(2026, 9, 7, 9, 0, 0))


def test_unknown_schedule_kind_rejected():
    with pytest.raises(InvalidScheduleError):
        compute_next_run("weekly", "08:00", datetime(2026, 9, 7, 9, 0, 0))


# --- next_run_after_firing ---------------------------------------------

def _job(schedule_kind, schedule_value) -> ScheduledJob:
    return ScheduledJob(
        id=1, name="x", instruction="x", schedule_kind=schedule_kind,
        schedule_value=schedule_value, next_run_at="2026-09-07T08:00:00",
        enabled=True, last_run_at=None, created_at="2026-09-01T00:00:00",
    )


def test_next_run_after_firing_once_job_is_none():
    fired_at = datetime(2026, 9, 8, 9, 0, 3)
    assert next_run_after_firing(_job("once", "2026-09-08T09:00:00"), fired_at) is None


def test_next_run_after_firing_daily_job_rolls_forward():
    fired_at = datetime(2026, 9, 7, 8, 0, 5)
    result = next_run_after_firing(_job("daily", "08:00"), fired_at)
    assert result == datetime(2026, 9, 8, 8, 0, 0)


# --- Scheduler.tick / _fire ---------------------------------------------

def test_tick_fires_due_job_and_reschedules(tmp_path):
    store = _store(tmp_path)
    past = datetime(2020, 1, 1, 8, 0, 0)
    job_id = store.create_job("iş", "günü özetle", "daily", "08:00", past.isoformat())
    agent = FakeAgent()
    fired = []
    scheduler = Scheduler(agent, store, on_fire=lambda job, text: fired.append((job.id, text)))

    import asyncio
    asyncio.run(scheduler.tick())

    assert agent.calls == ["günü özetle"]
    assert len(fired) == 1
    assert fired[0][0] == job_id
    updated = store.get_job(job_id)
    assert updated.last_run_at is not None
    assert datetime.fromisoformat(updated.next_run_at) > datetime.now()
    assert updated.enabled is True


def test_tick_disables_once_job_after_firing(tmp_path):
    store = _store(tmp_path)
    past = datetime(2020, 1, 1, 8, 0, 0)
    job_id = store.create_job("tek seferlik", "hatırlat", "once", past.isoformat(), past.isoformat())
    agent = FakeAgent()

    import asyncio
    asyncio.run(Scheduler(agent, store).tick())

    assert store.get_job(job_id).enabled is False


def test_tick_skips_jobs_not_yet_due(tmp_path):
    store = _store(tmp_path)
    future = datetime.now().replace(year=2099)
    store.create_job("gelecek", "x", "once", future.isoformat(), future.isoformat())
    agent = FakeAgent()

    import asyncio
    asyncio.run(Scheduler(agent, store).tick())

    assert agent.calls == []


def test_tick_skips_disabled_jobs(tmp_path):
    store = _store(tmp_path)
    past = datetime(2020, 1, 1, 8, 0, 0)
    job_id = store.create_job("iş", "x", "daily", "08:00", past.isoformat())
    store.set_enabled(job_id, False)
    agent = FakeAgent()

    import asyncio
    asyncio.run(Scheduler(agent, store).tick())

    assert agent.calls == []


def test_tick_only_fires_jobs_that_are_actually_due(tmp_path):
    store = _store(tmp_path)
    past = datetime(2020, 1, 1, 8, 0, 0)
    future = datetime.now().replace(year=2099)
    store.create_job("due", "yap", "once", past.isoformat(), past.isoformat())
    store.create_job("not due", "yapma", "once", future.isoformat(), future.isoformat())
    agent = FakeAgent()

    import asyncio
    asyncio.run(Scheduler(agent, store).tick())

    assert agent.calls == ["yap"]


def test_exception_while_firing_does_not_crash_tick_and_still_reports(tmp_path):
    store = _store(tmp_path)
    past = datetime(2020, 1, 1, 8, 0, 0)
    store.create_job("patlayan", "patla", "daily", "08:00", past.isoformat())
    agent = FakeAgent()
    agent.raise_on = "patla"
    fired = []
    scheduler = Scheduler(agent, store, on_fire=lambda job, text: fired.append(text))

    import asyncio
    asyncio.run(scheduler.tick())

    assert len(fired) == 1
    assert "hata" in fired[0].lower()


def test_reentrant_tick_does_not_fire_twice(tmp_path):
    """If tick() is somehow invoked again while a previous tick is still
    running (a slow LLM call under a live QTimer), the guard must prevent
    the same due job from being fired a second time concurrently."""
    store = _store(tmp_path)
    past = datetime(2020, 1, 1, 8, 0, 0)
    store.create_job("iş", "x", "daily", "08:00", past.isoformat())
    agent = FakeAgent()
    scheduler = Scheduler(agent, store)

    import asyncio

    async def run_concurrently():
        scheduler._running_tick = True
        await scheduler.tick()

    asyncio.run(run_concurrently())

    assert agent.calls == []
