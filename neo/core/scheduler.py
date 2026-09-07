from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable

from ..memory.scheduled_job_store import ScheduledJob, ScheduledJobStore

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)

# How often the scheduler checks whether anything is due. Coarse on purpose:
# nothing this app schedules needs second-level precision (a morning
# briefing arriving up to a minute late is unnoticeable), and checking more
# often would just be wasted wakeups on an app meant to idle cheaply.
DEFAULT_TICK_SECONDS = 60.0

OnFire = Callable[[ScheduledJob, str], None]

VALID_SCHEDULE_KINDS = ("daily", "once", "interval")


class InvalidScheduleError(ValueError):
    """Raised when a schedule_kind/schedule_value combination doesn't parse."""


def compute_next_run(schedule_kind: str, schedule_value: str, now: datetime) -> datetime | None:
    """The next time a job should fire, or None for a "once" job that has
    already run (nothing left to schedule).

    Shared between job creation (computing the first run) and the
    scheduler's own tick (computing the run after this one), so the two
    can never disagree about what a given schedule means.
    """
    if schedule_kind == "daily":
        hour, _, minute = schedule_value.partition(":")
        try:
            hour, minute = int(hour), int(minute)
        except ValueError as exc:
            raise InvalidScheduleError(
                f"daily için schedule_value 'HH:MM' olmalı, gelen: {schedule_value!r}"
            ) from exc
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    if schedule_kind == "once":
        try:
            candidate = datetime.fromisoformat(schedule_value)
        except ValueError as exc:
            raise InvalidScheduleError(
                f"once için schedule_value ISO tarih-saat olmalı, gelen: {schedule_value!r}"
            ) from exc
        return candidate

    if schedule_kind == "interval":
        try:
            seconds = float(schedule_value)
        except ValueError as exc:
            raise InvalidScheduleError(
                f"interval için schedule_value saniye cinsinden sayı olmalı, gelen: {schedule_value!r}"
            ) from exc
        if seconds <= 0:
            raise InvalidScheduleError("interval pozitif olmalı")
        return now + timedelta(seconds=seconds)

    raise InvalidScheduleError(f"bilinmeyen schedule_kind: {schedule_kind!r}")


def next_run_after_firing(job: ScheduledJob, fired_at: datetime) -> datetime | None:
    """What a job's next_run_at should become right after it fires --
    "once" jobs return None (nothing left to schedule, so record_run
    disables them); "daily"/"interval" jobs get pushed forward by exactly
    one period from the moment they actually fired.
    """
    if job.schedule_kind == "once":
        return None
    return compute_next_run(job.schedule_kind, job.schedule_value, fired_at)


class Scheduler:
    """Runs NEO's proactive jobs: a QTimer tick checks the store for
    anything due and, for each one, runs its instruction through
    Agent.run_subtask -- the exact same tool-use loop and permission
    checks an ordinary message gets, just triggered by a clock instead of
    the user typing something.

    A fired job's own text ("Bugün hava güneşli, takviminde ..." ) is
    handed to `on_fire`; this class does not decide how -- or whether -- to
    surface it on screen or speak it out loud. Wiring that to the actual
    chat window is main.py's job, kept deliberately separate here so the
    scheduling engine doesn't need to know anything about the GUI.
    """

    def __init__(
        self,
        agent: "Agent",
        store: ScheduledJobStore,
        on_fire: OnFire | None = None,
        tick_seconds: float = DEFAULT_TICK_SECONDS,
    ) -> None:
        self._agent = agent
        self._store = store
        self._on_fire = on_fire
        self._tick_seconds = tick_seconds
        self._qtimer = None
        self._running_tick = False

    def start(self) -> None:
        """Begins ticking. Deferred import: this module has no other Qt
        dependency, and every other core/ module in this codebase is
        Qt-free -- only actually starting a live timer needs it."""
        from PySide6.QtCore import QTimer

        if self._qtimer is not None:
            return
        self._qtimer = QTimer()
        self._qtimer.setInterval(int(self._tick_seconds * 1000))
        self._qtimer.timeout.connect(lambda: asyncio.ensure_future(self.tick()))
        self._qtimer.start()

    def stop(self) -> None:
        if self._qtimer is not None:
            self._qtimer.stop()
            self._qtimer = None

    async def tick(self) -> None:
        """Checks for due jobs and fires them. Public (not just the QTimer
        callback) so a test -- or a manual "check now" -- can call it
        directly without waiting on a real timer.

        Re-entrancy guard: if firing a job takes longer than one tick
        interval (an LLM call plus tool use easily can), a second tick
        must not start firing the same jobs again on top of the first.
        """
        if self._running_tick:
            return
        self._running_tick = True
        try:
            now = datetime.now()
            due = [
                job
                for job in self._store.list_jobs(enabled_only=True)
                if datetime.fromisoformat(job.next_run_at) <= now
            ]
            for job in due:
                await self._fire(job, now)
        finally:
            self._running_tick = False

    async def _fire(self, job: ScheduledJob, now: datetime) -> None:
        try:
            result = await self._agent.run_subtask("", job.instruction)
            text = result.text
        except Exception:
            logger.exception("Zamanlanmış görev çalıştırılamadı: %s", job.name)
            text = "Bu zamanlanmış görevi çalıştırırken bir hata oluştu."

        next_run = next_run_after_firing(job, now)
        self._store.record_run(
            job.id, now.isoformat(), next_run.isoformat() if next_run else None
        )
        if self._on_fire is not None:
            self._on_fire(job, text)
