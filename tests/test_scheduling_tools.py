import asyncio
from datetime import datetime, timedelta

from neo.memory.scheduled_job_store import ScheduledJobStore
from neo.tools.base import RiskLevel
from neo.tools.scheduling import (
    CancelScheduledTaskTool,
    ListScheduledTasksTool,
    ScheduleTaskTool,
)


def _store(tmp_path) -> ScheduledJobStore:
    return ScheduledJobStore(tmp_path / "jobs.db")


def test_schedule_daily_task_creates_a_job(tmp_path):
    store = _store(tmp_path)
    tool = ScheduleTaskTool(store)

    result = asyncio.run(tool.run(
        name="sabah brifingi", instruction="günü özetle",
        schedule_kind="daily", schedule_value="08:00",
    ))

    assert result.success is True
    job = store.get_job(result.data["id"])
    assert job.name == "sabah brifingi"
    assert job.schedule_kind == "daily"


def test_schedule_once_task_uses_given_time(tmp_path):
    store = _store(tmp_path)
    tool = ScheduleTaskTool(store)
    when = (datetime.now() + timedelta(days=1)).replace(microsecond=0)

    result = asyncio.run(tool.run(
        name="hatırlatma", instruction="ilaç al",
        schedule_kind="once", schedule_value=when.isoformat(),
    ))

    assert result.success is True
    assert result.data["next_run_at"] == when.isoformat()


def test_schedule_task_rejects_invalid_schedule_value(tmp_path):
    store = _store(tmp_path)
    tool = ScheduleTaskTool(store)

    result = asyncio.run(tool.run(
        name="bozuk", instruction="x", schedule_kind="daily", schedule_value="öğlen",
    ))

    assert result.success is False
    assert store.list_jobs() == []


def test_list_scheduled_tasks_returns_created_jobs(tmp_path):
    store = _store(tmp_path)
    store.create_job("iş", "x", "daily", "08:00", datetime.now().isoformat())

    result = asyncio.run(ListScheduledTasksTool(store).run())

    assert result.success is True
    assert len(result.data["jobs"]) == 1
    assert result.data["jobs"][0]["name"] == "iş"


def test_cancel_scheduled_task_removes_it(tmp_path):
    store = _store(tmp_path)
    job_id = store.create_job("iş", "x", "daily", "08:00", datetime.now().isoformat())

    result = asyncio.run(CancelScheduledTaskTool(store).run(job_id=job_id))

    assert result.success is True
    assert store.get_job(job_id) is None


def test_cancel_missing_scheduled_task_fails_cleanly(tmp_path):
    result = asyncio.run(CancelScheduledTaskTool(_store(tmp_path)).run(job_id=999))

    assert result.success is False


def test_scheduling_tools_are_low_risk():
    assert ScheduleTaskTool.risk == RiskLevel.LOW
    assert ListScheduledTasksTool.risk == RiskLevel.LOW
    assert CancelScheduledTaskTool.risk == RiskLevel.LOW
