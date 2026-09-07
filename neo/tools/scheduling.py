from __future__ import annotations

import asyncio
from datetime import datetime

from ..core.scheduler import InvalidScheduleError, compute_next_run
from ..memory.scheduled_job_store import ScheduledJobStore
from .base import RiskLevel, Tool, ToolResult


class ScheduleTaskTool(Tool):
    name = "schedule_task"
    description = (
        "İleride belirli bir zamanda ya da düzenli olarak yapılacak bir şeyi "
        "zamanlar (ör. 'her sabah 9'da günün özetini oku', 'yarın öğlen "
        "bana X'i hatırlat'). 'instruction', zamanı geldiğinde NEO'nun kendi "
        "kendine yapacağı bir istekmiş gibi yazılır (ör. 'Bugünkü takvim "
        "notlarını ve hava durumunu özetleyip kullanıcıya söyle'). "
        "'schedule_kind': 'daily' (her gün aynı saatte, schedule_value "
        "'HH:MM'), 'once' (tek seferlik, schedule_value ISO tarih-saat, ör. "
        "'2026-09-08T09:00:00'), 'interval' (belirli aralıklarla, "
        "schedule_value saniye cinsinden sayı)."
    )
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Kısa bir isim (ör. 'sabah brifingi')."},
            "instruction": {
                "type": "string",
                "description": "Zamanı geldiğinde NEO'nun yerine getireceği talimat.",
            },
            "schedule_kind": {"type": "string", "enum": ["daily", "once", "interval"]},
            "schedule_value": {
                "type": "string",
                "description": "daily: 'HH:MM' | once: ISO tarih-saat | interval: saniye.",
            },
        },
        "required": ["name", "instruction", "schedule_kind", "schedule_value"],
    }

    def __init__(self, store: ScheduledJobStore) -> None:
        self._store = store

    async def run(
        self, name: str, instruction: str, schedule_kind: str, schedule_value: str, **kwargs: object
    ) -> ToolResult:
        try:
            next_run = compute_next_run(schedule_kind, schedule_value, datetime.now())
        except InvalidScheduleError as exc:
            return ToolResult(success=False, error=str(exc))

        job_id = await asyncio.to_thread(
            self._store.create_job, name, instruction, schedule_kind, schedule_value,
            next_run.isoformat(),
        )
        return ToolResult(
            success=True,
            data={"id": job_id, "name": name, "next_run_at": next_run.isoformat()},
        )


class ListScheduledTasksTool(Tool):
    name = "list_scheduled_tasks"
    description = "Zamanlanmış tüm görevleri (aktif ve devre dışı) listeler."
    risk = RiskLevel.LOW
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, store: ScheduledJobStore) -> None:
        self._store = store

    async def run(self, **kwargs: object) -> ToolResult:
        jobs = await asyncio.to_thread(self._store.list_jobs)
        return ToolResult(
            success=True,
            data={
                "jobs": [
                    {
                        "id": j.id,
                        "name": j.name,
                        "schedule_kind": j.schedule_kind,
                        "schedule_value": j.schedule_value,
                        "next_run_at": j.next_run_at,
                        "enabled": j.enabled,
                    }
                    for j in jobs
                ]
            },
        )


class CancelScheduledTaskTool(Tool):
    name = "cancel_scheduled_task"
    description = (
        "Zamanlanmış bir görevi kalıcı olarak siler. Hangi görev olduğunu "
        "önce list_scheduled_tasks ile öğrenip 'id' değerini kullan."
    )
    # Unlike delete_calendar_note/forget_preference (MEDIUM -- those remove
    # content the user actually wrote, which can't be perfectly
    # reconstructed), a scheduled job is a re-creatable automation setting:
    # if this is undone by mistake, re-asking for it is one sentence.
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "job_id": {"type": "integer", "description": "Silinecek zamanlanmış görevin id değeri."},
        },
        "required": ["job_id"],
    }

    def __init__(self, store: ScheduledJobStore) -> None:
        self._store = store

    async def run(self, job_id: int, **kwargs: object) -> ToolResult:
        removed = await asyncio.to_thread(self._store.delete_job, int(job_id))
        if not removed:
            return ToolResult(success=False, error=f"{job_id} numaralı zamanlanmış görev bulunamadı.")
        return ToolResult(success=True, data={"id": job_id})
