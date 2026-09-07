from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScheduledJob:
    id: int
    name: str
    instruction: str
    schedule_kind: str  # "daily" | "once" | "interval"
    schedule_value: str  # "HH:MM" for daily, ISO datetime for once, seconds for interval
    next_run_at: str  # ISO datetime
    enabled: bool
    last_run_at: str | None
    created_at: str


class ScheduledJobStore:
    """SQLite-backed record of NEO's proactive jobs (core/scheduler.py runs
    against this) -- same one-file, no-server pattern as every other store
    in this codebase. A job survives NEO restarting; it does not survive
    the computer being off at the moment it was due (the next tick after
    startup picks up anything that's overdue, it does not try to "catch up"
    on every missed daily firing)."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scheduled_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    instruction TEXT NOT NULL,
                    schedule_kind TEXT NOT NULL,
                    schedule_value TEXT NOT NULL,
                    next_run_at TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    last_run_at TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> ScheduledJob:
        return ScheduledJob(
            row["id"], row["name"], row["instruction"], row["schedule_kind"],
            row["schedule_value"], row["next_run_at"], bool(row["enabled"]),
            row["last_run_at"], row["created_at"],
        )

    def create_job(
        self, name: str, instruction: str, schedule_kind: str, schedule_value: str, next_run_at: str
    ) -> int:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                """
                INSERT INTO scheduled_jobs
                    (name, instruction, schedule_kind, schedule_value, next_run_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, instruction, schedule_kind, schedule_value, next_run_at),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get_job(self, job_id: int) -> ScheduledJob | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM scheduled_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(self, enabled_only: bool = False) -> list[ScheduledJob]:
        query = "SELECT * FROM scheduled_jobs"
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY id"
        with closing(self._connect()) as conn:
            rows = conn.execute(query).fetchall()
        return [self._row_to_job(r) for r in rows]

    def find_by_name(self, name: str) -> ScheduledJob | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM scheduled_jobs WHERE name = ?", (name,)
            ).fetchone()
        return self._row_to_job(row) if row else None

    def record_run(self, job_id: int, ran_at: str, next_run_at: str | None) -> None:
        """Marks a job as having fired, and schedules (or disables) its
        next run. next_run_at=None means the job was one-off and is now
        disabled rather than deleted -- so it still shows up in history."""
        with closing(self._connect()) as conn:
            if next_run_at is None:
                conn.execute(
                    "UPDATE scheduled_jobs SET last_run_at = ?, enabled = 0 WHERE id = ?",
                    (ran_at, job_id),
                )
            else:
                conn.execute(
                    "UPDATE scheduled_jobs SET last_run_at = ?, next_run_at = ? WHERE id = ?",
                    (ran_at, next_run_at, job_id),
                )
            conn.commit()

    def delete_job(self, job_id: int) -> bool:
        with closing(self._connect()) as conn:
            cursor = conn.execute("DELETE FROM scheduled_jobs WHERE id = ?", (job_id,))
            conn.commit()
            return cursor.rowcount > 0

    def set_enabled(self, job_id: int, enabled: bool) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE scheduled_jobs SET enabled = ? WHERE id = ?", (int(enabled), job_id)
            )
            conn.commit()
