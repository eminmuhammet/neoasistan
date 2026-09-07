from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TaskStep:
    id: int
    task_id: int
    position: int
    description: str
    status: str  # "pending" | "running" | "done" | "failed"
    result_summary: str | None
    attempts: int


@dataclass
class Task:
    id: int
    goal: str
    status: str  # "planning" | "running" | "done" | "failed"
    created_at: str
    steps: list[TaskStep] = field(default_factory=list)


class TaskStore:
    """SQLite-backed record of multi-step tasks the planner (core/planner.py)
    runs -- goal, steps, and each step's live status. Same reasoning as
    CalendarStore/PreferenceStore: one small local file, no server.

    This exists so a long-running task's progress survives NEO closing
    mid-task -- the record is not silently lost. Automatically resuming an
    interrupted task on the next startup is a separate, not-yet-built
    feature; today this store only guarantees the history is there to look
    at, not that it picks back up on its own.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'planning',
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL REFERENCES tasks(id),
                    position INTEGER NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    result_summary TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_task(self, goal: str) -> int:
        with closing(self._connect()) as conn:
            cursor = conn.execute("INSERT INTO tasks (goal) VALUES (?)", (goal,))
            conn.commit()
            return int(cursor.lastrowid)

    def add_step(self, task_id: int, position: int, description: str) -> int:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "INSERT INTO task_steps (task_id, position, description) VALUES (?, ?, ?)",
                (task_id, position, description),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def update_task_status(self, task_id: int, status: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
            conn.commit()

    def update_step(
        self,
        step_id: int,
        status: str | None = None,
        result_summary: str | None = None,
        attempts: int | None = None,
    ) -> None:
        """Updates only the fields actually passed -- None means "leave
        unchanged", not "clear this field"."""
        fields, values = [], []
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if result_summary is not None:
            fields.append("result_summary = ?")
            values.append(result_summary)
        if attempts is not None:
            fields.append("attempts = ?")
            values.append(attempts)
        if not fields:
            return
        values.append(step_id)
        with closing(self._connect()) as conn:
            conn.execute(f"UPDATE task_steps SET {', '.join(fields)} WHERE id = ?", values)
            conn.commit()

    def get_task(self, task_id: int) -> Task | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT id, goal, status, created_at FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return None
            step_rows = conn.execute(
                "SELECT id, task_id, position, description, status, result_summary, attempts "
                "FROM task_steps WHERE task_id = ? ORDER BY position",
                (task_id,),
            ).fetchall()
        steps = [
            TaskStep(
                s["id"], s["task_id"], s["position"], s["description"],
                s["status"], s["result_summary"], s["attempts"],
            )
            for s in step_rows
        ]
        return Task(row["id"], row["goal"], row["status"], row["created_at"], steps)

    def recent_tasks(self, limit: int = 10) -> list[Task]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id FROM tasks ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        tasks = [self.get_task(row["id"]) for row in rows]
        return [t for t in tasks if t is not None]
