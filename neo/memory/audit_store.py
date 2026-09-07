from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Tool input can carry things a log shouldn't keep verbatim -- most
# obviously typed text, which could be anything the user dictated into a
# form. Logging the tool name and success/failure already answers "what
# did you do", so the full input is capped rather than stored whole.
MAX_INPUT_CHARS = 200


@dataclass
class AuditEntry:
    id: int
    tool_name: str
    risk: str
    input_summary: str
    success: bool
    error: str | None
    created_at: str


class AuditStore:
    """SQLite-backed log of every tool NEO actually ran (or was refused
    running), the backend half of the planned "security merkezi" -- answers
    "son 24 saatte ne yaptın" without needing the UI panel that will
    eventually display it."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    input_summary TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    error TEXT,
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
    def _row_to_entry(row: sqlite3.Row) -> AuditEntry:
        return AuditEntry(
            row["id"], row["tool_name"], row["risk"], row["input_summary"],
            bool(row["success"]), row["error"], row["created_at"],
        )

    def log(
        self, tool_name: str, risk: str, tool_input: dict[str, Any], success: bool, error: str | None
    ) -> None:
        summary = json.dumps(tool_input, ensure_ascii=False, default=str)[:MAX_INPUT_CHARS]
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO audit_log (tool_name, risk, input_summary, success, error)
                VALUES (?, ?, ?, ?, ?)
                """,
                (tool_name, risk, summary, int(success), error),
            )
            conn.commit()

    def recent(self, hours: float = 24.0, limit: int = 200) -> list[AuditEntry]:
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat(sep=" ")
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE created_at >= ? ORDER BY id DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]
