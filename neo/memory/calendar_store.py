from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

# Notes are free text ("Saat 15:00 - Diş hekimi randevusu"), so the time of
# day has to be read back out of the text to order a day's schedule.
# Tolerates "15:00", "15.00" and a leading "Saat".
_TIME_PATTERN = re.compile(r"(?:^|\b)(?:saat\s*)?([01]?\d|2[0-3])[:.]([0-5]\d)\b", re.IGNORECASE)


def _time_key(text: str) -> tuple[int, int]:
    """Minutes past midnight for sorting, or a sentinel that sorts last when
    the note carries no time -- an all-day note belongs after the timed ones,
    not silently at 00:00."""
    match = _TIME_PATTERN.search(text)
    if not match:
        return (1, 0)
    return (0, int(match.group(1)) * 60 + int(match.group(2)))


@dataclass
class CalendarNote:
    id: int
    date: str
    text: str
    created_at: str
    google_event_id: str | None


class CalendarStore:
    """SQLite-backed local notes/reminders store, keyed by ISO date
    (YYYY-MM-DD). One file, no server -- this is NEO's local calendar; an
    optional Google Calendar sync can mirror notes there too (see
    neo/tools/google_calendar.py)."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS calendar_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    google_event_id TEXT
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add_note(self, date: str, text: str, google_event_id: str | None = None) -> int:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "INSERT INTO calendar_notes (date, text, google_event_id) VALUES (?, ?, ?)",
                (date, text, google_event_id),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get_notes(self, date: str) -> list[CalendarNote]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, date, text, created_at, google_event_id "
                "FROM calendar_notes WHERE date = ? ORDER BY id",
                (date,),
            ).fetchall()
            notes = [
                CalendarNote(row["id"], row["date"], row["text"], row["created_at"], row["google_event_id"])
                for row in rows
            ]
        # A day's schedule is read chronologically, not in the order the
        # notes happened to be added -- otherwise "yarın neler var?" answers
        # with the 15:00 appointment before the 13:00 one. Insertion order
        # still breaks ties, so same-time notes stay stable.
        return sorted(notes, key=lambda note: (_time_key(note.text), note.id))

    def get_note(self, note_id: int) -> CalendarNote | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT id, date, text, created_at, google_event_id "
                "FROM calendar_notes WHERE id = ?",
                (note_id,),
            ).fetchone()
        if row is None:
            return None
        return CalendarNote(
            row["id"], row["date"], row["text"], row["created_at"], row["google_event_id"]
        )

    def delete_note(self, note_id: int) -> CalendarNote | None:
        """Removes a note, returning what was deleted so the caller can
        confirm it back to the user by name -- and can mirror the deletion to
        Google Calendar, which needs the stored event id."""
        note = self.get_note(note_id)
        if note is None:
            return None
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM calendar_notes WHERE id = ?", (note_id,))
            conn.commit()
        return note

    def update_note(self, note_id: int, text: str) -> CalendarNote | None:
        if self.get_note(note_id) is None:
            return None
        with closing(self._connect()) as conn:
            conn.execute("UPDATE calendar_notes SET text = ? WHERE id = ?", (text, note_id))
            conn.commit()
        return self.get_note(note_id)

    def is_free(self, date: str) -> bool:
        return len(self.get_notes(date)) == 0
