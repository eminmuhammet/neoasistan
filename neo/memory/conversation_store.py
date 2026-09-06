from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    id: int
    timestamp: str
    role: str
    text: str


class ConversationStore:
    """Durable conversation history.

    Writes to two places in the same directory:
      * `conversations.db` -- SQLite, what NEO reads back to remember earlier
        exchanges across restarts.
      * `konusmalar/YYYY-MM-DD.md` -- a plain-text daily transcript, so the
        history is readable from any other device (phone included) by just
        opening the file, without needing an app.

    Pointing that directory at a cloud-synced folder (OneDrive by default on
    this machine) is what makes the history available across devices.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._db_path = directory / "conversations.db"
        self._transcript_dir = directory / "konusmalar"
        directory.mkdir(parents=True, exist_ok=True)
        self._transcript_dir.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    role TEXT NOT NULL,
                    text TEXT NOT NULL
                )
                """
            )
            conn.commit()

    @property
    def directory(self) -> Path:
        return self._directory

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add_message(self, role: str, text: str) -> None:
        try:
            with closing(self._connect()) as conn:
                conn.execute("INSERT INTO messages (role, text) VALUES (?, ?)", (role, text))
                conn.commit()
            self._append_transcript(role, text)
        except Exception:
            # History is a convenience, never a reason to break a reply.
            logger.exception("Konuşma kaydı yazılamadı")

    def _append_transcript(self, role: str, text: str) -> None:
        now = datetime.now()
        path = self._transcript_dir / f"{now:%Y-%m-%d}.md"
        speaker = "Sen" if role == "user" else "NEO"
        entry = f"**{now:%H:%M}** · {speaker}\n\n{text}\n\n---\n\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(entry)

    def recent_messages(self, limit: int = 20) -> list[ConversationMessage]:
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT id, timestamp, role, text FROM messages ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        except Exception:
            logger.exception("Konuşma geçmişi okunamadı")
            return []
        messages = [
            ConversationMessage(row["id"], row["timestamp"], row["role"], row["text"])
            for row in rows
        ]
        messages.reverse()
        return messages

    def message_count(self) -> int:
        try:
            with closing(self._connect()) as conn:
                return int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
        except Exception:
            logger.exception("Konuşma sayısı okunamadı")
            return 0
