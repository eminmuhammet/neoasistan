from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Preference:
    key: str
    value: str
    learned_at: str
    source: str


# Defense in depth: the system prompt tells Claude never to remember a
# password, national ID or card number, but that is a soft instruction --
# a model can still misjudge one case. This is a second, deterministic gate
# at the actual persistence boundary, so a slip in the prompt can't turn
# into a credential sitting in plaintext on disk. Matched against the key
# only (not the value): a value's digits alone can't be told apart from a
# phone number, which is a legitimate thing to remember, but a key that
# names a credential is unambiguous regardless of wording.
_SENSITIVE_KEY_MARKERS = (
    "şifre",
    "sifre",
    "parola",
    "password",
    "pin",
    "cvv",
    "cvc",
    "kart no",
    "kart numarası",
    "kart numarasi",
    "iban",
    "tc kimlik",
    "tc no",
    "kimlik no",
    "kimlik numarası",
    "kimlik numarasi",
)


class SensitivePreferenceError(ValueError):
    """Raised when a preference looks like a credential or an ID/card
    number, which must never be written to disk in plain text."""


def _normalize_key(key: str) -> str:
    """Case/whitespace-insensitive so "Şehir" and "şehir " land on the same
    row -- Claude's own phrasing of a key can vary slightly between calls,
    and without this each variant would silently pile up as a separate,
    stale fact instead of the later one correcting the earlier one.

    Turkish "İ" is replaced before .lower() because Python's default
    casing doesn't know Turkish rules: "İ".lower() produces "i" plus a
    combining dot above (U+0307), not plain ASCII "i" -- so "ŞEHİR" and
    "şehir" would otherwise normalize to two different strings and silently
    fail to match, the same class of bug already worked around in
    phrase_match.py.
    """
    folded = key.replace("I", "ı").replace("İ", "i").lower()
    return " ".join(folded.strip().split())


class PreferenceStore:
    """SQLite-backed store for durable facts NEO has learned about the user
    ("şehri Bursa", "bana Emin diye hitap et") -- distinct from conversation
    history (ConversationStore), which is a transcript of what was said, not
    a queryable set of facts currently believed true.

    Keyed by a normalized `key`, so telling NEO "aslında İstanbul'da
    yaşıyorum" after it already knows "şehir: Bursa" overwrites that one row
    instead of leaving both a correct and a stale answer for the same
    question.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    learned_at TEXT NOT NULL DEFAULT (datetime('now')),
                    source TEXT NOT NULL DEFAULT 'sohbet'
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def remember(self, key: str, value: str, source: str = "sohbet") -> None:
        normalized = _normalize_key(key)
        if not normalized:
            raise ValueError("Tercih anahtarı boş olamaz.")
        if any(marker in normalized for marker in _SENSITIVE_KEY_MARKERS):
            raise SensitivePreferenceError(
                f"'{key}' bir kimlik bilgisi gibi görünüyor, kalıcı olarak saklanmaz."
            )
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO preferences (key, value, source) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    learned_at = datetime('now'),
                    source = excluded.source
                """,
                (normalized, value.strip(), source),
            )
            conn.commit()

    def forget(self, key: str) -> bool:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "DELETE FROM preferences WHERE key = ?", (_normalize_key(key),)
            )
            conn.commit()
            return cursor.rowcount > 0

    def get(self, key: str) -> Preference | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT key, value, learned_at, source FROM preferences WHERE key = ?",
                (_normalize_key(key),),
            ).fetchone()
        if row is None:
            return None
        return Preference(row["key"], row["value"], row["learned_at"], row["source"])

    def all(self) -> list[Preference]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT key, value, learned_at, source FROM preferences ORDER BY key"
            ).fetchall()
        return [Preference(r["key"], r["value"], r["learned_at"], r["source"]) for r in rows]
