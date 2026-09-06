from __future__ import annotations

import asyncio
import logging
from datetime import date as date_cls

from ..memory.calendar_store import CalendarStore
from .base import RiskLevel, Tool, ToolResult
from .google_calendar import GoogleCalendarSync, GoogleCalendarUnavailableError

logger = logging.getLogger(__name__)


class AddCalendarNoteTool(Tool):
    name = "add_calendar_note"
    description = (
        "Belirli bir tarihe not/hatırlatma/etkinlik ekler. 'date' MUTLAKA "
        "YYYY-MM-DD formatında olmalı -- 'bugün', 'yarın', 'cuma' gibi göreli "
        "ifadeleri sen (mevcut tarihi bilerek) gerçek bir tarihe çevir."
    )
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "YYYY-MM-DD formatında tarih."},
            "text": {"type": "string", "description": "Not/etkinlik metni."},
        },
        "required": ["date", "text"],
    }

    def __init__(self, store: CalendarStore, google_sync: GoogleCalendarSync | None = None) -> None:
        self._store = store
        self._google_sync = google_sync

    async def run(self, date: str, text: str, **kwargs: object) -> ToolResult:
        try:
            date_cls.fromisoformat(date)
        except ValueError:
            return ToolResult(success=False, error="Tarih formatı YYYY-MM-DD olmalı.")

        synced = False
        google_event_id = None
        if self._google_sync is not None and self._google_sync.is_configured():
            try:
                google_event_id = await self._google_sync.add_event(date, text)
                synced = True
            except GoogleCalendarUnavailableError:
                logger.warning("Google Takvim senkronizasyonu atlandı")

        note_id = await asyncio.to_thread(self._store.add_note, date, text, google_event_id)
        return ToolResult(
            success=True,
            data={"id": note_id, "date": date, "text": text, "synced_to_google": synced},
        )


class GetCalendarNotesTool(Tool):
    name = "get_calendar_notes"
    description = (
        "Belirli bir tarihteki notları/etkinlikleri döndürür. 'date' MUTLAKA "
        "YYYY-MM-DD formatında olmalı."
    )
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {"date": {"type": "string", "description": "YYYY-MM-DD formatında tarih."}},
        "required": ["date"],
    }

    def __init__(self, store: CalendarStore) -> None:
        self._store = store

    async def run(self, date: str, **kwargs: object) -> ToolResult:
        notes = await asyncio.to_thread(self._store.get_notes, date)
        return ToolResult(
            success=True,
            data={
                "date": date,
                "notes": [n.text for n in notes],
                # Ids come back too so a follow-up "bunu sil" has something
                # to refer to; without them the delete tool had no way to
                # name which note the user meant.
                "items": [{"id": n.id, "text": n.text} for n in notes],
                "free": len(notes) == 0,
            },
        )


class DeleteCalendarNoteTool(Tool):
    name = "delete_calendar_note"
    description = (
        "Bir takvim notunu/etkinliğini siler. Önce get_calendar_notes ile o "
        "günün notlarını al, silinecek notun 'id' değerini oradan kullan. "
        "Hangi notun kastedildiği belirsizse silme, kullanıcıya sor."
    )
    # Deleting the user's own data: the permission layer asks before this
    # runs, rather than silently discarding something they can't get back.
    risk = RiskLevel.MEDIUM
    input_schema = {
        "type": "object",
        "properties": {
            "note_id": {"type": "integer", "description": "Silinecek notun id değeri."}
        },
        "required": ["note_id"],
    }

    def __init__(self, store: CalendarStore, google_sync: GoogleCalendarSync | None = None) -> None:
        self._store = store
        self._google_sync = google_sync

    async def run(self, note_id: int, **kwargs: object) -> ToolResult:
        note = await asyncio.to_thread(self._store.delete_note, int(note_id))
        if note is None:
            return ToolResult(success=False, error=f"{note_id} numaralı not bulunamadı.")

        google_removed = False
        if (
            note.google_event_id
            and self._google_sync is not None
            and self._google_sync.is_configured()
        ):
            try:
                await self._google_sync.delete_event(note.google_event_id)
                google_removed = True
            except GoogleCalendarUnavailableError:
                # The local note is already gone; say so honestly rather than
                # reporting a clean sync that didn't happen.
                logger.warning("Google Takvim'den silme atlandı")

        return ToolResult(
            success=True,
            data={
                "deleted": note.text,
                "date": note.date,
                "removed_from_google": google_removed,
            },
        )


class UpdateCalendarNoteTool(Tool):
    name = "update_calendar_note"
    description = (
        "Var olan bir takvim notunun metnini değiştirir (ör. saatini veya "
        "içeriğini güncellemek için). Notun 'id' değerini get_calendar_notes "
        "ile al."
    )
    risk = RiskLevel.MEDIUM
    input_schema = {
        "type": "object",
        "properties": {
            "note_id": {"type": "integer", "description": "Güncellenecek notun id değeri."},
            "text": {"type": "string", "description": "Notun yeni metni."},
        },
        "required": ["note_id", "text"],
    }

    def __init__(self, store: CalendarStore, google_sync: GoogleCalendarSync | None = None) -> None:
        self._store = store
        self._google_sync = google_sync

    async def run(self, note_id: int, text: str, **kwargs: object) -> ToolResult:
        note = await asyncio.to_thread(self._store.update_note, int(note_id), text)
        if note is None:
            return ToolResult(success=False, error=f"{note_id} numaralı not bulunamadı.")

        google_updated = False
        if (
            note.google_event_id
            and self._google_sync is not None
            and self._google_sync.is_configured()
        ):
            try:
                await self._google_sync.update_event(note.google_event_id, text)
                google_updated = True
            except GoogleCalendarUnavailableError:
                logger.warning("Google Takvim güncellemesi atlandı")

        return ToolResult(
            success=True,
            data={"id": note.id, "date": note.date, "text": note.text, "synced_to_google": google_updated},
        )
