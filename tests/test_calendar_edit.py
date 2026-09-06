import asyncio
from pathlib import Path

import pytest

from neo.memory.calendar_store import CalendarStore
from neo.tools.base import RiskLevel
from neo.tools.calendar import (
    AddCalendarNoteTool,
    DeleteCalendarNoteTool,
    GetCalendarNotesTool,
    UpdateCalendarNoteTool,
)


@pytest.fixture
def store(tmp_path):
    return CalendarStore(tmp_path / "calendar.db")


def test_notes_can_be_deleted(store):
    """Observed live: asked to remove two events, NEO answered that it had no
    tool for deleting notes -- only adding and reading."""
    note_id = store.add_note("2026-09-07", "Saat 13:00 - Buluşma")

    result = asyncio.run(DeleteCalendarNoteTool(store).run(note_id=note_id))

    assert result.success is True
    assert result.data["deleted"] == "Saat 13:00 - Buluşma"
    assert store.get_notes("2026-09-07") == []


def test_deleting_a_missing_note_reports_failure(store):
    result = asyncio.run(DeleteCalendarNoteTool(store).run(note_id=999))

    assert result.success is False
    assert "bulunamadı" in result.error


def test_deleting_one_note_leaves_the_others(store):
    keep = store.add_note("2026-09-07", "Saat 09:00 - Spor")
    drop = store.add_note("2026-09-07", "Saat 13:00 - Buluşma")

    asyncio.run(DeleteCalendarNoteTool(store).run(note_id=drop))

    assert [n.id for n in store.get_notes("2026-09-07")] == [keep]


def test_notes_can_be_updated(store):
    note_id = store.add_note("2026-09-07", "Saat 13:00 - Buluşma")

    result = asyncio.run(
        UpdateCalendarNoteTool(store).run(note_id=note_id, text="Saat 16:00 - Buluşma")
    )

    assert result.success is True
    assert store.get_notes("2026-09-07")[0].text == "Saat 16:00 - Buluşma"


def test_updating_a_missing_note_reports_failure(store):
    result = asyncio.run(UpdateCalendarNoteTool(store).run(note_id=999, text="yeni"))
    assert result.success is False


def test_listing_exposes_ids_for_follow_up_edits(store):
    """Without ids in the listing, a follow-up "bunu sil" has no way to say
    which note it means."""
    store.add_note("2026-09-07", "Saat 09:00 - Spor")

    result = asyncio.run(GetCalendarNotesTool(store).run(date="2026-09-07"))

    assert result.data["items"][0]["text"] == "Saat 09:00 - Spor"
    assert isinstance(result.data["items"][0]["id"], int)


def test_destructive_calendar_tools_require_confirmation(store):
    """Removing the user's own data goes through the permission dialog; adding
    and reading do not."""
    assert DeleteCalendarNoteTool(store).risk is RiskLevel.MEDIUM
    assert UpdateCalendarNoteTool(store).risk is RiskLevel.MEDIUM
    assert AddCalendarNoteTool(store).risk is RiskLevel.LOW
    assert GetCalendarNotesTool(store).risk is RiskLevel.LOW


def test_all_day_event_ends_the_following_day():
    """Google treats end.date as exclusive for all-day events, so a one-day
    event ends on the next date. Sending end == start describes a zero-length
    event -- accepted, but rendered oddly by some calendar clients."""
    from datetime import date, timedelta

    from neo.tools.google_calendar import GoogleCalendarSync

    captured = {}

    class FakeEvents:
        def insert(self, calendarId, body):
            captured.update(body)
            return self

        def execute(self):
            return {"id": "evt1"}

    class FakeService:
        def events(self):
            return FakeEvents()

    sync = GoogleCalendarSync(Path("nope.json"), Path("nope.json"))
    sync._service = FakeService()

    sync._add_event_sync("2026-09-06", "Toplantı")

    assert captured["start"] == {"date": "2026-09-06"}
    assert captured["end"] == {
        "date": (date(2026, 9, 6) + timedelta(days=1)).isoformat()
    }


def test_google_sync_failure_still_deletes_locally(store):
    """The local note is already gone by then, so the result must say the
    Google side didn't happen rather than implying a clean sync."""

    class FailingSync:
        def is_configured(self):
            return True

        async def delete_event(self, event_id):
            from neo.tools.google_calendar import GoogleCalendarUnavailableError

            raise GoogleCalendarUnavailableError("offline")

    note_id = store.add_note("2026-09-07", "Toplantı", google_event_id="evt123")

    result = asyncio.run(
        DeleteCalendarNoteTool(store, google_sync=FailingSync()).run(note_id=note_id)
    )

    assert result.success is True
    assert result.data["removed_from_google"] is False
    assert store.get_notes("2026-09-07") == []
