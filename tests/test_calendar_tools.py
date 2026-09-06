import asyncio

from neo.memory.calendar_store import CalendarStore
from neo.tools.calendar import AddCalendarNoteTool, GetCalendarNotesTool


def test_add_note_without_google_sync(tmp_path):
    store = CalendarStore(tmp_path / "calendar.db")
    tool = AddCalendarNoteTool(store)

    result = asyncio.run(tool.run(date="2026-09-10", text="Doktor randevusu"))

    assert result.success
    assert result.data["synced_to_google"] is False
    assert store.get_notes("2026-09-10")[0].text == "Doktor randevusu"


def test_add_note_rejects_bad_date_format(tmp_path):
    store = CalendarStore(tmp_path / "calendar.db")
    tool = AddCalendarNoteTool(store)

    result = asyncio.run(tool.run(date="10 Eylül", text="Bir şey"))

    assert not result.success
    assert store.get_notes("10 Eylül") == []


def test_get_notes_reports_free_when_empty(tmp_path):
    store = CalendarStore(tmp_path / "calendar.db")
    tool = GetCalendarNotesTool(store)

    result = asyncio.run(tool.run(date="2026-09-10"))

    assert result.success
    assert result.data["free"] is True
    assert result.data["notes"] == []


def test_get_notes_returns_added_notes(tmp_path):
    store = CalendarStore(tmp_path / "calendar.db")
    store.add_note("2026-09-10", "Toplantı")
    tool = GetCalendarNotesTool(store)

    result = asyncio.run(tool.run(date="2026-09-10"))

    assert result.data["free"] is False
    assert result.data["notes"] == ["Toplantı"]
