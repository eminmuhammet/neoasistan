from neo.memory.calendar_store import CalendarStore


def test_add_and_get_note(tmp_path):
    store = CalendarStore(tmp_path / "calendar.db")
    store.add_note("2026-09-10", "Doktor randevusu")

    notes = store.get_notes("2026-09-10")

    assert len(notes) == 1
    assert notes[0].text == "Doktor randevusu"


def test_is_free_true_when_no_notes(tmp_path):
    store = CalendarStore(tmp_path / "calendar.db")
    assert store.is_free("2026-09-10") is True


def test_is_free_false_after_adding_note(tmp_path):
    store = CalendarStore(tmp_path / "calendar.db")
    store.add_note("2026-09-10", "Toplantı")
    assert store.is_free("2026-09-10") is False


def test_notes_are_isolated_per_date(tmp_path):
    store = CalendarStore(tmp_path / "calendar.db")
    store.add_note("2026-09-10", "A")
    store.add_note("2026-09-11", "B")

    assert [n.text for n in store.get_notes("2026-09-10")] == ["A"]
    assert [n.text for n in store.get_notes("2026-09-11")] == ["B"]


def test_multiple_notes_same_date_preserve_order(tmp_path):
    store = CalendarStore(tmp_path / "calendar.db")
    store.add_note("2026-09-10", "İlk")
    store.add_note("2026-09-10", "İkinci")

    assert [n.text for n in store.get_notes("2026-09-10")] == ["İlk", "İkinci"]


def test_persists_across_instances(tmp_path):
    path = tmp_path / "calendar.db"
    CalendarStore(path).add_note("2026-09-10", "Kalıcı not")

    reopened = CalendarStore(path)
    assert [n.text for n in reopened.get_notes("2026-09-10")] == ["Kalıcı not"]
