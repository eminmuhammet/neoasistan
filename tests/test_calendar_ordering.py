from neo.memory.calendar_store import CalendarStore


def _store(tmp_path):
    return CalendarStore(tmp_path / "calendar.db")


def test_notes_come_back_in_time_order_not_insertion_order(tmp_path):
    """Observed live: asking "yarın programımda neler var?" answered with the
    15:00 appointment before the 13:00 one, because the 15:00 note had been
    added first."""
    store = _store(tmp_path)
    store.add_note("2026-09-07", "Saat 15:00 - Diş hekimi randevusu")
    store.add_note("2026-09-07", "Saat 13:00 - Buluşma")
    store.add_note("2026-09-07", "Saat 09:30 - Spor")

    texts = [note.text for note in store.get_notes("2026-09-07")]
    assert texts == [
        "Saat 09:30 - Spor",
        "Saat 13:00 - Buluşma",
        "Saat 15:00 - Diş hekimi randevusu",
    ]


def test_untimed_notes_sort_after_timed_ones(tmp_path):
    store = _store(tmp_path)
    store.add_note("2026-09-07", "Annemi ara")
    store.add_note("2026-09-07", "Saat 14:00 - Toplantı")

    texts = [note.text for note in store.get_notes("2026-09-07")]
    assert texts == ["Saat 14:00 - Toplantı", "Annemi ara"]


def test_dotted_and_bare_times_are_understood(tmp_path):
    store = _store(tmp_path)
    store.add_note("2026-09-07", "18.45 akşam yemeği")
    store.add_note("2026-09-07", "07:15 kalk")

    texts = [note.text for note in store.get_notes("2026-09-07")]
    assert texts == ["07:15 kalk", "18.45 akşam yemeği"]


def test_same_time_notes_keep_insertion_order(tmp_path):
    store = _store(tmp_path)
    store.add_note("2026-09-07", "Saat 10:00 - önce bu")
    store.add_note("2026-09-07", "Saat 10:00 - sonra bu")

    texts = [note.text for note in store.get_notes("2026-09-07")]
    assert texts == ["Saat 10:00 - önce bu", "Saat 10:00 - sonra bu"]


def test_a_date_in_the_text_is_not_read_as_a_time(tmp_path):
    store = _store(tmp_path)
    store.add_note("2026-09-07", "Saat 08:00 - erken iş")
    store.add_note("2026-09-07", "Fatura son ödeme 30.09 tarihinde")

    texts = [note.text for note in store.get_notes("2026-09-07")]
    assert texts[0] == "Saat 08:00 - erken iş"


def test_other_days_are_unaffected(tmp_path):
    store = _store(tmp_path)
    store.add_note("2026-09-07", "Saat 13:00 - Buluşma")
    store.add_note("2026-09-08", "Saat 09:00 - Başka gün")

    assert len(store.get_notes("2026-09-07")) == 1
    assert store.is_free("2026-09-09") is True
