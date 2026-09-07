from datetime import datetime, timedelta

from neo.memory.audit_store import AuditStore


def _store(tmp_path) -> AuditStore:
    return AuditStore(tmp_path / "audit.db")


def test_log_and_recent_round_trip(tmp_path):
    store = _store(tmp_path)
    store.log("click", "high", {"x": 1, "y": 2}, True, None)

    entries = store.recent()

    assert len(entries) == 1
    assert entries[0].tool_name == "click"
    assert entries[0].success is True
    assert entries[0].error is None


def test_log_records_failure_with_error(tmp_path):
    store = _store(tmp_path)
    store.log("delete_calendar_note", "medium", {"id": 5}, False, "izin verilmedi")

    entries = store.recent()

    assert entries[0].success is False
    assert entries[0].error == "izin verilmedi"


def test_input_summary_is_truncated_and_json_encoded(tmp_path):
    store = _store(tmp_path)
    long_text = "a" * 500
    store.log("type_text", "high", {"text": long_text}, True, None)

    entries = store.recent()

    assert len(entries[0].input_summary) <= 200


def test_recent_excludes_entries_older_than_window(tmp_path):
    store = _store(tmp_path)
    old_time = (datetime.now() - timedelta(hours=48)).isoformat(sep=" ")
    import sqlite3
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        conn.execute(
            "INSERT INTO audit_log (tool_name, risk, input_summary, success, error, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("old_tool", "low", "{}", 1, None, old_time),
        )
        conn.commit()
    store.log("new_tool", "low", {}, True, None)

    entries = store.recent(hours=24)

    assert [e.tool_name for e in entries] == ["new_tool"]


def test_recent_orders_newest_first(tmp_path):
    store = _store(tmp_path)
    store.log("first", "low", {}, True, None)
    store.log("second", "low", {}, True, None)

    entries = store.recent()

    assert [e.tool_name for e in entries] == ["second", "first"]


def test_recent_respects_limit(tmp_path):
    store = _store(tmp_path)
    for i in range(5):
        store.log(f"tool{i}", "low", {}, True, None)

    entries = store.recent(limit=2)

    assert len(entries) == 2
