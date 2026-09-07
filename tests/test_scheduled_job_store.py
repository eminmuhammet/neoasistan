from neo.memory.scheduled_job_store import ScheduledJobStore


def _store(tmp_path):
    return ScheduledJobStore(tmp_path / "jobs.db")


def test_create_job_starts_enabled(tmp_path):
    store = _store(tmp_path)
    job_id = store.create_job("sabah brifingi", "özet ver", "daily", "08:00", "2026-09-08T08:00:00")

    job = store.get_job(job_id)
    assert job.name == "sabah brifingi"
    assert job.enabled is True
    assert job.last_run_at is None


def test_get_missing_job_returns_none(tmp_path):
    assert _store(tmp_path).get_job(999) is None


def test_list_jobs_can_filter_to_enabled_only(tmp_path):
    store = _store(tmp_path)
    active = store.create_job("aktif", "x", "daily", "08:00", "2026-09-08T08:00:00")
    disabled = store.create_job("pasif", "x", "daily", "08:00", "2026-09-08T08:00:00")
    store.set_enabled(disabled, False)

    all_jobs = store.list_jobs()
    enabled_jobs = store.list_jobs(enabled_only=True)

    assert {j.id for j in all_jobs} == {active, disabled}
    assert {j.id for j in enabled_jobs} == {active}


def test_find_by_name(tmp_path):
    store = _store(tmp_path)
    store.create_job("sabah brifingi", "x", "daily", "08:00", "2026-09-08T08:00:00")

    assert store.find_by_name("sabah brifingi") is not None
    assert store.find_by_name("yok öyle bir şey") is None


def test_record_run_updates_last_run_and_next_run(tmp_path):
    store = _store(tmp_path)
    job_id = store.create_job("iş", "x", "daily", "08:00", "2026-09-08T08:00:00")

    store.record_run(job_id, "2026-09-08T08:00:05", "2026-09-09T08:00:00")

    job = store.get_job(job_id)
    assert job.last_run_at == "2026-09-08T08:00:05"
    assert job.next_run_at == "2026-09-09T08:00:00"
    assert job.enabled is True


def test_record_run_with_no_next_run_disables_the_job(tmp_path):
    """A "once" job has nothing left to schedule after it fires -- it must
    disable itself rather than staying enabled with a stale next_run_at
    that would make it look due forever."""
    store = _store(tmp_path)
    job_id = store.create_job("tek seferlik", "x", "once", "2026-09-08T09:00:00", "2026-09-08T09:00:00")

    store.record_run(job_id, "2026-09-08T09:00:03", None)

    job = store.get_job(job_id)
    assert job.enabled is False
    assert job.last_run_at == "2026-09-08T09:00:03"


def test_delete_job(tmp_path):
    store = _store(tmp_path)
    job_id = store.create_job("iş", "x", "daily", "08:00", "2026-09-08T08:00:00")

    assert store.delete_job(job_id) is True
    assert store.get_job(job_id) is None


def test_delete_missing_job_reports_nothing_removed(tmp_path):
    assert _store(tmp_path).delete_job(999) is False
