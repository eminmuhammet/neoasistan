"""TaskStore: persists the planner's tasks and steps so a long-running
task's record survives NEO closing mid-task."""

from neo.memory.task_store import TaskStore


def _store(tmp_path):
    return TaskStore(tmp_path / "tasks.db")


def test_create_task_starts_in_planning_status(tmp_path):
    store = _store(tmp_path)
    task_id = store.create_task("Masaüstünü düzenle")

    task = store.get_task(task_id)
    assert task.goal == "Masaüstünü düzenle"
    assert task.status == "planning"
    assert task.steps == []


def test_get_missing_task_returns_none(tmp_path):
    assert _store(tmp_path).get_task(999) is None


def test_steps_come_back_in_position_order(tmp_path):
    store = _store(tmp_path)
    task_id = store.create_task("Hedef")
    store.add_step(task_id, 2, "üçüncü adım")
    store.add_step(task_id, 0, "birinci adım")
    store.add_step(task_id, 1, "ikinci adım")

    steps = store.get_task(task_id).steps
    assert [s.description for s in steps] == ["birinci adım", "ikinci adım", "üçüncü adım"]


def test_new_steps_start_pending_with_zero_attempts(tmp_path):
    store = _store(tmp_path)
    task_id = store.create_task("Hedef")
    store.add_step(task_id, 0, "bir adım")

    step = store.get_task(task_id).steps[0]
    assert step.status == "pending"
    assert step.attempts == 0
    assert step.result_summary is None


def test_update_task_status(tmp_path):
    store = _store(tmp_path)
    task_id = store.create_task("Hedef")

    store.update_task_status(task_id, "running")
    assert store.get_task(task_id).status == "running"

    store.update_task_status(task_id, "done")
    assert store.get_task(task_id).status == "done"


def test_update_step_only_changes_given_fields(tmp_path):
    """None means "leave unchanged", not "clear this field" -- calling
    update_step(status="running") mid-task must not wipe out an
    already-recorded result_summary or attempts count."""
    store = _store(tmp_path)
    task_id = store.create_task("Hedef")
    step_id = store.add_step(task_id, 0, "bir adım")

    store.update_step(step_id, attempts=1, result_summary="ilk deneme notu")
    store.update_step(step_id, status="running")

    step = store.get_task(task_id).steps[0]
    assert step.status == "running"
    assert step.attempts == 1
    assert step.result_summary == "ilk deneme notu"


def test_update_step_with_no_fields_is_a_no_op(tmp_path):
    store = _store(tmp_path)
    task_id = store.create_task("Hedef")
    step_id = store.add_step(task_id, 0, "bir adım")

    store.update_step(step_id)  # nothing passed

    step = store.get_task(task_id).steps[0]
    assert step.status == "pending"


def test_recent_tasks_returns_newest_first(tmp_path):
    store = _store(tmp_path)
    first = store.create_task("İlk görev")
    second = store.create_task("İkinci görev")

    tasks = store.recent_tasks(limit=10)
    assert [t.id for t in tasks] == [second, first]


def test_recent_tasks_respects_the_limit(tmp_path):
    store = _store(tmp_path)
    for i in range(5):
        store.create_task(f"görev {i}")

    assert len(store.recent_tasks(limit=2)) == 2


def test_recent_tasks_include_their_steps(tmp_path):
    store = _store(tmp_path)
    task_id = store.create_task("Hedef")
    store.add_step(task_id, 0, "adım")

    tasks = store.recent_tasks()
    assert len(tasks[0].steps) == 1
