"""AccessModeManager: NEO always starts, and always eventually falls back
to, assistant mode. Helper mode (full authority) only exists behind an
explicit unlock and expires on its own after inactivity."""

from neo.core.access_mode import AccessMode, AccessModeManager


def test_starts_in_assistant_mode():
    assert AccessModeManager().mode is AccessMode.ASSISTANT


def test_unlock_switches_to_helper_mode():
    manager = AccessModeManager()
    manager.unlock_helper_mode()
    assert manager.mode is AccessMode.HELPER


def test_drop_returns_to_assistant_mode():
    manager = AccessModeManager()
    manager.unlock_helper_mode()
    manager.drop_to_assistant_mode()
    assert manager.mode is AccessMode.ASSISTANT


def test_helper_mode_expires_after_the_idle_timeout(monkeypatch):
    import neo.core.access_mode as am

    clock = [1000.0]
    monkeypatch.setattr(am.time, "monotonic", lambda: clock[0])

    manager = AccessModeManager(idle_timeout_seconds=600.0)
    manager.unlock_helper_mode()
    assert manager.mode is AccessMode.HELPER

    clock[0] += 599.0
    assert manager.mode is AccessMode.HELPER  # not expired yet

    clock[0] += 2.0
    assert manager.mode is AccessMode.ASSISTANT  # now past 600s idle


def test_touch_resets_the_idle_countdown(monkeypatch):
    """Talking to NEO in helper mode -- even about something that doesn't
    need elevated authority -- must keep it from expiring mid-conversation."""
    import neo.core.access_mode as am

    clock = [1000.0]
    monkeypatch.setattr(am.time, "monotonic", lambda: clock[0])

    manager = AccessModeManager(idle_timeout_seconds=600.0)
    manager.unlock_helper_mode()

    clock[0] += 590.0
    manager.touch()
    clock[0] += 590.0  # would be 1180s since unlock, but only 590s since touch()

    assert manager.mode is AccessMode.HELPER


def test_seconds_until_drop_is_none_in_assistant_mode():
    assert AccessModeManager().seconds_until_drop() is None


def test_seconds_until_drop_counts_down_in_helper_mode(monkeypatch):
    import neo.core.access_mode as am

    clock = [1000.0]
    monkeypatch.setattr(am.time, "monotonic", lambda: clock[0])

    manager = AccessModeManager(idle_timeout_seconds=600.0)
    manager.unlock_helper_mode()

    clock[0] += 100.0
    remaining = manager.seconds_until_drop()
    assert remaining is not None
    assert 490 <= remaining <= 500


def test_on_change_callback_fires_on_real_transitions_only():
    """Must not fire for a no-op call (already in that mode) -- a UI badge
    wired to this would otherwise flash/re-render for nothing."""
    changes = []
    manager = AccessModeManager(on_change=changes.append)

    manager.drop_to_assistant_mode()  # already assistant: no-op
    assert changes == []

    manager.unlock_helper_mode()
    assert changes == [AccessMode.HELPER]

    manager.unlock_helper_mode()  # already helper: no-op
    assert changes == [AccessMode.HELPER]

    manager.drop_to_assistant_mode()
    assert changes == [AccessMode.HELPER, AccessMode.ASSISTANT]


def test_on_change_callback_fires_on_automatic_expiry(monkeypatch):
    import neo.core.access_mode as am

    clock = [1000.0]
    monkeypatch.setattr(am.time, "monotonic", lambda: clock[0])
    changes = []

    manager = AccessModeManager(idle_timeout_seconds=600.0, on_change=changes.append)
    manager.unlock_helper_mode()

    clock[0] += 601.0
    _ = manager.mode  # triggers the expiry check

    assert changes == [AccessMode.HELPER, AccessMode.ASSISTANT]
