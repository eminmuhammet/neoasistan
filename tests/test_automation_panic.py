from neo.core import automation_panic


def teardown_function(_fn):
    automation_panic.reset_panic()


def test_panic_starts_disengaged():
    assert automation_panic.is_panic_engaged() is False


def test_engage_panic_sets_flag():
    automation_panic.engage_panic()
    assert automation_panic.is_panic_engaged() is True


def test_reset_panic_clears_flag():
    automation_panic.engage_panic()
    automation_panic.reset_panic()
    assert automation_panic.is_panic_engaged() is False


def test_engage_panic_is_idempotent():
    automation_panic.engage_panic()
    automation_panic.engage_panic()
    assert automation_panic.is_panic_engaged() is True
