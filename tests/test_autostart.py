from pathlib import Path

from neo.config import autostart


def test_launch_command_points_at_the_launcher_script_by_full_path():
    command = autostart._launch_command()

    # Must not rely on the working directory: Windows starts logon entries
    # from an arbitrary cwd, so `-m neo.main` would fail to find the package.
    assert "-m neo.main" not in command
    assert "run_neo.py" in command
    assert command.startswith('"')


def test_launcher_script_exists_where_the_command_points():
    project_root = Path(__file__).resolve().parent.parent
    assert (project_root / "run_neo.py").exists()


def test_enable_then_disable_round_trips(monkeypatch):
    # Uses the real registry under HKCU\...\Run, then removes the entry
    # again, so the machine is left exactly as it was found.
    was_enabled = autostart.is_enabled()
    try:
        assert autostart.enable() is True
        assert autostart.is_enabled() is True
        assert autostart.disable() is True
        assert autostart.is_enabled() is False
    finally:
        if was_enabled:
            autostart.enable()
        else:
            autostart.disable()


def test_disable_is_idempotent_when_not_registered():
    was_enabled = autostart.is_enabled()
    try:
        autostart.disable()
        assert autostart.disable() is True
    finally:
        if was_enabled:
            autostart.enable()
