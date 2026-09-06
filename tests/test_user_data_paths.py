"""Where a packaged NEO keeps user data.

The packaged build wrote its calendar, wake-word enrollment and Google token
into the bundle's `_internal` folder -- the exact directory the updater
replaces, so every update would have wiped them.
"""

import sys

import neo.config.settings as settings_module
from neo.config.settings import PROJECT_ROOT, _migrate_legacy_data, user_data_root


def test_source_run_uses_the_project_folder(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert user_data_root() == PROJECT_ROOT


def test_packaged_run_uses_local_appdata(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    root = user_data_root()

    assert root == tmp_path / "NEO"
    # Must not be anywhere under the installation, which the updater replaces.
    assert PROJECT_ROOT not in root.parents


def test_packaged_run_falls_back_to_home_without_localappdata(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    assert user_data_root().name == "NEO"


def test_first_packaged_run_inherits_an_existing_setup(monkeypatch, tmp_path):
    """Otherwise the packaged build starts blank and the user has to redo
    the API key, the calendar and the wake-word enrollment they already did."""
    legacy = tmp_path / "legacy"
    (legacy / "data").mkdir(parents=True)
    (legacy / "data" / "calendar.db").write_text("notes")
    (legacy / "data" / "wake_word_templates.npz").write_text("voice")
    (legacy / ".env").write_text("ANTHROPIC_API_KEY=x")
    (legacy / "credentials.json").write_text("{}")
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", legacy)

    target = tmp_path / "appdata" / "NEO"
    _migrate_legacy_data(target)

    assert (target / "data" / "calendar.db").read_text() == "notes"
    assert (target / "data" / "wake_word_templates.npz").read_text() == "voice"
    assert (target / ".env").read_text() == "ANTHROPIC_API_KEY=x"
    assert (target / "credentials.json").exists()


def test_migration_copies_rather_than_moves(monkeypatch, tmp_path):
    """The source checkout has to keep working after the packaged build runs."""
    legacy = tmp_path / "legacy"
    (legacy / "data").mkdir(parents=True)
    (legacy / "data" / "calendar.db").write_text("notes")
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", legacy)

    _migrate_legacy_data(tmp_path / "appdata" / "NEO")

    assert (legacy / "data" / "calendar.db").exists()


def test_migration_never_overwrites_existing_user_data(monkeypatch, tmp_path):
    """It runs on every packaged start, so it must be a one-time inheritance
    and never clobber data the packaged build has since accumulated."""
    legacy = tmp_path / "legacy"
    (legacy / "data").mkdir(parents=True)
    (legacy / "data" / "calendar.db").write_text("eski")
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", legacy)

    target = tmp_path / "appdata" / "NEO"
    (target / "data").mkdir(parents=True)
    (target / "data" / "calendar.db").write_text("guncel")

    _migrate_legacy_data(target)

    assert (target / "data" / "calendar.db").read_text() == "guncel"


def test_migration_failure_does_not_crash_startup(monkeypatch, tmp_path):
    """A failed copy must leave NEO starting with empty config, not dead."""
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path / "yok")

    _migrate_legacy_data(tmp_path / "hedef")  # must not raise
