"""Where a packaged NEO keeps user data.

The packaged build wrote its calendar, wake-word enrollment and Google token
into the bundle's `_internal` folder -- the exact directory the updater
replaces, so every update would have wiped them.
"""

import sys
from pathlib import Path

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


def _make_legacy(root: Path) -> Path:
    (root / "data").mkdir(parents=True)
    (root / "run_neo.py").write_text("# launcher")  # what marks a real install
    (root / "data" / "calendar.db").write_text("notes")
    (root / "data" / "wake_word_templates.npz").write_text("voice")
    (root / ".env").write_text("ANTHROPIC_API_KEY=x")
    (root / "credentials.json").write_text("{}")
    return root


def test_packaged_run_inherits_an_existing_setup(monkeypatch, tmp_path):
    """Otherwise the packaged build starts blank and the user has to redo
    the API key, the calendar and the wake-word enrollment they already did."""
    legacy = _make_legacy(tmp_path / "legacy")
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", legacy)

    target = tmp_path / "appdata" / "NEO"
    _migrate_legacy_data(target)

    assert (target / "data" / "calendar.db").read_text() == "notes"
    assert (target / "data" / "wake_word_templates.npz").read_text() == "voice"
    assert (target / ".env").read_text() == "ANTHROPIC_API_KEY=x"
    assert (target / "credentials.json").exists()


def test_home_install_is_searched_too(monkeypatch, tmp_path):
    """Under PyInstaller, PROJECT_ROOT resolves inside the bundle's own
    `_internal` folder -- so migration found only what a previous packaged
    run had left there, copied an empty calendar, and left the API key,
    Google credentials and voice enrollment behind."""
    home = tmp_path / "home"
    _make_legacy(home / "NEO")
    monkeypatch.setattr(settings_module.Path, "home", staticmethod(lambda: home))
    # PROJECT_ROOT points at a bundle folder with nothing useful in it.
    bundle = tmp_path / "dist" / "NEO" / "_internal"
    bundle.mkdir(parents=True)
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", bundle)

    target = tmp_path / "appdata" / "NEO"
    _migrate_legacy_data(target)

    assert (target / ".env").read_text() == "ANTHROPIC_API_KEY=x"
    assert (target / "data" / "wake_word_templates.npz").exists()


def test_a_folder_without_a_launcher_is_not_treated_as_an_install(monkeypatch, tmp_path):
    decoy = tmp_path / "NEO"
    (decoy / "data").mkdir(parents=True)
    (decoy / ".env").write_text("SAHTE=1")
    monkeypatch.setattr(settings_module.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path / "yok")

    target = tmp_path / "appdata" / "NEO"
    _migrate_legacy_data(target)

    assert not (target / ".env").exists()


def test_migration_copies_rather_than_moves(monkeypatch, tmp_path):
    """The source checkout has to keep working after the packaged build runs."""
    legacy = _make_legacy(tmp_path / "legacy")
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", legacy)

    _migrate_legacy_data(tmp_path / "appdata" / "NEO")

    assert (legacy / "data" / "calendar.db").exists()
    assert (legacy / ".env").exists()


def test_migration_never_overwrites_existing_user_data(monkeypatch, tmp_path):
    """It runs on every packaged start, so it must only fill in what's
    missing and never clobber data the packaged build has accumulated."""
    legacy = _make_legacy(tmp_path / "legacy")
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", legacy)

    target = tmp_path / "appdata" / "NEO"
    (target / "data").mkdir(parents=True)
    (target / "data" / "calendar.db").write_text("guncel")

    _migrate_legacy_data(target)

    assert (target / "data" / "calendar.db").read_text() == "guncel"
    # ...while still bringing across what wasn't there yet.
    assert (target / "data" / "wake_word_templates.npz").read_text() == "voice"


def test_migration_fills_gaps_on_a_later_run(monkeypatch, tmp_path):
    """The earlier version bailed out entirely once the target folder
    existed, so a partial first migration could never be completed."""
    legacy = _make_legacy(tmp_path / "legacy")
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", legacy)

    target = tmp_path / "appdata" / "NEO"
    (target / "data").mkdir(parents=True)  # target already exists

    _migrate_legacy_data(target)

    assert (target / ".env").exists()
    assert (target / "data" / "calendar.db").exists()


def test_migration_failure_does_not_crash_startup(monkeypatch, tmp_path):
    """A failed copy must leave NEO starting with empty config, not dead."""
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path / "yok")
    monkeypatch.setattr(settings_module.Path, "home", staticmethod(lambda: tmp_path / "yok"))

    _migrate_legacy_data(tmp_path / "hedef")  # must not raise
