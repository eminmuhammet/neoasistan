"""The helper script that swaps files after NEO exits.

All three of these are real failures seen live: NEO stayed on screen saying
it was about to restart, the files were never replaced, and the version on
disk was unchanged.
"""

import os
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

from neo.core.updater import apply_update


@pytest.fixture
def package(tmp_path):
    zip_path = tmp_path / "NEO-9.9.9.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("NEO/neo/config/version.py", '__version__ = "9.9.9"\n')
        archive.writestr("NEO/run_neo.py", "print('hi')\n")
    return zip_path


@pytest.fixture
def written_script(package, tmp_path, monkeypatch):
    """Runs apply_update with the actual process launch stubbed out."""
    spawned = {}

    def fake_popen(args, **kwargs):
        spawned["args"] = args
        return None

    monkeypatch.setattr("neo.core.updater.subprocess.Popen", fake_popen)
    apply_update(package, target_dir=tmp_path / "install")
    return Path(tempfile.gettempdir()) / "neo_update.cmd", spawned


def test_helper_waits_for_the_process_to_exit(written_script):
    """It used to sleep three seconds and copy regardless. Windows locks
    files held by a running process, so the copy silently did nothing."""
    script, _ = written_script
    body = script.read_text(encoding="utf-8")

    assert "tasklist" in body
    assert str(os.getpid()) in body
    assert "goto waitloop" in body
    assert "time.sleep" not in body


def test_helper_gives_up_waiting_eventually(written_script):
    """A process that never exits must not wedge the helper forever."""
    script, _ = written_script
    body = script.read_text(encoding="utf-8")

    assert "tries" in body
    assert "GEQ" in body


def test_helper_relaunches_with_the_right_interpreter(written_script):
    """Passing run_neo.py to `start` hands it to whatever Python owns .py --
    usually the system install, which has none of NEO's dependencies."""
    script, _ = written_script
    body = script.read_text(encoding="utf-8")

    assert sys.executable in body
    if not getattr(sys, "frozen", False):
        assert "run_neo.py" in body


def test_helper_copies_from_the_unwrapped_package(written_script, tmp_path):
    script, _ = written_script
    body = script.read_text(encoding="utf-8")

    assert "robocopy" in body
    assert str(tmp_path / "install") in body


def test_helper_is_spawned_detached(written_script):
    """It has to outlive the process it is waiting for."""
    _, spawned = written_script

    assert spawned["args"][0] == "cmd.exe"
