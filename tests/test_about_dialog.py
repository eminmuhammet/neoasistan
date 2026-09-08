"""The About dialog is where a user checks whether an update landed.

It carried a hardcoded "v0.1.0". After an update genuinely installed and NEO
restarted at 0.1.5, this screen still said 0.1.0 -- so a working update
looked like a failed one, and the real fix got chased for another round.
"""

import re
from pathlib import Path

from neo.config.version import __version__
from neo.ui.main_window import build_about_text

SOURCE = Path(__file__).resolve().parent.parent / "neo" / "ui" / "main_window.py"


def test_about_reports_the_running_version():
    assert __version__ in build_about_text()


def test_about_follows_the_version_it_is_given(monkeypatch):
    """Behaviour, not source text: change the version the module sees and the
    dialog must change with it. A hardcoded literal fails this even though it
    happens to be correct on the day it was written."""
    import neo.ui.main_window as mw

    monkeypatch.setattr(mw, "__version__", "9.9.9")
    rendered = mw.build_about_text()

    assert "9.9.9" in rendered
    assert not re.search(r"\b0\.1\.\d+\b", rendered)


def test_about_uses_the_configured_wake_phrase():
    """It told users to say "Neo" long after the phrase became "Neo uyan"."""
    assert "Neo uyan" in build_about_text("Neo uyan")
    assert "bilgisayar dinle" in build_about_text("bilgisayar dinle")


def test_about_does_not_claim_a_stale_roadmap_position():
    assert "Phase" not in build_about_text()


def test_the_info_button_is_actually_wired_into_the_header():
    """The dialog is unreachable without the button that opens it."""
    source = SOURCE.read_text(encoding="utf-8")
    assert 'QPushButton("ℹ")' in source, "başlıkta ℹ butonu yok"
    assert "clicked.connect(self._on_info_clicked)" in source, "ℹ butonu bağlı değil"


def test_about_lists_what_neo_can_actually_do():
    """It advertised a handful of tools long after there were dozens, so the
    one screen explaining NEO undersold it."""
    text = build_about_text()

    for capability in ("takvim", "posta", "hava durumu", "ekran görüntüsü",
                       "ses seviyesi", "zamanla"):
        assert capability.lower() in text.lower(), f"eksik: {capability}"
