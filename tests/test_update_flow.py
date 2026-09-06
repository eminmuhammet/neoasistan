"""The updater existed but nothing ever called it -- these cover the wiring
that now runs it from the app."""

import asyncio

import pytest

from neo.core.updater import UpdateError, UpdateInfo


class FakeWindow:
    """Only the update surface of MainWindow, so the flow can be exercised
    without a Qt application."""

    def __init__(self, manifest_url=None, approve=True):
        from neo.ui.main_window import MainWindow

        self._update_manifest_url = manifest_url
        self._pending_update = None
        self.messages = []
        self.button_text = None
        self.button_visible = False
        self._approve = approve
        self.applied = None
        self.check_for_update = MainWindow.check_for_update.__get__(self)

    def _append(self, sender, text):
        self.messages.append(text)

    class _Button:
        def __init__(self, owner):
            self.owner = owner

        def setText(self, text):
            self.owner.button_text = text

        def show(self):
            self.owner.button_visible = True

        def setEnabled(self, value):
            pass

    @property
    def _update_button(self):
        return self._Button(self)


def test_no_manifest_url_stays_silent():
    window = FakeWindow(manifest_url=None)
    asyncio.run(window.check_for_update())

    assert window.messages == []
    assert window.button_visible is False


def test_no_manifest_url_can_report_when_asked():
    window = FakeWindow(manifest_url=None)
    asyncio.run(window.check_for_update(announce_when_current=True))

    assert "tanımlı değil" in window.messages[0]


def test_unreachable_server_is_silent_on_startup(monkeypatch):
    """An assistant that has to work offline shouldn't nag about a failed
    update check the user never asked for."""
    import neo.ui.main_window as mw

    async def failing(url):
        raise UpdateError("Güncelleme sunucusuna ulaşamadım.")

    monkeypatch.setattr(mw, "check_for_update", failing)
    window = FakeWindow(manifest_url="https://example.com/latest.json")
    asyncio.run(window.check_for_update())

    assert window.messages == []
    assert window.button_visible is False


def test_up_to_date_is_silent_on_startup(monkeypatch):
    import neo.ui.main_window as mw

    async def current(url):
        return None

    monkeypatch.setattr(mw, "check_for_update", current)
    window = FakeWindow(manifest_url="https://example.com/latest.json")
    asyncio.run(window.check_for_update())

    assert window.messages == []
    assert window.button_visible is False


def test_newer_version_offers_the_update(monkeypatch):
    import neo.ui.main_window as mw

    info = UpdateInfo(
        version="9.9.9", url="https://example.com/n.zip", sha256="ab", notes="Yenilikler var."
    )

    async def newer(url):
        return info

    monkeypatch.setattr(mw, "check_for_update", newer)
    window = FakeWindow(manifest_url="https://example.com/latest.json")
    asyncio.run(window.check_for_update())

    assert window._pending_update is info
    assert window.button_visible is True
    assert "9.9.9" in window.button_text
    assert any("9.9.9" in m for m in window.messages)
    assert any("Yenilikler var." in m for m in window.messages)


@pytest.mark.parametrize("url", ["http://example.com/x.json", "ftp://example.com/x.json"])
def test_manifest_must_be_https(url):
    """Whatever the manifest points at gets unpacked over the installation,
    so it can't be fetched over a channel that allows tampering."""
    from neo.core.updater import check_for_update as real_check

    with pytest.raises(UpdateError):
        asyncio.run(real_check(url))
