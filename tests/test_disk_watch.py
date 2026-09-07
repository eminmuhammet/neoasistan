import asyncio

from neo.core.disk_watch import DiskSpaceWatcher
from neo.tools.base import ToolResult


class FakeDiskTool:
    def __init__(self, result: ToolResult) -> None:
        self._result = result

    async def run(self, **kwargs):
        return self._result


def _ok(drives):
    return ToolResult(success=True, data={"drives": drives})


def _drive(name, percent, free_gb=10.0):
    return {"drive": name, "percent_used": percent, "free_gb": free_gb}


def _watch(monkeypatch, result, threshold=90.0):
    alerts: list[str] = []
    watcher = DiskSpaceWatcher(on_alert=lambda text: alerts.append(text), threshold_percent=threshold)

    async def fake_run(self, **kwargs):
        return result

    monkeypatch.setattr("neo.tools.system_info.GetDiskUsageTool.run", fake_run)
    return watcher, alerts


def test_alert_fires_once_when_crossing_threshold(monkeypatch):
    watcher, alerts = _watch(monkeypatch, _ok([_drive("C:", 95.0)]))

    asyncio.run(watcher.check())
    asyncio.run(watcher.check())

    assert len(alerts) == 1
    assert "C:" in alerts[0]


def test_no_alert_when_under_threshold(monkeypatch):
    watcher, alerts = _watch(monkeypatch, _ok([_drive("C:", 50.0)]))

    asyncio.run(watcher.check())

    assert alerts == []


def test_alert_refires_after_dropping_and_crossing_again(monkeypatch):
    watcher = DiskSpaceWatcher(on_alert=lambda text: alerts.append(text), threshold_percent=90.0)
    alerts = []

    async def over(self, **kwargs):
        return _ok([_drive("C:", 95.0)])

    async def under(self, **kwargs):
        return _ok([_drive("C:", 50.0)])

    monkeypatch.setattr("neo.tools.system_info.GetDiskUsageTool.run", over)
    asyncio.run(watcher.check())
    monkeypatch.setattr("neo.tools.system_info.GetDiskUsageTool.run", under)
    asyncio.run(watcher.check())
    monkeypatch.setattr("neo.tools.system_info.GetDiskUsageTool.run", over)
    asyncio.run(watcher.check())

    assert len(alerts) == 2


def test_multiple_drives_tracked_independently(monkeypatch):
    watcher, alerts = _watch(monkeypatch, _ok([_drive("C:", 95.0), _drive("D:", 20.0)]))

    asyncio.run(watcher.check())

    assert len(alerts) == 1
    assert "C:" in alerts[0]


def test_failed_read_never_fires_an_alert(monkeypatch):
    watcher, alerts = _watch(monkeypatch, ToolResult(success=False, error="okunamadı"))

    asyncio.run(watcher.check())

    assert alerts == []
