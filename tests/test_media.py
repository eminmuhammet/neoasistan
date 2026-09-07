import asyncio

import pytest

from neo.tools import media
from neo.tools.base import RiskLevel


class FakeEndpoint:
    def __init__(self, level: float = 0.5, muted: bool = False) -> None:
        self._level = level
        self._muted = muted
        self.calls: list[tuple] = []

    def GetMasterVolumeLevelScalar(self):
        return self._level

    def GetMute(self):
        return int(self._muted)

    def SetMasterVolumeLevelScalar(self, level, _guid):
        self.calls.append(("set_volume", level))
        self._level = level

    def SetMute(self, muted, _guid):
        self.calls.append(("set_mute", muted))
        self._muted = muted


def _patch_endpoint(monkeypatch, endpoint):
    monkeypatch.setattr(media, "_get_endpoint_volume", lambda: endpoint)


def test_get_volume_reports_percent_and_mute_state(monkeypatch):
    _patch_endpoint(monkeypatch, FakeEndpoint(level=0.42, muted=True))

    result = asyncio.run(media.GetVolumeTool().run())

    assert result.success is True
    assert result.data["volume_percent"] == 42
    assert result.data["muted"] is True


def test_get_volume_reports_failure_cleanly(monkeypatch):
    def _raise():
        raise RuntimeError("no audio device")

    monkeypatch.setattr(media, "_get_endpoint_volume", _raise)

    result = asyncio.run(media.GetVolumeTool().run())

    assert result.success is False


def test_set_volume_clamps_to_valid_range(monkeypatch):
    endpoint = FakeEndpoint()
    _patch_endpoint(monkeypatch, endpoint)

    result = asyncio.run(media.SetVolumeTool().run(percent=150))

    assert result.success is True
    assert result.data["volume_percent"] == 100
    assert endpoint.calls == [("set_volume", 1.0)]


def test_set_volume_clamps_negative_to_zero(monkeypatch):
    endpoint = FakeEndpoint()
    _patch_endpoint(monkeypatch, endpoint)

    result = asyncio.run(media.SetVolumeTool().run(percent=-10))

    assert result.data["volume_percent"] == 0
    assert endpoint.calls == [("set_volume", 0.0)]


def test_set_mute_calls_endpoint(monkeypatch):
    endpoint = FakeEndpoint()
    _patch_endpoint(monkeypatch, endpoint)

    result = asyncio.run(media.SetMuteTool().run(muted=True))

    assert result.success is True
    assert endpoint.calls == [("set_mute", True)]


def test_media_control_sends_correct_key(monkeypatch):
    pressed = []
    monkeypatch.setattr(media, "_press_media_key", lambda vk: pressed.append(vk))

    result = asyncio.run(media.MediaControlTool().run(action="next"))

    assert result.success is True
    assert pressed == [media._VK_MEDIA_NEXT_TRACK]


def test_media_control_rejects_unknown_action():
    result = asyncio.run(media.MediaControlTool().run(action="rewind"))

    assert result.success is False


@pytest.mark.parametrize(
    "tool_cls",
    [media.GetVolumeTool, media.SetVolumeTool, media.SetMuteTool, media.MediaControlTool],
)
def test_media_tools_are_low_risk(tool_cls):
    assert tool_cls.risk == RiskLevel.LOW
