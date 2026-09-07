import asyncio

import pytest

from neo.core import automation_panic
from neo.tools import computer_control as cc
from neo.tools.base import RiskLevel


class FailSafeException(Exception):
    pass


class FakePyAutoGui:
    def __init__(self, raise_failsafe: bool = False) -> None:
        self.FailSafeException = FailSafeException
        self.FAILSAFE = False
        self.calls: list[tuple] = []
        self._raise_failsafe = raise_failsafe

    def _maybe_raise(self):
        if self._raise_failsafe:
            raise FailSafeException("corner")

    def click(self, x, y, clicks=1, button="left"):
        self.calls.append(("click", x, y, clicks, button))
        self._maybe_raise()

    def moveTo(self, x, y, duration=0.0):
        self.calls.append(("moveTo", x, y, duration))
        self._maybe_raise()

    def dragTo(self, x, y, duration=0.0):
        self.calls.append(("dragTo", x, y, duration))
        self._maybe_raise()

    def write(self, text, interval=0.0):
        self.calls.append(("write", text, interval))
        self._maybe_raise()

    def hotkey(self, *keys):
        self.calls.append(("hotkey", keys))
        self._maybe_raise()


def _patch(monkeypatch, fake):
    monkeypatch.setattr(cc, "_pyautogui", lambda: fake)
    monkeypatch.setattr(cc, "_verify_with_screenshot", _no_screenshot)


async def _no_screenshot():
    return {}


def teardown_function(_fn):
    automation_panic.reset_panic()


def test_click_calls_pyautogui_and_reports_success(monkeypatch):
    fake = FakePyAutoGui()
    _patch(monkeypatch, fake)

    result = asyncio.run(cc.ClickTool().run(x=100, y=200))

    assert result.success is True
    assert fake.calls == [("click", 100, 200, 1, "left")]


def test_double_click_passes_two_clicks(monkeypatch):
    fake = FakePyAutoGui()
    _patch(monkeypatch, fake)

    asyncio.run(cc.ClickTool().run(x=1, y=2, double=True))

    assert fake.calls == [("click", 1, 2, 2, "left")]


def test_failsafe_exception_engages_panic_and_reports_error(monkeypatch):
    fake = FakePyAutoGui(raise_failsafe=True)
    _patch(monkeypatch, fake)

    result = asyncio.run(cc.ClickTool().run(x=0, y=0))

    assert result.success is False
    assert automation_panic.is_panic_engaged() is True


def test_when_panic_already_engaged_tool_refuses_without_calling_pyautogui(monkeypatch):
    fake = FakePyAutoGui()
    _patch(monkeypatch, fake)
    automation_panic.engage_panic()

    result = asyncio.run(cc.ClickTool().run(x=1, y=1))

    assert result.success is False
    assert fake.calls == []


def test_move_mouse(monkeypatch):
    fake = FakePyAutoGui()
    _patch(monkeypatch, fake)

    result = asyncio.run(cc.MoveMouseTool().run(x=5, y=6))

    assert result.success is True
    assert fake.calls == [("moveTo", 5, 6, 0.2)]


def test_drag_moves_then_drags(monkeypatch):
    fake = FakePyAutoGui()
    _patch(monkeypatch, fake)

    result = asyncio.run(cc.DragTool().run(start_x=1, start_y=2, end_x=3, end_y=4))

    assert result.success is True
    assert fake.calls == [("moveTo", 1, 2, 0.0), ("dragTo", 3, 4, 0.3)]


def test_type_text_writes_given_text(monkeypatch):
    fake = FakePyAutoGui()
    _patch(monkeypatch, fake)

    result = asyncio.run(cc.TypeTextTool().run(text="merhaba"))

    assert result.success is True
    assert fake.calls == [("write", "merhaba", 0.02)]


def test_press_keys_sends_hotkey(monkeypatch):
    fake = FakePyAutoGui()
    _patch(monkeypatch, fake)

    result = asyncio.run(cc.PressKeysTool().run(keys=["ctrl", "c"]))

    assert result.success is True
    assert fake.calls == [("hotkey", ("ctrl", "c"))]


def test_press_keys_rejects_empty_list(monkeypatch):
    fake = FakePyAutoGui()
    _patch(monkeypatch, fake)

    result = asyncio.run(cc.PressKeysTool().run(keys=[]))

    assert result.success is False
    assert fake.calls == []


@pytest.mark.parametrize(
    "tool_cls",
    [cc.ClickTool, cc.MoveMouseTool, cc.DragTool, cc.TypeTextTool, cc.PressKeysTool],
)
def test_all_computer_control_tools_are_high_risk(tool_cls):
    assert tool_cls.risk == RiskLevel.HIGH
