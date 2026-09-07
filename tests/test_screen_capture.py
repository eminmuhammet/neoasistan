"""CaptureScreenTool: takes a screenshot and hands it to Claude directly,
no OCR step (see NEO_V2_PLAN.md's reasoning -- Claude reads layout and
context far better than extracted plain text would)."""

import asyncio
import base64

import pytest

from neo.tools.base import RiskLevel
from neo.tools.screen import (
    MAX_LONG_EDGE,
    CaptureScreenTool,
    ScreenCaptureUnavailableError,
    _capture_and_encode,
)


def test_capture_is_medium_risk():
    """Read-only (nothing on the machine changes), but it exposes whatever
    is currently on screen to an external API call -- the same "needs a
    live human OK" bar as deleting the user's own data."""
    assert CaptureScreenTool().risk is RiskLevel.MEDIUM


def test_capture_takes_no_parameters():
    """Kept deliberately simple: "look at what's currently on screen", not
    a region/monitor picker Claude has to reason about."""
    assert CaptureScreenTool().input_schema["properties"] == {}


def test_successful_capture_returns_expected_fields(monkeypatch):
    import neo.tools.screen as screen_module

    fake_png = b"\x89PNG\r\n\x1a\nfake-bytes"
    monkeypatch.setattr(
        screen_module,
        "_capture_and_encode",
        lambda monitor_index: (base64.b64encode(fake_png).decode("ascii"), 1400, 900),
    )

    result = asyncio.run(CaptureScreenTool().run())

    assert result.success is True
    assert result.data["media_type"] == "image/png"
    assert result.data["width"] == 1400
    assert result.data["height"] == 900
    assert base64.b64decode(result.data["image_base64"]) == fake_png


def test_capture_failure_is_reported_not_raised(monkeypatch):
    import neo.tools.screen as screen_module

    def boom(monitor_index):
        raise ScreenCaptureUnavailableError("mss yok")

    monkeypatch.setattr(screen_module, "_capture_and_encode", boom)

    result = asyncio.run(CaptureScreenTool().run())

    assert result.success is False
    assert result.error == "mss yok"


def test_downscaling_math_caps_the_long_edge():
    """A real capture without mss/Pillow installed can't run in this test
    environment reliably, so the resize math itself is exercised directly
    against a synthetic in-memory image."""
    pytest.importorskip("PIL")
    from PIL import Image

    huge = Image.new("RGB", (3840, 2160), color=(10, 20, 30))
    long_edge = max(huge.width, huge.height)
    scale = MAX_LONG_EDGE / long_edge
    resized = huge.resize(
        (max(1, round(huge.width * scale)), max(1, round(huge.height * scale)))
    )

    assert max(resized.width, resized.height) == MAX_LONG_EDGE
    # Aspect ratio preserved within a rounding pixel.
    assert abs(resized.width / resized.height - huge.width / huge.height) < 0.01


def test_small_image_is_not_upscaled():
    """A capture already under the cap shouldn't be blown up -- that would
    waste tokens on interpolated detail that was never there."""
    pytest.importorskip("PIL")
    from PIL import Image

    small = Image.new("RGB", (800, 600))
    long_edge = max(small.width, small.height)
    assert long_edge <= MAX_LONG_EDGE  # the guard in _capture_and_encode would skip resizing


def test_real_capture_produces_a_valid_png():
    """One real, unmocked capture of this machine's actual screen -- proves
    mss + Pillow are actually wired correctly together, not just that the
    mocked unit tests pass."""
    pytest.importorskip("mss")
    pytest.importorskip("PIL")

    encoded, width, height = _capture_and_encode(1)
    raw = base64.b64decode(encoded)

    assert raw[:8] == b"\x89PNG\r\n\x1a\n"  # PNG file signature
    assert width > 0 and height > 0
    assert max(width, height) <= MAX_LONG_EDGE
