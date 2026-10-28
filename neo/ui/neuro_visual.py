from __future__ import annotations

from functools import lru_cache

import numpy as np
import threading

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

from ..core.state import AgentState

# ── State parameters ──────────────────────────────────────────────────────
# NEO's logo is green (#35e08a -> hue 0.416), and this light is the logo at
# this size, so every state stays in that family. States separate by shade
# alone -- the old blue LISTENING made NEO look like a different application
# mid-sentence. Error keeps red: it has to read as wrong at a glance, and
# that convention outranks matching the palette.
_LOGO_HUE = 0.388   # the logo's #39ff7a
_STATE_HUE: dict[AgentState, float] = {
    AgentState.IDLE:       _LOGO_HUE,
    AgentState.LISTENING:  0.448,  # cooler green, still unmistakably green
    AgentState.PROCESSING: 0.360,  # warmer lime
    AgentState.SPEAKING:   _LOGO_HUE,
    AgentState.ERROR:      0.000,  # red
}
_STATE_SAT: dict[AgentState, float] = {
    AgentState.IDLE:       0.76,   # the logo's own saturation
    AgentState.LISTENING:  0.68,
    AgentState.PROCESSING: 0.82,
    AgentState.SPEAKING:   0.72,
    AgentState.ERROR:      0.90,
}

# Rendered 1:1 with the widget so nothing is ever upscaled. Capped mainly
# to bound memory on very large/HiDPI displays.
_MIN_RENDER_SIZE = 240
_MAX_RENDER_SIZE = 1200

# Upper end of the glow range the colour table covers. Above 1.0 so the
# brightest core still clips channel by channel and burns toward white.
_GLOW_RANGE = 2.0

# How far the light reaches, as a fraction of the widget's half-width.
_CORE_RADIUS = 0.22

# Background colour — exact match for app theme #060a08
_BG = np.array([6 / 255, 10 / 255, 8 / 255], dtype=np.float32)


@lru_cache(maxsize=4)
def _distance_grid(S: int) -> np.ndarray:
    """Normalized distance from the centre (0 at centre, 1 at the widget's
    edge), depending only on S -- cached because recomputing an 875x875
    grid (np.mgrid + sqrt) from scratch cost ~37 ms on its own, and this
    value never changes unless the widget itself is resized. lru_cache is
    thread-safe for this read-mostly pattern, so it stays callable from
    _render_frame's background thread without a lock.
    """
    cx = cy = (S - 1) * 0.5
    yy, xx = np.mgrid[0:S, 0:S]
    return np.sqrt((xx.astype(np.float32) - cx) ** 2
                    + (yy.astype(np.float32) - cy) ** 2) / (S * 0.5)


@lru_cache(maxsize=4)
def _glow_shape(S: int) -> np.ndarray:
    """The light's fixed intensity shape (tight core + soft halo), cached
    per render size for the same reason as _distance_grid -- nothing about
    its shape depends on state, so it is computed once rather than on
    every render."""
    dist = _distance_grid(S)
    core = np.exp(-0.5 * (dist / (_CORE_RADIUS * 0.55)) ** 2)
    halo = np.exp(-0.5 * (dist / (_CORE_RADIUS * 1.8)) ** 2)
    return core * 1.4 + halo * 0.55


def _hsv_to_rgb(h: float, s: float, v: np.ndarray) -> np.ndarray:
    """Vectorised HSV → RGB. h,s scalar; v (N,) array. Returns (N, 3) float32."""
    h6   = (h * 6.0) % 6.0
    seg  = int(h6)
    f    = h6 - seg
    c    = v * s
    x    = c * (1.0 - abs(h6 % 2.0 - 1.0))
    m    = v - c
    z    = np.zeros_like(v)
    mapping = [(c, x, z), (x, c, z), (z, c, x),
               (z, x, c), (x, z, c), (c, z, x)]
    r0, g0, b0 = mapping[seg]
    return np.stack([r0 + m, g0 + m, b0 + m], axis=1)


def _glow_ramp(colour: np.ndarray) -> np.ndarray:
    """256 colours from background to white-hot, as (256, 3) floats 0-255.

    A glow that scales one colour by brightness is a straight line through
    colour space, and it looks flat. Real emission shifts hue as it gets
    hotter: deep and saturated where it is dim, the material's own colour
    in the middle, desaturating to white at the core. Three stops are
    enough to get that, and since this only ever fills a lookup table it
    costs nothing per frame.
    """
    deep = colour * 0.55                       # dim: darker, still saturated
    mid  = colour                              # the logo's own green
    hot  = colour + (1.0 - colour) * 0.50      # core: desaturated toward white

    t = np.linspace(0.0, 1.0, 256, dtype=np.float32)[:, None]
    # Two linear segments, joined at the middle stop.
    lower = deep[None, :] + (mid - deep)[None, :] * (t / 0.55)
    upper = mid[None, :] + (hot - mid)[None, :] * ((t - 0.55) / 0.45)
    ramp = np.where(t < 0.55, lower, upper)

    # Brightness still rises across the whole range; the stops above only
    # decide the hue at each level.
    ramp = ramp * t

    return np.clip(_BG[None, :] * 255.0 + ramp * (_GLOW_RANGE * 255.0), 0.0, 255.0)


class NeuroVisual(QWidget):
    """NEO's central indicator: a plain, static coloured light -- a
    deliberate, much simpler step back from the rotating particle sphere
    (and then the breathing/voice-reactive glow) this replaces while that
    design is reworked. It only redraws when AgentState actually changes,
    not on a timer, so an idle conversation costs no CPU/GPU at all."""

    _COMPACT_SIZE = 280
    _FOCUS_SIZE   = 700

    _frame_ready = Signal(QPixmap)  # emitted from render thread → main thread

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._focus = False
        self.setFixedSize(self._COMPACT_SIZE, self._COMPACT_SIZE)

        self._state = AgentState.IDLE

        # Widget must be visually transparent so no Qt background colour
        # paints on top of (or slightly differs from) our rendered _BG.
        self.setStyleSheet("background: transparent; border: none;")

        self._cached_pixmap: QPixmap | None = None
        self._rendering = False
        self._frame_ready.connect(self._on_frame_ready)

        self._render_async()

    # ── Public API ────────────────────────────────────────────────────────
    def set_focus(self, focus: bool) -> None:
        if focus == self._focus:
            return
        self._focus = focus
        size = self._FOCUS_SIZE if focus else self._COMPACT_SIZE
        self.setFixedSize(size, size)
        self._render_async()

    def set_state(self, state: AgentState) -> None:
        if state == self._state:
            return
        self._state = state
        self._render_async()

    def set_audio_level(self, level: float) -> None:
        """Kept so callers (push-to-talk, wake-word) don't need to know
        this is a static light now -- it simply has nothing to react with
        at the moment. See the class docstring."""

    def _on_frame_ready(self, pixmap: QPixmap) -> None:
        self._cached_pixmap = pixmap
        self._rendering = False
        self.update()

    def _render_async(self) -> None:
        """Renders one frame in the background for the current state/size.
        Not on any timer -- called only when something actually changed."""
        if self._rendering:
            return
        self._rendering = True
        snap = (self._state, _STATE_HUE[self._state], _STATE_SAT[self._state],
                self._render_size())
        t = threading.Thread(target=self._render_thread, args=(snap,), daemon=True)
        t.start()

    def _render_thread(self, snap: tuple) -> None:
        try:
            pixmap = self._render_frame(snap)
            self._frame_ready.emit(pixmap)
        except RuntimeError:
            # Widget was destroyed while this frame was still rendering --
            # normal on shutdown, and the frame has nowhere to go anyway.
            pass

    # ── Rendering ─────────────────────────────────────────────────────────
    def _render_size(self) -> int:
        """Pixel size of the frame to render: the widget's own size, so the
        result is blitted 1:1 instead of being stretched."""
        side = max(self.width(), self.height()) * self.devicePixelRatioF()
        return int(max(_MIN_RENDER_SIZE, min(_MAX_RENDER_SIZE, side)))

    def _render_frame(self, snap: tuple) -> QPixmap:
        """Pure function — runs on a background thread, no self mutation."""
        state, hue, sat, S = snap

        glow = _glow_shape(S)
        colour = _hsv_to_rgb(hue % 1.0, sat, np.array([1.0], np.float32))[0]

        # Colourize through a 256-entry lookup table rather than a full
        # (S, S, 3) float broadcast -- the glow is one channel, so the
        # whole colour mapping fits in a table and the frame becomes a
        # single gather.
        lut = np.empty((256, 4), dtype=np.uint8)
        lut[:, :3] = _glow_ramp(colour)
        lut[:, 3] = 255

        index = np.clip(glow * (255.0 / _GLOW_RANGE), 0.0, 255.0).astype(np.uint8)
        rgba = np.ascontiguousarray(lut[index])
        img = QImage(rgba.data, S, S, S * 4, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(img.copy())

    def paintEvent(self, event) -> None:  # noqa: N802
        W, H = self.width(), self.height()
        pixmap = self._cached_pixmap
        if pixmap is None:
            return  # first frame not ready; parent background shows through

        painter = QPainter(self)
        if pixmap.width() == W and pixmap.height() == H:
            # The common case: frames are rendered at the widget's own
            # size, so this is a direct blit. Calling scaled() here would
            # still allocate and copy the whole frame on every repaint to
            # produce an identical image.
            painter.drawPixmap(0, 0, pixmap)
        else:
            # Only while a resize is in flight and the cached frame is
            # still the old size.
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            scaled = pixmap.scaled(
                W, H,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(
                (W - scaled.width()) // 2, (H - scaled.height()) // 2, scaled
            )
        painter.end()
