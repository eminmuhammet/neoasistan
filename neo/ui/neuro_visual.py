from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QRadialGradient
from PySide6.QtWidgets import QWidget

from ..core.state import AgentState

# ── State colours ──────────────────────────────────────────────────────────
# NEO's logo is green (#35e08a -> hue 0.388), and this lamp is the logo at
# this size, so every state stays in that family. States separate by shade
# and blink speed -- the old blue LISTENING made NEO look like a different
# application mid-sentence. Error keeps red: it has to read as wrong at a
# glance, and that convention outranks matching the palette.
_LOGO_HUE = 0.388
_STATE_COLOR: dict[AgentState, QColor] = {
    AgentState.IDLE:       QColor.fromHsvF(_LOGO_HUE, 0.76, 1.0),
    AgentState.LISTENING:  QColor.fromHsvF(0.448, 0.68, 1.0),  # cooler green
    AgentState.PROCESSING: QColor.fromHsvF(0.360, 0.82, 1.0),  # warmer lime
    AgentState.SPEAKING:   QColor.fromHsvF(_LOGO_HUE, 0.72, 1.0),
    AgentState.ERROR:      QColor.fromHsvF(0.0, 0.90, 1.0),
}
# Radians added to the blink phase per millisecond -- how fast the lamp
# pulses in each state.
_STATE_BLINK_SPEED: dict[AgentState, float] = {
    AgentState.IDLE:       0.0018,
    AgentState.LISTENING:  0.0055,
    AgentState.PROCESSING: 0.0090,
    AgentState.SPEAKING:   0.0070,
    AgentState.ERROR:      0.0018,
}
# How dim the lamp gets at the bottom of each blink, 0..1.
_MIN_BRIGHTNESS = 0.45


class NeuroVisual(QWidget):
    """NEO's central indicator: a plain blinking lamp, coloured by
    AgentState -- drawn natively with QPainter/QRadialGradient, not
    computed pixel-by-pixel, so there is no background render thread and
    nothing to cost CPU/GPU beyond a cheap periodic repaint of one gradient
    circle."""

    _COMPACT_SIZE = 280
    _FOCUS_SIZE   = 700

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._focus = False
        self.setFixedSize(self._COMPACT_SIZE, self._COMPACT_SIZE)

        self._state = AgentState.IDLE
        self._phase = 0.0

        # Transparent so the app's own background shows through around the
        # lamp instead of a Qt-painted square behind it.
        self.setStyleSheet("background: transparent; border: none;")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(50)  # 20 fps -- plenty smooth for a soft pulse
        self._timer.start()

    # ── Public API ────────────────────────────────────────────────────────
    def set_focus(self, focus: bool) -> None:
        if focus == self._focus:
            return
        self._focus = focus
        size = self._FOCUS_SIZE if focus else self._COMPACT_SIZE
        self.setFixedSize(size, size)

    def set_state(self, state: AgentState) -> None:
        self._state = state

    def set_audio_level(self, level: float) -> None:
        """Kept so callers (push-to-talk, wake-word) don't need to know
        this is a plain lamp now -- it simply has nothing to react with at
        the moment."""

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    # ── Animation ─────────────────────────────────────────────────────────
    def _tick(self) -> None:
        self._phase += self._timer.interval() * _STATE_BLINK_SPEED[self._state]
        self.update()

    # ── Rendering ─────────────────────────────────────────────────────────
    def paintEvent(self, event) -> None:  # noqa: N802
        W, H = self.width(), self.height()
        cx, cy = W / 2.0, H / 2.0
        radius = min(W, H) * 0.14

        pulse = 0.5 + 0.5 * math.sin(self._phase)
        brightness = _MIN_BRIGHTNESS + (1.0 - _MIN_BRIGHTNESS) * pulse

        colour = _STATE_COLOR[self._state]

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        # Soft halo, fading out from the lamp.
        halo_radius = radius * 3.2
        gradient = QRadialGradient(QPointF(cx, cy), halo_radius)
        inner = QColor(colour)
        inner.setAlphaF(brightness)
        mid = QColor(colour)
        mid.setAlphaF(brightness * 0.35)
        outer = QColor(colour)
        outer.setAlphaF(0.0)
        gradient.setColorAt(0.0, inner)
        gradient.setColorAt(0.35, mid)
        gradient.setColorAt(1.0, outer)
        painter.setBrush(gradient)
        painter.drawEllipse(QPointF(cx, cy), halo_radius, halo_radius)

        # Bright core.
        core = QColor(colour)
        core.setAlphaF(min(1.0, brightness * 1.3))
        painter.setBrush(core)
        painter.drawEllipse(QPointF(cx, cy), radius * 0.6, radius * 0.6)

        painter.end()
