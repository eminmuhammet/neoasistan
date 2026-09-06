from __future__ import annotations

from PySide6.QtCore import Property, QEasingCurve, QPointF, QPropertyAnimation, QSequentialAnimationGroup, Qt
from PySide6.QtGui import QColor, QPainter, QRadialGradient
from PySide6.QtWidgets import QWidget

from ..core.state import AgentState

_STATE_COLORS = {
    AgentState.IDLE: QColor("#3a4550"),
    AgentState.LISTENING: QColor("#23c9ff"),
    AgentState.PROCESSING: QColor("#f2b134"),
    AgentState.SPEAKING: QColor("#35e08a"),
    AgentState.ERROR: QColor("#ff4d4f"),
}

_STATE_PERIOD_MS = {
    AgentState.IDLE: 1600,
    AgentState.LISTENING: 900,
    AgentState.PROCESSING: 550,
    AgentState.SPEAKING: 500,
    AgentState.ERROR: 1600,
}


class StatusOrb(QWidget):
    """A small glowing, breathing orb standing in for NEO's status dot --
    color-coded by AgentState, pulsing faster while listening/processing/
    speaking so the panel reads as "alive" rather than a static icon."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(64, 64)
        self._color = _STATE_COLORS[AgentState.IDLE]
        self._pulse = 0.0

        grow = QPropertyAnimation(self, b"pulse", self)
        grow.setStartValue(0.0)
        grow.setEndValue(1.0)
        grow.setEasingCurve(QEasingCurve.Type.InOutSine)
        shrink = QPropertyAnimation(self, b"pulse", self)
        shrink.setStartValue(1.0)
        shrink.setEndValue(0.0)
        shrink.setEasingCurve(QEasingCurve.Type.InOutSine)

        self._group = QSequentialAnimationGroup(self)
        self._group.addAnimation(grow)
        self._group.addAnimation(shrink)
        self._group.setLoopCount(-1)
        self._set_period(_STATE_PERIOD_MS[AgentState.IDLE])
        # Deliberately NOT started here: while idle the orb is drawn static.
        # A permanently looping animation repaints this widget forever, which
        # is wasted CPU/GPU for an app meant to sit in the background all day.

    def _set_period(self, period_ms: int) -> None:
        half = max(period_ms // 2, 100)
        self._group.animationAt(0).setDuration(half)
        self._group.animationAt(1).setDuration(half)

    def get_pulse(self) -> float:
        return self._pulse

    def set_pulse(self, value: float) -> None:
        self._pulse = value
        self.update()

    pulse = Property(float, get_pulse, set_pulse)

    def set_state(self, state: AgentState) -> None:
        self._color = _STATE_COLORS[state]
        self._set_period(_STATE_PERIOD_MS[state])
        if state in (AgentState.IDLE, AgentState.ERROR):
            self._group.stop()
            self._pulse = 0.35  # a calm, static glow
        elif self._group.state() != QSequentialAnimationGroup.State.Running:
            self._group.start()
        self.update()

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # Nothing to animate while the window is hidden in the tray.
        self._group.stop()
        super().hideEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        size = min(self.width(), self.height())
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        core_radius = size * 0.22
        glow_radius = size * (0.30 + 0.22 * self._pulse)

        glow = QRadialGradient(center, glow_radius)
        inner = QColor(self._color)
        inner.setAlpha(140)
        outer = QColor(self._color)
        outer.setAlpha(0)
        glow.setColorAt(0.0, inner)
        glow.setColorAt(1.0, outer)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(center, glow_radius, glow_radius)

        painter.setBrush(QColor(self._color))
        painter.drawEllipse(center, core_radius, core_radius)
