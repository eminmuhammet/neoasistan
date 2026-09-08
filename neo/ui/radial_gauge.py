from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

_START_ANGLE = 90 * 16  # Qt arc angles are in 1/16ths of a degree, 0 = 3 o'clock
_SPAN_ANGLE = -300 * 16  # a 300-degree sweep, leaving a gap at the bottom


class RadialGauge(QWidget):
    """A HUD-style circular arc gauge (colored progress arc + centered
    percentage + label) used for CPU/RAM/GPU instead of a flat progress bar."""

    def __init__(self, label: str, color: str = "#39ff7a", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = label
        self._color = QColor(color)
        self._value = 0.0
        self._extra_text = ""
        self.setFixedSize(96, 112)

    def set_value(self, value: float) -> None:
        self._value = max(0.0, min(100.0, value))
        self.update()

    def set_extra_text(self, text: str) -> None:
        self._extra_text = text
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        diameter = 80
        circle_rect = QRectF((self.width() - diameter) / 2, 4, diameter, diameter)
        arc_rect = circle_rect.adjusted(5, 5, -5, -5)

        track_pen = QPen(QColor("#1c2733"))
        track_pen.setWidthF(7)
        track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.drawArc(arc_rect, _START_ANGLE, _SPAN_ANGLE)

        value_pen = QPen(self._color)
        value_pen.setWidthF(7)
        value_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(value_pen)
        painter.drawArc(arc_rect, _START_ANGLE, int(_SPAN_ANGLE * (self._value / 100.0)))

        painter.setPen(QColor("#d7e3ea"))
        painter.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        painter.drawText(circle_rect, Qt.AlignmentFlag.AlignCenter, f"{round(self._value)}%")

        painter.setPen(QColor("#7c8b98"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        label_text = f"{self._label}  {self._extra_text}".strip()
        label_rect = QRectF(0, circle_rect.bottom() + 2, self.width(), 16)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label_text)
