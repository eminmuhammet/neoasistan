from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def _hexagon_path(center: QPointF, radius: float) -> QPainterPath:
    path = QPainterPath()
    for i in range(6):
        angle = math.radians(60 * i - 90)
        point = QPointF(
            center.x() + radius * math.cos(angle),
            center.y() + radius * math.sin(angle),
        )
        if i == 0:
            path.moveTo(point)
        else:
            path.lineTo(point)
    path.closeSubpath()
    return path


def build_app_icon(size: int = 64) -> QIcon:
    """Matches NEO's brand mark: a glowing green hexagon frame around an "N"
    whose right stroke doubles as an upward arrow -- drawn procedurally so
    the app ships no external image asset for the window/taskbar/tray icon.
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    center = QPointF(size / 2.0, size / 2.0)
    hex_radius = size * 0.46
    green = QColor("#35e08a")

    # Dark hexagon fill behind everything.
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#06120c"))
    painter.drawPath(_hexagon_path(center, hex_radius))

    # Glowing hexagon outline.
    pen = QPen(green)
    pen.setWidthF(size * 0.045)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(_hexagon_path(center, hex_radius * 0.92))

    # "N" strokes, sized to sit inside the hexagon with room for the arrowhead.
    n_pen = QPen(green)
    n_pen.setWidthF(size * 0.09)
    n_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    n_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(n_pen)

    left_x = size * 0.35
    right_x = size * 0.62
    top_y = size * 0.62
    bottom_y = size * 0.34

    painter.drawLine(QPointF(left_x, size * 0.66), QPointF(left_x, size * 0.30))
    painter.drawLine(QPointF(left_x, size * 0.30), QPointF(right_x, size * 0.62))
    painter.drawLine(QPointF(right_x, size * 0.62), QPointF(right_x, size * 0.24))

    # Arrowhead capping the right stroke, reading as "N" rising into an
    # upward arrow -- the brand's growth motif.
    arrow_tip = QPointF(right_x, size * 0.16)
    arrow_path = QPainterPath()
    spread = size * 0.09
    arrow_path.moveTo(arrow_tip)
    arrow_path.lineTo(right_x - spread, size * 0.16 + spread)
    arrow_path.lineTo(right_x + spread, size * 0.16 + spread)
    arrow_path.closeSubpath()
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(green)
    painter.drawPath(arrow_path)

    painter.end()
    return QIcon(pixmap)
