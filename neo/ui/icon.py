from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap


def build_app_icon(size: int = 64) -> QIcon:
    """A small procedurally-drawn icon (dark circle, cyan ring, "N") so the
    app doesn't need to ship/load an external image asset just for the
    window/taskbar/tray icon."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    margin = size * 0.06
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#0b0f14"))
    painter.drawEllipse(int(margin), int(margin), int(size - 2 * margin), int(size - 2 * margin))

    ring_width = size * 0.07
    pen = painter.pen()
    painter.setPen(QColor("#23c9ff"))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    pen = painter.pen()
    pen.setWidthF(ring_width)
    painter.setPen(pen)
    inset = margin + ring_width / 2
    painter.drawEllipse(
        int(inset), int(inset), int(size - 2 * inset), int(size - 2 * inset)
    )

    painter.setPen(QColor("#23c9ff"))
    font = QFont("Segoe UI", int(size * 0.42), QFont.Weight.Bold)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "N")

    painter.end()
    return QIcon(pixmap)
