from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient,
)


def _hex_path(cx: float, cy: float, r: float) -> QPainterPath:
    path = QPainterPath()
    for i in range(6):
        angle = math.radians(60 * i - 90)
        pt = QPointF(cx + r * math.cos(angle), cy + r * math.sin(angle))
        if i == 0:
            path.moveTo(pt)
        else:
            path.lineTo(pt)
    path.closeSubpath()
    return path


def _hex_pts(cx: float, cy: float, r: float) -> list[QPointF]:
    return [
        QPointF(cx + r * math.cos(math.radians(60 * i - 90)),
                cy + r * math.sin(math.radians(60 * i - 90)))
        for i in range(6)
    ]


def build_app_icon(size: int = 64) -> QIcon:
    """NEO brand mark: glowing hex + N-arrow + 'NEO' text, drawn procedurally."""
    px = QPixmap(size, size)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

    s = float(size)
    cx, cy = s / 2, s / 2
    GREEN = QColor("#39ff7a")
    BG    = QColor("#060d08")

    r_hex  = s * 0.455
    r_fill = s * 0.415

    # Soft radial glow behind hex
    rg = QRadialGradient(cx, cy, r_hex * 1.1)
    rg.setColorAt(0.0, QColor(40, 120, 60, 55))
    rg.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(rg)
    p.drawEllipse(QRectF(cx - r_hex*1.2, cy - r_hex*1.2, r_hex*2.4, r_hex*2.4))

    # Hex dark fill
    p.setBrush(BG)
    p.drawPath(_hex_path(cx, cy, r_hex))

    # Hex border — glow in 3 layers
    for width, alpha in [(s*0.09, 30), (s*0.048, 90), (s*0.022, 255)]:
        c = QColor(GREEN); c.setAlpha(alpha)
        pen = QPen(c); pen.setWidthF(width)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(_hex_path(cx, cy, r_fill))

    # Corner dots on hex vertices
    p.setPen(Qt.PenStyle.NoPen)
    for pt in _hex_pts(cx, cy, r_fill):
        p.setBrush(GREEN);  p.drawEllipse(pt, s*0.022, s*0.022)
        p.setBrush(BG);     p.drawEllipse(pt, s*0.011, s*0.011)

    # ── N + arrow — N spans upper 55% of hex interior ──────────────────────
    # The icon is square; put N in the top portion and NEO text in bottom 30%
    n_left  = cx - s * 0.155
    n_right = cx + s * 0.135
    n_top   = cy - s * 0.240   # top of N strokes
    n_bot   = cy + s * 0.035   # bottom of N strokes

    arrow_tip_y   = n_top - s * 0.055
    arrow_spread  = s * 0.065
    arrow_base_y  = n_top + s * 0.060

    stroke_w = s * 0.062
    for width, alpha in [(stroke_w*2.0, 25), (stroke_w*1.4, 70), (stroke_w, 255)]:
        c = QColor(GREEN); c.setAlpha(alpha)
        pen = QPen(c); pen.setWidthF(width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.drawLine(QPointF(n_left,  n_bot), QPointF(n_left,  n_top))
        p.drawLine(QPointF(n_left,  n_top), QPointF(n_right, n_bot))
        p.drawLine(QPointF(n_right, n_bot), QPointF(n_right, arrow_base_y))

    # Arrowhead (filled triangle)
    arrow_path = QPainterPath()
    arrow_path.moveTo(QPointF(n_right, arrow_tip_y))
    arrow_path.lineTo(QPointF(n_right - arrow_spread, arrow_base_y))
    arrow_path.lineTo(QPointF(n_right + arrow_spread, arrow_base_y))
    arrow_path.closeSubpath()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(GREEN)
    p.drawPath(arrow_path)

    # ── "NEO" text in bottom third of hex ──────────────────────────────────
    if size >= 32:
        font_size = max(6, int(s * 0.130))
        font = QFont("Arial", font_size, QFont.Weight.Bold)
        p.setFont(font)
        text_rect = QRectF(0, cy + s * 0.090, s, s * 0.170)
        # Glow
        for alpha in [40, 140, 255]:
            c = QColor(GREEN); c.setAlpha(alpha)
            p.setPen(QPen(c))
            p.drawText(text_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, "NEO")

    p.end()
    return QIcon(px)
