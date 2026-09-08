from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon, QPixmap


def _logo_path() -> Path:
    """Return path to the bundled logo PNG (works both dev and PyInstaller)."""
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path(__file__).parent.parent.parent
    return base / "neo_logo.png"


def build_app_icon(size: int = 64) -> QIcon:
    """Load the NEO brand mark from the bundled PNG."""
    path = _logo_path()
    if path.exists():
        px = QPixmap(str(path))
        if not px.isNull():
            return QIcon(px.scaled(size, size))
    # Fallback: plain green square so the app never crashes
    px = QPixmap(size, size)
    px.fill(0xFF39FF7A)
    return QIcon(px)
