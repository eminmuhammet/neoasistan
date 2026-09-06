from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_ENTRY_NAME = "NEO"


def _launch_command() -> str:
    """The command Windows should run at logon.

    Uses pythonw.exe (no console window) with `-m neo.main` from the project
    root while running from source; a frozen/packaged build points at the exe
    itself instead."""
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'

    project_root = Path(__file__).resolve().parent.parent.parent
    launcher = project_root / "run_neo.py"
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    interpreter = pythonw if pythonw.exists() else Path(sys.executable)
    # Full path to the launcher script (not `-m neo.main`): Windows starts
    # logon entries with an arbitrary working directory, and running the
    # script by path is what puts the project root on sys.path.
    return f'"{interpreter}" "{launcher}"'


def is_enabled() -> bool:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as key:
            winreg.QueryValueEx(key, _ENTRY_NAME)
            return True
    except OSError:
        return False


def enable() -> bool:
    """Registers NEO to start at logon (per-user, HKCU -- no admin rights,
    and removable again with disable())."""
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, _ENTRY_NAME, 0, winreg.REG_SZ, _launch_command())
        return True
    except OSError:
        logger.exception("Otomatik başlatma etkinleştirilemedi")
        return False


def disable() -> bool:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, _ENTRY_NAME)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        logger.exception("Otomatik başlatma kapatılamadı")
        return False
