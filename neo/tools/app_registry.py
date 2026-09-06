from __future__ import annotations

import shutil
import winreg
from pathlib import Path

_ALIASES: dict[str, list[str]] = {
    "chrome": ["google chrome", "chrome"],
    "krom": ["google chrome", "chrome"],
    "vs code": ["visual studio code"],
    "vscode": ["visual studio code"],
    "code": ["visual studio code"],
    "word": ["microsoft word", "word"],
    "excel": ["microsoft excel", "excel"],
    "powerpoint": ["microsoft powerpoint", "powerpoint"],
    "explorer": ["file explorer"],
    "dosya gezgini": ["file explorer"],
    "hesap makinesi": ["calculator"],
    "not defteri": ["notepad"],
}

_START_MENU_DIRS = [
    Path(r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs"),
    Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs",
]


def _search_app_paths_registry(name: str) -> str | None:
    candidates = [name, f"{name}.exe"]
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for candidate in candidates:
            key_path = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{candidate}"
            try:
                with winreg.OpenKey(hive, key_path) as key:
                    value, _ = winreg.QueryValueEx(key, None)
                    if value:
                        return value
            except OSError:
                continue
    return None


def _search_start_menu(name: str) -> Path | None:
    terms = {name.lower()}
    terms.update(alias.lower() for alias in _ALIASES.get(name.lower(), []))

    best: Path | None = None
    for base in _START_MENU_DIRS:
        if not base.exists():
            continue
        for shortcut in base.rglob("*.lnk"):
            stem = shortcut.stem.lower()
            if stem in terms:
                return shortcut
            if best is None and any(term in stem for term in terms):
                best = shortcut
    return best


def resolve_application(name: str) -> str | Path | None:
    """Locate an installed application by its common name.

    Tries, in order: the Windows "App Paths" registry, Start Menu shortcuts
    (matched via a small alias table for common name variants), then PATH.
    Returns None rather than guessing when nothing matches.
    """
    path = _search_app_paths_registry(name)
    if path:
        return path

    shortcut = _search_start_menu(name)
    if shortcut:
        return shortcut

    which = shutil.which(name) or shutil.which(f"{name}.exe")
    if which:
        return which

    return None
