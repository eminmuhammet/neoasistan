from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Module-level, not per-tool-instance: every computer_control tool must see
# the same panic state regardless of which Tool object triggered it, and
# there is exactly one automation session per running NEO process.
_panic_engaged = False


def is_panic_engaged() -> bool:
    return _panic_engaged


def engage_panic() -> None:
    global _panic_engaged
    if not _panic_engaged:
        logger.warning("Otomasyon durduruldu (panik tuşu veya FAILSAFE).")
    _panic_engaged = True


def reset_panic() -> None:
    """Deliberately not exposed as a tool NEO can call on itself -- once
    panic is engaged, only a human (restarting NEO, for now) re-enables
    automation. Kept for tests and any future manual UI control."""
    global _panic_engaged
    _panic_engaged = False


def register_panic_hotkey() -> None:
    """Ctrl+Alt+Shift+Q immediately halts all mouse/keyboard automation,
    independent of pyautogui's own move-to-corner FAILSAFE. Deferred
    import: only the live app needs the global keyboard hook, so core/
    stays free of that dependency for tests and any headless use."""
    import keyboard

    keyboard.add_hotkey("ctrl+alt+shift+q", engage_panic)
