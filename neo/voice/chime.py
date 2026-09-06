from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


def _beep(frequency: int, duration_ms: int) -> None:
    try:
        import winsound

        winsound.Beep(frequency, duration_ms)
    except Exception:
        logger.exception("Activation chime failed")


async def play_activation_chime() -> None:
    """A short audio cue for when the wake word is detected, so the user
    knows NEO is listening without having to look at the screen."""
    await asyncio.to_thread(_beep, 880, 120)
