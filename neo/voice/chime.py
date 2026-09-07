from __future__ import annotations

import asyncio
import ctypes
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

# Spoken acknowledgements, picked at random so waking NEO doesn't feel like
# tripping a sensor. A raw winsound.Beep(880) was the previous cue and it
# sounded like a BIOS error, not an assistant answering.
CUE_PHRASES = (
    "Dinliyorum efendim.",
    "Buyurun efendim.",
    "Efendim?",
)

# Synthesized once and kept on disk. Generating on every wake would put a
# network round trip between the user saying the phrase and hearing that
# they were heard -- the exact delay this cue exists to remove.
_CACHE_DIRNAME = "cues"

_cache_dir: Path | None = None
_ready: list[Path] = []


def configure(data_dir: Path) -> None:
    """Points the cue cache at the user data folder (survives updates)."""
    global _cache_dir
    _cache_dir = data_dir / _CACHE_DIRNAME


async def prepare_cues(voice: str = "tr-TR-AhmetNeural") -> None:
    """Synthesizes any missing cue audio. Safe to call on every start: it
    only does work the first time, and failing is not fatal -- the fallback
    tone still tells the user NEO woke up."""
    if _cache_dir is None:
        return
    try:
        import edge_tts
    except ImportError:
        logger.info("edge-tts yok, uyandırma sesi için basit ton kullanılacak")
        return

    _cache_dir.mkdir(parents=True, exist_ok=True)
    for index, phrase in enumerate(CUE_PHRASES):
        path = _cache_dir / f"cue_{index}.mp3"
        if path in _ready:
            # Calling this more than once (a restart within the same
            # process, or a config reload) would otherwise re-append every
            # already-ready path on each call, silently skewing
            # random.choice toward whichever cue got prepared first.
            continue
        if path.exists() and path.stat().st_size > 0:
            _ready.append(path)
            continue
        try:
            await edge_tts.Communicate(phrase, voice).save(str(path))
            _ready.append(path)
            logger.info("Uyandırma sesi hazırlandı: %r", phrase)
        except Exception:
            logger.exception("Uyandırma sesi üretilemedi: %r", phrase)
            path.unlink(missing_ok=True)


def _play_file(path: Path) -> None:
    winmm = ctypes.windll.winmm
    alias = "neo_cue"
    winmm.mciSendStringW(f'open "{path}" type mpegvideo alias {alias}', None, 0, None)
    try:
        winmm.mciSendStringW(f"play {alias} wait", None, 0, None)
    finally:
        winmm.mciSendStringW(f"close {alias}", None, 0, None)


def _fallback_tone() -> None:
    """Two soft descending notes rather than one flat square-wave beep."""
    try:
        import winsound

        winsound.Beep(784, 90)
        winsound.Beep(587, 110)
    except Exception:
        logger.exception("Uyandırma sesi çalınamadı")


async def play_activation_chime() -> None:
    """Tells the user NEO is listening without them having to look at the
    screen.

    The caller mutes the microphone around this: it is speech now, and an
    open mic would capture it and hand NEO its own greeting as the command.
    """
    if _ready:
        try:
            await asyncio.to_thread(_play_file, random.choice(_ready))
            return
        except Exception:
            logger.exception("Uyandırma sesi çalınamadı, tona düşülüyor")
    await asyncio.to_thread(_fallback_tone)
