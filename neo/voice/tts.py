from __future__ import annotations

import re as _re

# Emoji and other pictographs read aloud as their Unicode names -- Edge TTS
# says "beyaz onay işareti" for a ✅ that was only ever meant to be seen, and
# SAPI does something similar. Stripping them leaves the written reply
# unchanged; only the spoken version is affected.
_EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),  # pictographs, emoticons, transport, symbols
    (0x2190, 0x21FF),  # arrows
    (0x2300, 0x23FF),  # misc technical (⏹ ⏱ …)
    (0x2460, 0x27BF),  # enclosed alphanumerics, dingbats (✅ ✨ …)
    (0x2B00, 0x2BFF),  # misc symbols and arrows
    (0xFE00, 0xFE0F),  # variation selectors
)


def strip_speech_noise(text: str) -> str:
    """Removes characters that are meaningful on screen but nonsense aloud."""
    # Matching on the Unicode "So" category would be tidier but takes out
    # symbols that *should* be spoken -- "24°C" became "24C". The explicit
    # ranges cover emoji without touching degree signs, currency or maths.
    kept = [
        char
        for char in text
        if not any(low <= ord(char) <= high for low, high in _EMOJI_RANGES)
    ]
    # Collapse the gaps left behind (" ✅ ." -> " .") so the voice doesn't
    # pause where a stripped emoji used to be.
    return _re.sub(r"[ \t]{2,}", " ", "".join(kept)).strip()

import asyncio
import ctypes
import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

_PREFERRED_VOICE_SUBSTRINGS = ("tolga",)

_SVSF_ASYNC = 1
_SVSF_PURGE_BEFORE_SPEAK = 2

_POLL_INTERVAL_SECONDS = 0.1


class TextToSpeech(Protocol):
    async def speak(self, text: str) -> None: ...
    def stop(self) -> None: ...


class TTSUnavailableError(Exception):
    """Raised when a text-to-speech engine cannot be used."""


class SapiTTS:
    """Offline text-to-speech via Windows SAPI, preferring the installed
    Turkish 'Microsoft Tolga' voice when present.

    Speaks asynchronously (SVSFlagsAsync) and polls for completion in short
    steps rather than one blocking Speak() call -- a single blocking call
    per sentence meant stop() could only take effect *between* sentences,
    so a short (single-sentence) reply couldn't be interrupted at all.
    """

    def __init__(self) -> None:
        self._stop_event = threading.Event()

    def _speak_sync(self, text: str) -> None:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        try:
            try:
                voice = win32com.client.Dispatch("SAPI.SpVoice")
            except Exception as exc:
                raise TTSUnavailableError(
                    "Windows konuşma motoruna (SAPI) erişilemedi."
                ) from exc

            for token in voice.GetVoices():
                if any(s in token.GetDescription().lower() for s in _PREFERRED_VOICE_SUBSTRINGS):
                    voice.Voice = token
                    break

            voice.Speak(text, _SVSF_ASYNC)
            while not voice.WaitUntilDone(int(_POLL_INTERVAL_SECONDS * 1000)):
                if self._stop_event.is_set():
                    voice.Speak("", _SVSF_PURGE_BEFORE_SPEAK)
                    break
        finally:
            pythoncom.CoUninitialize()

    async def speak(self, text: str) -> None:
        if not text.strip():
            return
        self._stop_event.clear()
        try:
            await asyncio.to_thread(self._speak_sync, text)
        except TTSUnavailableError:
            raise
        except Exception as exc:
            logger.exception("TTS playback failed")
            raise TTSUnavailableError("Sesli okuma sırasında bir hata oluştu.") from exc

    def stop(self) -> None:
        self._stop_event.set()


class EdgeTTS:
    """Natural-sounding, online neural TTS via Microsoft Edge's "Read Aloud"
    voices (the `edge-tts` package). No API key needed, but it is an
    unofficial API and requires internet -- use FallbackTTS to drop back to
    offline SAPI when it's unavailable."""

    def __init__(self, voice: str = "tr-TR-AhmetNeural") -> None:
        self._voice = voice
        self._stop_event = threading.Event()

    async def speak(self, text: str) -> None:
        if not text.strip():
            return
        try:
            import edge_tts
        except ImportError as exc:
            raise TTSUnavailableError("edge-tts kurulu değil.") from exc

        self._stop_event.clear()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
            temp_path = handle.name

        try:
            communicate = edge_tts.Communicate(text, self._voice)
            await communicate.save(temp_path)
            if self._stop_event.is_set():
                return
            await asyncio.to_thread(self._play, temp_path)
        except TTSUnavailableError:
            raise
        except Exception as exc:
            logger.exception("Edge TTS playback failed")
            raise TTSUnavailableError("Sesli okuma sırasında bir hata oluştu.") from exc
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _play(self, path: str) -> None:
        # Play without MCI's "wait" flag and poll status instead: a blocking
        # "play ... wait" call only ends when the OS-level stop propagates
        # across threads, which isn't always prompt/reliable -- polling our
        # own stop flag every ~100ms is simpler and guaranteed responsive.
        winmm = ctypes.windll.winmm
        alias = "neo_tts"
        winmm.mciSendStringW(f'open "{path}" type mpegvideo alias {alias}', None, 0, None)
        try:
            if self._stop_event.is_set():
                return
            winmm.mciSendStringW(f"play {alias}", None, 0, None)
            status = ctypes.create_unicode_buffer(32)
            while True:
                if self._stop_event.is_set():
                    winmm.mciSendStringW(f"stop {alias}", None, 0, None)
                    break
                winmm.mciSendStringW(f"status {alias} mode", status, len(status), None)
                if status.value.strip().lower() != "playing":
                    break
                time.sleep(_POLL_INTERVAL_SECONDS)
        finally:
            winmm.mciSendStringW(f"close {alias}", None, 0, None)

    def stop(self) -> None:
        self._stop_event.set()


class FallbackTTS:
    """Tries `primary` first; if it raises TTSUnavailableError (no internet,
    missing dependency, etc.), falls back to `secondary` so NEO can always
    speak something rather than staying silent."""

    def __init__(self, primary, secondary) -> None:
        self._primary = primary
        self._secondary = secondary

    async def speak(self, text: str) -> None:
        try:
            await self._primary.speak(text)
        except TTSUnavailableError:
            logger.warning("Birincil TTS kullanılamadı, yedek sese geçiliyor")
            await self._secondary.speak(text)

    def stop(self) -> None:
        self._primary.stop()
        self._secondary.stop()
