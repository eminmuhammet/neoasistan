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

_PREFERRED_VOICE_SUBSTRINGS = ("tolga", "turkish", "türkçe", "turkce")
# SAPI reports a voice's language as a hex LCID; 0x41F is Turkish. Matching
# on this as well as on the name catches Turkish voices whose description is
# localized or named something other than "Tolga".
_TURKISH_LCID = "41f"

# Retries before giving up on the online voice. The offline fallback is only
# acceptable when a Turkish SAPI voice is installed -- otherwise it reads
# Turkish with an English voice, which the user cannot understand.
EDGE_ATTEMPTS = 3
EDGE_RETRY_SECONDS = 0.8

# Roughly a sentence. Short enough that the first one is synthesized almost
# instantly, long enough that the voice doesn't sound chopped between them.
_MIN_CHUNK_CHARS = 90
_SENTENCE_END = _re.compile(r"(?<=[.!?…:])\s+")


def split_for_speech(text: str) -> list[str]:
    """Splits a reply into chunks that can be spoken as they are produced.

    Short replies stay whole -- splitting "Saat 14.30." would only add a
    seam. Longer ones are broken on sentence boundaries so the first words
    can start playing while the rest is still being synthesized.
    """
    stripped = text.strip()
    if len(stripped) <= _MIN_CHUNK_CHARS:
        return [stripped] if stripped else []

    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(stripped):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) >= _MIN_CHUNK_CHARS:
            chunks.append(candidate)
            current = ""
        else:
            current = candidate
    if current:
        # A short tail is appended to the previous chunk rather than spoken
        # on its own, which would land as an odd clipped fragment.
        if chunks and len(current) < 40:
            chunks[-1] = f"{chunks[-1]} {current}"
        else:
            chunks.append(current)
    return chunks

_SVSF_ASYNC = 1
_SVSF_PURGE_BEFORE_SPEAK = 2

_POLL_INTERVAL_SECONDS = 0.1


class TextToSpeech(Protocol):
    async def speak(self, text: str) -> None: ...
    def stop(self) -> None: ...


class TTSUnavailableError(Exception):
    """Raised when a text-to-speech engine cannot be used."""


def _select_turkish_voice(voice) -> bool:
    """Points a SAPI voice object at an installed Turkish voice.

    Returns False when none is installed, so the caller can report that
    rather than silently speaking Turkish in whatever the system default is.
    """
    for token in voice.GetVoices():
        try:
            description = token.GetDescription().lower()
        except Exception:
            continue
        if any(s in description for s in _PREFERRED_VOICE_SUBSTRINGS):
            voice.Voice = token
            return True
        try:
            if _TURKISH_LCID in str(token.GetAttribute("Language")).lower():
                voice.Voice = token
                return True
        except Exception:
            # Not every token exposes every attribute; name matching above
            # is the primary path.
            continue
    return False


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

            if not _select_turkish_voice(voice):
                # Windows falls back to its default voice, which on this
                # machine is US English -- it will pronounce Turkish text as
                # if it were English, and the result is not understandable.
                # Say so once rather than letting it sound like a bug.
                logger.warning(
                    "Türkçe SAPI sesi bulunamadı; Windows varsayılan sesi "
                    "Türkçeyi doğru okuyamaz. Ayarlar > Saat ve Dil > Konuşma "
                    "bölümünden Türkçe ses eklenebilir."
                )

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
            import edge_tts  # noqa: F401
        except ImportError as exc:
            raise TTSUnavailableError("edge-tts kurulu değil.") from exc

        self._stop_event.clear()

        # Synthesized and played one chunk at a time. Doing the whole reply
        # first meant nothing was audible until the entire text had been
        # turned into audio over the network -- on a long answer that is
        # several seconds of silence after NEO has already decided what to
        # say. Speaking the first sentence while the rest is still being
        # made removes that wait almost entirely.
        for chunk in split_for_speech(text):
            if self._stop_event.is_set():
                return
            await self._speak_chunk(chunk)

    async def _speak_chunk(self, text: str) -> None:
        import edge_tts

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
            temp_path = handle.name

        try:
            # A dropped websocket used to fall straight through to the
            # offline fallback, which on a machine with no Turkish SAPI voice
            # means an English voice reading Turkish aloud -- unintelligible.
            # A brief retry costs a second and covers the transient network
            # blips that caused it (see logs/neo.log 22:43:57).
            last_error: Exception | None = None
            for attempt in range(EDGE_ATTEMPTS):
                try:
                    communicate = edge_tts.Communicate(text, self._voice)
                    await communicate.save(temp_path)
                    last_error = None
                    break
                except Exception as exc:  # network/websocket flakiness
                    last_error = exc
                    if self._stop_event.is_set():
                        return
                    if attempt + 1 < EDGE_ATTEMPTS:
                        logger.warning(
                            "Edge TTS denemesi %d/%d başarısız, tekrar denenecek",
                            attempt + 1,
                            EDGE_ATTEMPTS,
                        )
                        await asyncio.sleep(EDGE_RETRY_SECONDS)
            if last_error is not None:
                raise last_error

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
