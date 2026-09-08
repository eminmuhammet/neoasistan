from __future__ import annotations

import asyncio
import io
import logging
import threading
import wave
from pathlib import Path

import numpy as np

from .audio_features import trim_silence

logger = logging.getLogger(__name__)

# The cue's length is subtracted directly from the front of the user's
# command, so it is the number this module is built around.
#
# NEO mutes its microphone while the cue plays and drains whatever queued
# up, because that audio has the cue bleeding into it from the speakers.
# Anything the user says in that window is unrecoverable.
#
# Measured on the previous implementation -- a spoken "Dinliyorum efendim."
# played straight from edge-tts mp3 through MCI -- the window was about
# three seconds: 1.8s of that mp3 was trailing silence edge-tts pads onto
# every clip, and MCI added roughly a second opening and closing the
# decoder. Users start talking the moment the screen says "Dinliyor", so
# "bana neler yapabildiğini anlat" reached the recognizer as "yapabildiğini
# anlat".
#
# Three things bring it to about half a second, which is short enough that
# the user is still hearing the cue when they start:
#   * a one-word phrase,
#   * the padding trimmed off before caching, and
#   * a persistent output stream, so the device is opened once at startup
#     rather than on every wake (that alone was ~1s per play).
CUE_PHRASE = "Buyrun."
CUE_VOICE = "tr-TR-AhmetNeural"

_SAMPLE_RATE = 22050

# Synthesized once and kept on disk as ready-to-play PCM. Generating on
# every wake would put a network round trip between the user speaking and
# hearing that they were heard -- the exact delay this cue exists to remove.
_CACHE_DIRNAME = "cues"
_CUE_FILENAME = "cue.wav"

# Fallback tone, used when the spoken cue could not be synthesized (no
# network on first run, edge-tts missing). Two rising sine notes with an
# envelope rather than a winsound.Beep square wave, which genuinely did
# sound like a BIOS error.
_NOTES = ((784.0, 0.09), (1046.5, 0.13))   # G5 -> C6: "I'm listening"
_ATTACK = 0.012
_RELEASE = 0.045

_cache_dir: Path | None = None
_cue_audio: np.ndarray | None = None
_stream = None
_stream_lock = threading.Lock()


def configure(data_dir: Path) -> None:
    """Points the cue cache at the user data folder (survives updates)."""
    global _cache_dir
    _cache_dir = data_dir / _CACHE_DIRNAME


# -- synthesis ---------------------------------------------------------------


def _decode(path: Path) -> np.ndarray:
    """mp3 -> mono float32 at _SAMPLE_RATE, via the decoder faster-whisper
    already pulls in."""
    import av

    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=_SAMPLE_RATE
    )
    parts: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.audio[0]):
            for out in resampler.resample(frame):
                parts.append(out.to_ndarray().reshape(-1))
    for out in resampler.resample(None):
        parts.append(out.to_ndarray().reshape(-1))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32) / 32768.0


def _write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(_SAMPLE_RATE)
        out.writeframes(pcm.tobytes())


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as src:
        raw = src.readframes(src.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


async def prepare_cues(voice: str = CUE_VOICE) -> None:
    """Synthesizes the cue if it isn't cached yet. Safe to call on every
    start: it only does work the first time, and failing is not fatal --
    the fallback tone still tells the user NEO woke up."""
    global _cue_audio
    if _cache_dir is None:
        return

    cached = _cache_dir / _CUE_FILENAME
    if cached.exists() and cached.stat().st_size > 0:
        try:
            _cue_audio = _read_wav(cached)
            return
        except Exception:
            logger.exception("Önbellekteki uyandırma sesi okunamadı, yeniden üretilecek")

    try:
        import edge_tts
    except ImportError:
        logger.info("edge-tts yok, uyandırma sesi için ton kullanılacak")
        return

    _cache_dir.mkdir(parents=True, exist_ok=True)
    raw = _cache_dir / "cue_raw.mp3"
    try:
        await edge_tts.Communicate(CUE_PHRASE, voice).save(str(raw))
        # Trimmed before caching, not at play time: edge-tts pads every clip
        # with over a second of silence, and that silence would be muted
        # microphone time on every single wake.
        audio = trim_silence(_decode(raw), _SAMPLE_RATE)
        if audio.size == 0:
            raise ValueError("kırpma sonrası ses kalmadı")
        _write_wav(cached, audio)
        _cue_audio = audio
        logger.info(
            "Uyandırma sesi hazırlandı: %r (%.2f sn)", CUE_PHRASE, audio.size / _SAMPLE_RATE
        )
    except Exception:
        logger.exception("Uyandırma sesi üretilemedi, ton kullanılacak")
        cached.unlink(missing_ok=True)
    finally:
        raw.unlink(missing_ok=True)


# -- fallback tone -----------------------------------------------------------


def _render_tone() -> np.ndarray:
    chunks = []
    for freq, seconds in _NOTES:
        n = int(_SAMPLE_RATE * seconds)
        t = np.arange(n, dtype=np.float32) / _SAMPLE_RATE
        # A quiet second harmonic keeps it from sounding like a test tone.
        tone = np.sin(2 * np.pi * freq * t) + 0.18 * np.sin(4 * np.pi * freq * t)

        env = np.ones(n, dtype=np.float32)
        attack = min(int(_ATTACK * _SAMPLE_RATE), n)
        release = min(int(_RELEASE * _SAMPLE_RATE), n - attack) if n > attack else 0
        if attack:
            env[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
        if release:
            env[-release:] = np.linspace(1.0, 0.0, release, dtype=np.float32) ** 2
        chunks.append(tone * env)

    samples = np.concatenate(chunks)
    return samples / max(float(np.abs(samples).max()), 1e-6) * 0.35


# -- playback ----------------------------------------------------------------


def _play(audio: np.ndarray) -> None:
    """Writes to a stream kept open for the life of the process.

    sounddevice's one-shot `play()` opens and closes the output device on
    every call, which cost about a second here -- twice the length of the
    cue itself, and all of it muted microphone.
    """
    global _stream
    import sounddevice as sd

    with _stream_lock:
        if _stream is None:
            _stream = sd.OutputStream(
                samplerate=_SAMPLE_RATE, channels=1, dtype="float32"
            )
            _stream.start()
        try:
            _stream.write(audio.reshape(-1, 1))
        except Exception:
            # The device can disappear (headphones unplugged, default
            # output switched). Drop the stream so the next wake reopens it
            # rather than failing forever.
            try:
                _stream.close()
            finally:
                _stream = None
            raise


def cue_seconds() -> float:
    """How long the microphone stays muted for the cue."""
    audio = _cue_audio if _cue_audio is not None else _render_tone()
    return audio.size / _SAMPLE_RATE


async def play_activation_chime() -> None:
    """Tells the user NEO is listening without them having to look at the
    screen.

    The caller mutes the microphone around this, so this must stay brief:
    see the note on CUE_PHRASE for what a long cue costs.
    """
    audio = _cue_audio if _cue_audio is not None else _render_tone()
    try:
        await asyncio.to_thread(_play, audio)
    except Exception:
        logger.exception("Uyandırma sesi çalınamadı")
