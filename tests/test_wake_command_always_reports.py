"""Every way a command capture can end has to tell the UI it ended.

NEO sets the on-screen state to "Dinliyor" the moment the wake word fires,
and only the command callback moves it off again. So a capture that gets
discarded quietly -- too little speech, VAD found nothing, an empty
transcript, a broken recognizer -- used to leave the interface stuck on
"Dinliyor" indefinitely while the listener had already gone back to waiting
for the wake word. Reported twice as "hala dinliyorda kalıyor".
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

import neo.voice.wake_word as ww
from neo.voice.stt import STTUnavailableError
from neo.voice.wake_word import SAMPLE_RATE, WakeWordListener

LOUD = np.full(int(0.3 * SAMPLE_RATE), 0.5, dtype=np.float32)
QUIET = np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32)


class _Spotter:
    def has_enough_templates(self, minimum=3):
        return True

    detection_window_seconds = 1.4

    def __init__(self):
        self.fired = False

    def is_match(self, audio):
        # Fire exactly once, so the loop enters command capture a single time.
        if self.fired:
            return False
        self.fired = True
        return True


class _STT:
    def __init__(self, transcript="", error=False):
        self.transcript, self.error = transcript, error

    async def transcribe(self, audio):
        if self.error:
            raise STTUnavailableError("model yok")
        return self.transcript


def _run_capture(monkeypatch, stt, *, enough_speech=True, vad_seconds=1.0):
    """Drive run() through one wake + one command capture, return the
    arguments on_command was called with."""
    listener = WakeWordListener(_Spotter(), stt)
    monkeypatch.setattr(listener, "_open_stream", lambda: None)
    monkeypatch.setattr(listener, "_confirm_wake_phrase", lambda w: _true())
    monkeypatch.setattr(ww, "has_enough_speech", lambda *a, **k: enough_speech)
    monkeypatch.setattr(ww, "speech_seconds", lambda *a, **k: vad_seconds)

    # Loud to trip the wake word, loud again so the capture counts as having
    # heard speech (a pause before the user starts talking is not an end of
    # command), then enough silence to cross the end-of-command gap.
    script = [LOUD, LOUD] + [QUIET] * 8
    calls: list[str] = []

    def collect(n):
        if script:
            return script.pop(0)
        listener._running = False
        return np.zeros(0, dtype=np.float32)

    monkeypatch.setattr(listener, "_collect_chunk", collect)

    async def on_wake():
        pass

    async def on_command(text):
        calls.append(text)
        listener._running = False

    async def drive():
        await listener.run(on_wake, on_command)
        # on_command is scheduled, not awaited, by the listener.
        for _ in range(10):
            await asyncio.sleep(0)

    asyncio.run(drive())
    return calls


async def _true():
    return True


def test_reports_when_there_is_too_little_speech(monkeypatch):
    calls = _run_capture(monkeypatch, _STT("merhaba"), enough_speech=False)
    assert calls == [""]


def test_reports_when_vad_finds_no_speech(monkeypatch):
    calls = _run_capture(monkeypatch, _STT("merhaba"), vad_seconds=0.0)
    assert calls == [""]


def test_reports_when_the_transcript_is_empty(monkeypatch):
    calls = _run_capture(monkeypatch, _STT(""))
    assert calls == [""]


def test_reports_when_the_recognizer_is_broken(monkeypatch):
    calls = _run_capture(monkeypatch, _STT(error=True))
    assert calls == [""]


def test_a_real_command_is_still_passed_through(monkeypatch):
    calls = _run_capture(monkeypatch, _STT("saat kaç"))
    assert calls == ["saat kaç"]


def test_the_activation_chime_does_not_swallow_the_command(monkeypatch):
    """NEO mutes its own microphone while the "Dinliyorum" chime plays,
    which happens immediately after the wake word fires -- so the mute lands
    squarely inside the capture the wake word just opened.

    Unmuting used to clear that capture, and since nothing reported the
    capture had ended, the interface sat on "Dinliyor" indefinitely while
    the listener had already gone back to waiting for the wake word. The
    user had done nothing wrong and got no way to tell.
    """
    stt = _STT("saat kaç")
    listener = WakeWordListener(_Spotter(), stt)
    monkeypatch.setattr(listener, "_open_stream", lambda: None)
    monkeypatch.setattr(listener, "_confirm_wake_phrase", lambda w: _true())
    monkeypatch.setattr(ww, "has_enough_speech", lambda *a, **k: True)
    monkeypatch.setattr(ww, "speech_seconds", lambda *a, **k: 1.0)

    # Wake, two chunks swallowed by the chime, then the user speaks.
    script = [LOUD, LOUD, LOUD, LOUD] + [QUIET] * 8
    muted = iter([False, True, True] + [False] * 20)
    calls: list[str] = []

    def collect(n):
        if script:
            return script.pop(0)
        listener._running = False
        return np.zeros(0, dtype=np.float32)

    monkeypatch.setattr(listener, "_collect_chunk", collect)

    async def on_wake():
        pass

    async def on_command(text):
        calls.append(text)
        listener._running = False

    async def drive():
        await listener.run(on_wake, on_command, is_muted=lambda: next(muted, False))
        for _ in range(10):
            await asyncio.sleep(0)

    asyncio.run(drive())
    assert calls == ["saat kaç"], "susturma sonrasi komut yakalamasi kaybolmamali"
