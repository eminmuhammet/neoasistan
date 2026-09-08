"""A follow-up window lets the user reply without repeating the wake word.

Requested directly: saying "Neo uyan" with the crisp diction it needs to be
recognized is realistic once, deliberately, but not for every turn of an
ordinary back-and-forth -- "gunluk hayatta bu her zaman mumkun degil".
MainWindow opens the window right after speaking a reply
(neo/ui/main_window.py); this file covers the primitive in isolation.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

import neo.voice.wake_word as ww
from neo.voice.wake_word import SAMPLE_RATE, WakeWordListener

LOUD = np.full(int(0.3 * SAMPLE_RATE), 0.5, dtype=np.float32)
QUIET = np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32)


class _Spotter:
    detection_window_seconds = 1.4

    def __init__(self):
        self.match_calls = 0

    def has_enough_templates(self, minimum=3):
        return True

    def is_match(self, audio):
        # Never matches acoustically -- any wake-up seen in these tests can
        # only have come from the follow-up bypass, not this path.
        self.match_calls += 1
        return False


class _STT:
    def __init__(self, transcript="saat kaç"):
        self.transcript = transcript

    async def transcribe(self, audio):
        return self.transcript


def _listener(monkeypatch, spotter=None):
    listener = WakeWordListener(spotter or _Spotter(), _STT())
    monkeypatch.setattr(listener, "_open_stream", lambda: None)
    monkeypatch.setattr(ww, "has_enough_speech", lambda *a, **k: True)
    monkeypatch.setattr(ww, "speech_seconds", lambda *a, **k: 1.0)
    return listener


def _drive(listener, script, monkeypatch):
    """Feeds `script` (a list of audio chunks) through one run() call and
    returns (on_wake call count, on_command call list)."""
    wakes = []
    calls = []

    def collect(n):
        if script:
            return script.pop(0)
        listener._running = False
        return np.zeros(0, dtype=np.float32)

    monkeypatch.setattr(listener, "_collect_chunk", collect)

    async def on_wake():
        wakes.append(1)

    async def on_command(text):
        calls.append(text)
        listener._running = False

    async def run():
        await listener.run(on_wake, on_command)
        for _ in range(10):
            await asyncio.sleep(0)

    asyncio.run(run())
    return wakes, calls


def test_a_followup_window_lets_speech_through_without_the_wake_word(monkeypatch):
    spotter = _Spotter()
    listener = _listener(monkeypatch, spotter)
    listener.open_followup_window()

    wakes, calls = _drive(listener, [LOUD, LOUD] + [QUIET] * 8, monkeypatch)

    assert wakes == [1]
    assert calls == ["saat kaç"]
    assert spotter.match_calls == 0, "acoustic wake-word matching must be bypassed"


def test_without_an_open_window_the_wake_word_is_still_required(monkeypatch):
    spotter = _Spotter()
    listener = _listener(monkeypatch, spotter)
    # No open_followup_window() call.

    wakes, calls = _drive(listener, [LOUD, LOUD] + [QUIET] * 8, monkeypatch)

    assert wakes == []
    assert calls == []
    assert spotter.match_calls > 0, "should have gone through the normal acoustic gate"


def test_one_utterance_consumes_the_window(monkeypatch):
    """A follow-up window is meant for the next thing said, not every
    utterance until it times out."""
    spotter = _Spotter()
    listener = _listener(monkeypatch, spotter)

    # A fake clock: the loop's own reactivation cooldown after the first
    # command (2s of real time) would otherwise swallow the second
    # utterance before it ever reaches the acoustic gate, since this test
    # drives the loop far faster than real audio arrives. Patched before
    # open_followup_window() so the window's own deadline is computed off
    # the same clock -- otherwise a real wall-clock deadline would outlast
    # the whole fake-time test and the bypass would never expire.
    clock = [1_000.0]
    monkeypatch.setattr(ww.time, "monotonic", lambda: clock[0])
    listener.open_followup_window()

    # First utterance consumes the window; second, later utterance in the
    # same run must go through the normal (here: always-rejecting) gate.
    # _drive stops as soon as on_command fires once, so this test drives
    # the loop itself to keep going past that first result. The 8 quiet
    # chunks between the two utterances aren't just for the end-of-command
    # pause (4 would cover that) -- they also have to outlast the loop's
    # own 2s reactivation cooldown after finalizing the first command, or
    # the second utterance never reaches the gate this test is checking.
    script = [LOUD, LOUD] + [QUIET] * 4 + [QUIET] * 8 + [LOUD]
    wakes, calls = [], []

    def collect(n):
        clock[0] += 0.3  # WakeWordConfig.step_seconds
        if script:
            return script.pop(0)
        listener._running = False
        return np.zeros(0, dtype=np.float32)

    monkeypatch.setattr(listener, "_collect_chunk", collect)

    async def on_wake():
        wakes.append(1)

    async def on_command(text):
        calls.append(text)

    async def run():
        await listener.run(on_wake, on_command)
        for _ in range(10):
            await asyncio.sleep(0)

    asyncio.run(run())

    assert wakes == [1]
    assert calls == ["saat kaç"]
    assert spotter.match_calls > 0, "the second utterance should have hit the normal gate"


def test_an_expired_window_no_longer_bypasses_the_wake_word(monkeypatch):
    spotter = _Spotter()
    listener = _listener(monkeypatch, spotter)
    listener.open_followup_window(seconds=-1.0)  # already in the past

    wakes, calls = _drive(listener, [LOUD, LOUD] + [QUIET] * 8, monkeypatch)

    assert wakes == []
    assert calls == []
    assert spotter.match_calls > 0


def test_stopping_the_listener_closes_any_open_window(monkeypatch):
    listener = _listener(monkeypatch)
    listener.open_followup_window()

    listener.stop()

    assert listener._followup_deadline == 0.0


def test_cancel_followup_window(monkeypatch):
    listener = _listener(monkeypatch)
    listener.open_followup_window()

    listener.cancel_followup_window()

    assert listener._followup_deadline == 0.0


def test_a_noise_during_the_window_still_goes_through_vad_before_reaching_claude(monkeypatch):
    """The bypass skips the wake-word match, not the hallucination guards a
    normal command still has to clear -- a stray sound during the window
    must not reach the agent as if it were a real command."""
    spotter = _Spotter()
    listener = _listener(monkeypatch, spotter)
    monkeypatch.setattr(ww, "has_enough_speech", lambda *a, **k: False)
    listener.open_followup_window()

    wakes, calls = _drive(listener, [LOUD, LOUD] + [QUIET] * 8, monkeypatch)

    assert wakes == [1], "on_wake still fires -- the UI has to show it's listening"
    assert calls == [""], "but the empty result resets the UI rather than reaching the agent"
