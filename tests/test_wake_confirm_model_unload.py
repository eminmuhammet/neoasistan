"""The wake-phrase confirm model has to stay loaded while NEO is listening.

This loop used to call unload_if_idle() on it every tick, freeing it after
five idle minutes. But NEO idles for hours -- that is its normal state --
so in practice every real wake-up became a cold start: the user said the
phrase, the model started loading from disk, and seconds went by before
anything was transcribed. Reported as "it never wakes up on the first try,
only the second"; the second try worked because the first had loaded the
model.

Its memory is the price of a wake word that answers. NEO_WAKE_CONFIRM_MODEL
is the knob for anyone who would rather pay less of it.
"""

import asyncio

import numpy as np

from neo.voice.wake_word import WakeWordListener


class StubSpotter:
    detection_window_seconds = 1.0

    def has_enough_templates(self, minimum=1):
        return True


class StubSTT:
    async def transcribe(self, audio):
        return ""


class RecordingConfirmSTT:
    def __init__(self) -> None:
        self.unload_calls = 0
        self.preloads = 0

    def unload_if_idle(self, idle_seconds=300.0):
        self.unload_calls += 1
        return False

    async def preload(self):
        self.preloads += 1

    async def transcribe(self, audio):
        return ""


def _run(confirm, wake_phrase="Neo uyan", monkeypatch=None):
    listener = WakeWordListener(
        StubSpotter(), StubSTT(), wake_phrase=wake_phrase, confirm_stt=confirm
    )
    monkeypatch.setattr(listener, "_open_stream", lambda: None)

    ticks = {"n": 0}

    def fake_collect_chunk(num_samples):
        ticks["n"] += 1
        if ticks["n"] >= 3:
            listener._running = False
        return np.zeros(0, dtype="float32")

    monkeypatch.setattr(listener, "_collect_chunk", fake_collect_chunk)

    async def on_wake():
        pass

    async def on_command(text):
        pass

    asyncio.run(listener.run(on_wake, on_command))
    return listener


def test_the_confirm_model_is_never_unloaded_while_listening(monkeypatch):
    confirm = RecordingConfirmSTT()
    _run(confirm, monkeypatch=monkeypatch)

    assert confirm.unload_calls == 0


def test_the_confirm_model_is_loaded_before_the_first_wake_attempt(monkeypatch):
    """Loading takes seconds. Paying that on the session's first wake puts
    the delay exactly where the user is waiting for an answer."""
    confirm = RecordingConfirmSTT()
    _run(confirm, monkeypatch=monkeypatch)

    assert confirm.preloads == 1


def test_nothing_is_preloaded_when_confirmation_is_disabled(monkeypatch):
    """An empty wake phrase turns confirmation off, so there is no model to
    pay for."""
    confirm = RecordingConfirmSTT()
    _run(confirm, wake_phrase="", monkeypatch=monkeypatch)

    assert confirm.preloads == 0
