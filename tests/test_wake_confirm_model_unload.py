"""The wake-word model has to be loaded once, up front, before the loop
starts watching for it.

Loading (and, on first run, downloading) the ONNX model takes real time.
Doing it lazily on the first genuine "Hey Jarvis" attempt would put that
delay directly between the user speaking and NEO reacting -- which reads
as "it didn't hear me" -- so run() preloads it before entering the loop.
"""

import asyncio

import numpy as np

from neo.voice.wake_word import WakeWordListener


class StubSTT:
    async def transcribe(self, audio):
        return ""


class RecordingWakeSpotter:
    def __init__(self) -> None:
        self.preloads = 0

    def preload(self):
        self.preloads += 1

    def feed(self, chunk):
        return False


def _run_briefly(listener, monkeypatch):
    monkeypatch.setattr(listener, "_open_stream", lambda: None)

    ticks = {"n": 0}

    def fake_collect_chunk(num_samples):
        ticks["n"] += 1
        if ticks["n"] >= 3:
            listener._running = False
        return np.zeros(0, dtype="float32")

    monkeypatch.setattr(listener, "_collect_chunk", fake_collect_chunk)

    asyncio.run(listener.run(on_wake=lambda: _noop(), on_command=lambda text: _noop()))


async def _noop():
    pass


def test_the_wake_model_is_preloaded_once_before_listening_starts(monkeypatch):
    spotter = RecordingWakeSpotter()
    listener = WakeWordListener(spotter, StubSTT())

    _run_briefly(listener, monkeypatch)

    assert spotter.preloads == 1
