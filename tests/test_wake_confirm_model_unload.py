"""The confirm model (see neo/voice/wake_word.py) has no other owner
checking on it -- unlike the main STT model, which the GUI's idle timer
unloads, this one used to stay resident in memory for as long as
continuous listening was on, which for most users is "always". The fix
was making WakeWordListener.run() check unload_if_idle() on its own confirm
model every loop tick (cheap: a single monotonic-time comparison unless
actually time to unload).
"""

import asyncio

import numpy as np

from neo.voice.wake_word import WakeWordListener


class StubSpotter:
    def has_enough_templates(self, minimum=1):
        return True

    def detection_window_seconds(self):
        return 1.0


class StubSTT:
    async def transcribe(self, audio):
        return ""


class CountingConfirmSTT:
    def __init__(self) -> None:
        self.unload_calls = 0

    def unload_if_idle(self, idle_seconds=300.0):
        self.unload_calls += 1
        return False

    async def transcribe(self, audio):
        return ""


def test_run_loop_checks_confirm_model_idle_unload_every_tick(monkeypatch):
    confirm = CountingConfirmSTT()
    spotter = StubSpotter()
    # detection_window_seconds is a plain attribute on the real class, not
    # a method -- match that here.
    spotter.detection_window_seconds = 1.0
    listener = WakeWordListener(spotter, StubSTT(), wake_phrase="Neo uyan", confirm_stt=confirm)

    monkeypatch.setattr(listener, "_open_stream", lambda: None)

    call_count = {"n": 0}

    def fake_collect_chunk(num_samples):
        call_count["n"] += 1
        if call_count["n"] >= 3:
            listener._running = False
        return np.zeros(0, dtype="float32")

    monkeypatch.setattr(listener, "_collect_chunk", fake_collect_chunk)

    asyncio.run(listener.run(on_wake=lambda: None, on_command=lambda text: None))

    assert confirm.unload_calls >= 3
