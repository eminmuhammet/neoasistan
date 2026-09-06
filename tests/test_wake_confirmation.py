import asyncio

import numpy as np

from neo.voice.keyword_spotter import KeywordSpotter
from neo.voice.stt import STTUnavailableError
from neo.voice.wake_word import WakeWordListener

AUDIO = np.zeros(16000, dtype=np.float32)


class FakeSTT:
    def __init__(self, transcript="", error=False):
        self.transcript = transcript
        self.error = error
        self.calls = 0

    async def transcribe(self, audio):
        self.calls += 1
        if self.error:
            raise STTUnavailableError("model yok")
        return self.transcript


def _listener(tmp_path, confirm_stt, phrase="Neo uyan"):
    spotter = KeywordSpotter(tmp_path / "t.npz")
    return WakeWordListener(
        spotter, FakeSTT(), wake_phrase=phrase, confirm_stt=confirm_stt
    )


def test_confirmation_accepts_the_wake_phrase(tmp_path):
    stt = FakeSTT("Neo uyan")
    listener = _listener(tmp_path, stt)

    assert asyncio.run(listener._confirm_wake_phrase(AUDIO)) is True
    assert stt.calls == 1


def test_confirmation_rejects_unrelated_speech(tmp_path):
    """The acoustic matcher accepted this window; recognition is what stops
    ordinary conversation from waking NEO, since on this user's voice the
    enrolled takes were no closer to each other than to unrelated speech."""
    listener = _listener(tmp_path, FakeSTT("bugün programımda neler var"))

    assert asyncio.run(listener._confirm_wake_phrase(AUDIO)) is False


def test_confirmation_rejects_an_empty_transcript(tmp_path):
    listener = _listener(tmp_path, FakeSTT(""))
    assert asyncio.run(listener._confirm_wake_phrase(AUDIO)) is False


def test_a_broken_confirmer_never_makes_neo_deaf(tmp_path):
    """Confirmation refines the acoustic decision; it must not be able to
    disable waking up altogether."""
    listener = _listener(tmp_path, FakeSTT(error=True))

    assert asyncio.run(listener._confirm_wake_phrase(AUDIO)) is True


def test_confirmation_is_optional(tmp_path):
    """Without a confirmer configured the acoustic verdict stands, so
    continuous listening keeps working with the feature turned off."""
    listener = _listener(tmp_path, None)

    assert asyncio.run(listener._confirm_wake_phrase(AUDIO)) is True


def test_confirmation_follows_the_configured_phrase(tmp_path):
    listener = _listener(tmp_path, FakeSTT("bilgisayar dinle"), phrase="bilgisayar dinle")
    assert asyncio.run(listener._confirm_wake_phrase(AUDIO)) is True

    other = _listener(tmp_path, FakeSTT("bilgisayar dinle"), phrase="Neo uyan")
    assert asyncio.run(other._confirm_wake_phrase(AUDIO)) is False
