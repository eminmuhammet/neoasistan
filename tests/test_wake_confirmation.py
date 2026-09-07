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


def _listener(tmp_path, confirm_stt, phrase="Neo uyan", confirm_threshold=None):
    spotter = KeywordSpotter(tmp_path / "t.npz")
    kwargs = {} if confirm_threshold is None else {"confirm_threshold": confirm_threshold}
    return WakeWordListener(
        spotter, FakeSTT(), wake_phrase=phrase, confirm_stt=confirm_stt, **kwargs
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


def test_confirmation_is_rate_limited(tmp_path, monkeypatch):
    """Live log: the acoustic matcher fired repeatedly on the same noise and
    the recognizer ran three times inside two seconds -- confirmation costs
    about a second of CPU, so the machine spent its time transcribing an
    empty room instead of answering the user, which was a large part of why
    replies felt slow."""
    import neo.voice.wake_word as ww

    stt = FakeSTT("Neo uyan")
    listener = _listener(tmp_path, stt)

    clock = [100.0]
    monkeypatch.setattr(ww.time, "monotonic", lambda: clock[0])

    assert asyncio.run(listener._confirm_wake_phrase(AUDIO)) is True
    assert stt.calls == 1

    # Immediately again: within the cooldown window, must not re-transcribe.
    clock[0] += 0.1
    assert asyncio.run(listener._confirm_wake_phrase(AUDIO)) is False
    assert stt.calls == 1, "cooldown icindeyken tekrar transcribe cagirilmamali"

    # After the cooldown elapses, confirmation runs again.
    clock[0] += ww.CONFIRM_COOLDOWN_SECONDS
    assert asyncio.run(listener._confirm_wake_phrase(AUDIO)) is True
    assert stt.calls == 2


def test_confirm_threshold_is_configurable_per_listener(tmp_path):
    """The right false-accept/false-reject tradeoff genuinely depends on
    the room and microphone -- every wake-word engine exposes this as a
    tunable sensitivity rather than one fixed value for everyone."""
    marginal = FakeSTT("Ne o ya")  # scores ~0.80 against "Neo uyan"

    lenient = _listener(tmp_path, marginal, confirm_threshold=0.5)
    assert asyncio.run(lenient._confirm_wake_phrase(AUDIO)) is True

    strict = _listener(tmp_path, marginal, confirm_threshold=0.99)
    assert asyncio.run(strict._confirm_wake_phrase(AUDIO)) is False


def test_confirm_threshold_defaults_to_the_module_constant(tmp_path):
    import neo.voice.wake_word as ww

    listener = _listener(tmp_path, FakeSTT("Neo uyan"))
    assert listener._confirm_threshold == ww.WAKE_PHRASE_THRESHOLD


def test_the_default_threshold_separates_real_attempts_from_noise(tmp_path):
    """Scored against every wake transcript in logs/neo.log rather than
    tidy text, because that gap is what the constant is chosen from.

    'uyan.' is the case that drove the current value: a genuine "Neo uyan"
    whose quieter first word Whisper dropped. It scored 0.67 whole-string
    and was rejected, forcing the user to repeat something already said
    correctly.
    """
    real = ["uyan.", "Ne yok, uyan.", "Ne o ya?", "Neo uyan.", "Ne o uyan"]
    noise = ["Ne oluyor?", "Ne oldu?", "Teşekkürler.", "Zeynep.",
             "Yemek yapacağım.", "Söyledik mi?", "uyuyor musun"]

    for transcript in real:
        listener = _listener(tmp_path, FakeSTT(transcript))
        assert asyncio.run(listener._confirm_wake_phrase(AUDIO)) is True, transcript

    for transcript in noise:
        listener = _listener(tmp_path, FakeSTT(transcript))
        assert asyncio.run(listener._confirm_wake_phrase(AUDIO)) is False, transcript
