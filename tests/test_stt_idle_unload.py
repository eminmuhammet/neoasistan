import time

from neo.voice.stt import WhisperSTT


def test_nothing_to_unload_when_model_was_never_loaded():
    stt = WhisperSTT()
    assert stt.is_loaded is False
    assert stt.unload_if_idle(idle_seconds=0) is False


def test_unloads_after_idle_period():
    stt = WhisperSTT()
    stt._model = object()  # stand-in for a loaded model
    stt._last_used = time.monotonic() - 10

    assert stt.unload_if_idle(idle_seconds=5) is True
    assert stt.is_loaded is False


def test_keeps_model_while_recently_used():
    stt = WhisperSTT()
    stt._model = object()
    stt._last_used = time.monotonic()

    assert stt.unload_if_idle(idle_seconds=300) is False
    assert stt.is_loaded is True
