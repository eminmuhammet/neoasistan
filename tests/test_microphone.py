import numpy as np

from neo.voice.microphone import PushToTalkRecorder


def test_stop_without_start_returns_empty_array():
    recorder = PushToTalkRecorder()
    audio = recorder.stop()
    assert isinstance(audio, np.ndarray)
    assert audio.size == 0
