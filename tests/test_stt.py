import asyncio
from unittest.mock import patch

from neo.voice.stt import WhisperSTT


class _FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


def test_transcribe_joins_and_strips_segments():
    stt = WhisperSTT()
    fake_model = type("FakeModel", (), {})()
    fake_model.transcribe = lambda audio, **kwargs: (
        [_FakeSegment(" merhaba "), _FakeSegment("dünya ")],
        None,
    )

    with patch.object(WhisperSTT, "_ensure_model", return_value=fake_model):
        result = asyncio.run(stt.transcribe("fake_audio"))

    assert result == "merhaba dünya"
