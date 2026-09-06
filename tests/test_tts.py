import asyncio

from neo.voice.tts import EdgeTTS, FallbackTTS, SapiTTS, TTSUnavailableError


class _RecordingTTS:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.spoken: list[str] = []
        self.stopped = False

    async def speak(self, text: str) -> None:
        if self.fail:
            raise TTSUnavailableError("unavailable")
        self.spoken.append(text)

    def stop(self) -> None:
        self.stopped = True


def test_speak_empty_text_is_noop():
    tts = SapiTTS()
    asyncio.run(tts.speak(""))
    asyncio.run(tts.speak("   "))


def test_edge_tts_empty_text_is_noop():
    tts = EdgeTTS()
    asyncio.run(tts.speak(""))


def test_fallback_uses_primary_when_it_succeeds():
    primary = _RecordingTTS()
    secondary = _RecordingTTS()
    tts = FallbackTTS(primary, secondary)

    asyncio.run(tts.speak("merhaba"))

    assert primary.spoken == ["merhaba"]
    assert secondary.spoken == []


def test_fallback_falls_back_when_primary_unavailable():
    primary = _RecordingTTS(fail=True)
    secondary = _RecordingTTS()
    tts = FallbackTTS(primary, secondary)

    asyncio.run(tts.speak("merhaba"))

    assert secondary.spoken == ["merhaba"]


def test_fallback_stop_stops_both():
    primary = _RecordingTTS()
    secondary = _RecordingTTS()
    tts = FallbackTTS(primary, secondary)

    tts.stop()

    assert primary.stopped and secondary.stopped
