"""The activation cue's defining constraint is its length, not its sound.

NEO mutes the microphone while the cue plays and drains whatever queued up,
because that audio has the cue bleeding into it from the speakers. So the
cue's duration is subtracted directly from the front of the user's command.

The cue used to be a spoken "Dinliyorum efendim." at 2 seconds, and users --
who start talking the moment the screen says "Dinliyor" -- lost the first two
seconds of every command: "bana neler yapabildiğini anlat" reached the
recognizer as "yapabildiğini anlat".
"""

import asyncio
import io
import wave

import numpy as np

from neo.voice import chime

# Comfortably longer than the current cue, far shorter than the spoken one
# it replaced. A cue that grows past this is back to eating commands.
MAX_CUE_SECONDS = 0.9


def _tone_wav() -> bytes:
    """The fallback tone as a WAV, so the audio checks below can read it
    with the stdlib rather than reaching into numpy internals."""
    pcm = (np.clip(chime._render_tone(), -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(chime._SAMPLE_RATE)
        out.writeframes(pcm.tobytes())
    return buffer.getvalue()


def test_the_cue_is_short_enough_not_to_eat_the_command():
    assert chime.cue_seconds() <= MAX_CUE_SECONDS


def test_the_rendered_audio_matches_the_declared_length():
    """cue_seconds() is what the "keep it short" guarantee is checked
    against, so it has to describe the audio actually played."""
    with wave.open(io.BytesIO(_tone_wav())) as w:
        seconds = w.getnframes() / w.getframerate()

    assert abs(seconds - chime.cue_seconds()) < 0.02


def test_the_cue_is_playable_pcm():
    with wave.open(io.BytesIO(_tone_wav())) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() > 0


def test_the_cue_does_not_clip():
    """Rendered as floats and scaled before quantizing; a cue that clipped
    would buzz."""
    import numpy as np

    with wave.open(io.BytesIO(_tone_wav())) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")

    assert int(np.abs(pcm).max()) < 32767


def test_the_cue_starts_and_ends_silent():
    """Without an envelope the abrupt edges click, which is what made the
    old winsound.Beep cue sound like a BIOS error."""
    import numpy as np

    with wave.open(io.BytesIO(_tone_wav())) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")

    peak = int(np.abs(pcm).max())
    assert abs(int(pcm[0])) < peak * 0.05
    assert abs(int(pcm[-1])) < peak * 0.05


def test_the_spoken_cue_is_used_when_one_has_been_prepared(monkeypatch):
    """The synthesized word is cached as trimmed PCM, so waking NEO costs no
    network round trip and no decode."""
    spoken = np.zeros(int(chime._SAMPLE_RATE * 0.4), dtype=np.float32)
    monkeypatch.setattr(chime, "_cue_audio", spoken)

    played = []
    monkeypatch.setattr(chime, "_play", lambda a: played.append(a))
    asyncio.run(chime.play_activation_chime())

    assert len(played) == 1
    assert played[0] is spoken
    assert chime.cue_seconds() <= MAX_CUE_SECONDS


def test_the_tone_is_used_when_no_spoken_cue_was_prepared(monkeypatch):
    """First run with no network must still acknowledge the wake word."""
    monkeypatch.setattr(chime, "_cue_audio", None)

    played = []
    monkeypatch.setattr(chime, "_play", lambda a: played.append(a))
    asyncio.run(chime.play_activation_chime())

    assert len(played) == 1
    assert played[0].size > 0


def test_playback_failure_never_propagates(monkeypatch):
    """A machine with no working audio output must still be able to wake up."""
    def boom(_data):
        raise OSError("ses aygıtı yok")

    monkeypatch.setattr(chime, "_play", boom)

    asyncio.run(chime.play_activation_chime())  # must not raise
