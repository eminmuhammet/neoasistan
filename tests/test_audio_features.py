import numpy as np

from neo.voice.audio_features import (
    dtw_distance,
    extract_mfcc,
    has_enough_speech,
    normalize_features,
    speech_seconds,
    trim_silence,
)

SAMPLE_RATE = 16000


def _voiced(seconds: float, total_seconds: float, offset_seconds: float = 0.5) -> np.ndarray:
    """A crude voiced-speech stand-in: a harmonic stack (like vocal folds)
    under a smooth envelope, which Silero VAD accepts as speech where flat
    tones and noise are rejected."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    tone = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate([120, 240, 360, 480, 720]))
    tone = (tone * np.hanning(len(t)) * 0.25).astype(np.float32)

    buffer = np.zeros(int(total_seconds * SAMPLE_RATE), dtype=np.float32)
    start = int(offset_seconds * SAMPLE_RATE)
    buffer[start : start + len(tone)] = tone
    return buffer


def test_near_silent_buffer_does_not_have_enough_speech():
    # This is the guard against Whisper hallucinating fluent-sounding but
    # fabricated text ("Altyazı M.K.", "abone olmayı unutmayın") when fed
    # audio that's mostly silence/noise.
    buffer = np.zeros(16000, dtype=np.float32)
    assert has_enough_speech(buffer, energy_threshold=0.02, min_voiced_samples=6400) is False


def test_buffer_with_real_speech_has_enough_speech():
    buffer = np.zeros(16000, dtype=np.float32)
    buffer[4000:12000] = 0.3  # a loud 0.5s stretch, well above threshold
    assert has_enough_speech(buffer, energy_threshold=0.02, min_voiced_samples=6400) is True


def test_borderline_buffer_just_under_minimum_is_rejected():
    buffer = np.zeros(16000, dtype=np.float32)
    buffer[:6399] = 0.3  # one sample short of the minimum
    assert has_enough_speech(buffer, energy_threshold=0.02, min_voiced_samples=6400) is False


def test_loud_transient_passes_amplitude_guard_but_is_not_speech():
    """The bug this pair of guards exists for: a single loud transient (a key
    press, a door) clears the amplitude threshold, so `has_enough_speech`
    waves it through and Whisper gets handed 7.8s of effective silence --
    which it then either transcribes as nothing (wasted CPU) or hallucinates
    over. VAD has to be the one making the real call."""
    buffer = np.zeros(int(7.8 * SAMPLE_RATE), dtype=np.float32)
    buffer[3 * SAMPLE_RATE : 3 * SAMPLE_RATE + 2400] = 0.35  # 0.15s spike

    assert has_enough_speech(buffer, energy_threshold=0.006, min_voiced_samples=1920) is True
    assert speech_seconds(buffer, SAMPLE_RATE) == 0.0


def test_speech_seconds_rejects_silence_and_room_noise():
    silence = np.zeros(int(4.0 * SAMPLE_RATE), dtype=np.float32)
    noise = (np.random.RandomState(0).randn(int(4.0 * SAMPLE_RATE)) * 0.01).astype(np.float32)

    assert speech_seconds(silence, SAMPLE_RATE) == 0.0
    assert speech_seconds(noise, SAMPLE_RATE) == 0.0


def test_speech_seconds_accepts_real_speech():
    # Comfortably over the 0.25s the wake-word loop requires.
    assert speech_seconds(_voiced(1.2, total_seconds=4.0), SAMPLE_RATE) >= 0.25


def test_wake_gate_separates_short_speech_from_impacts():
    """The wake gate rejects only an exact 0.00s reading, because the wake
    word is one short word and this must never be what stops a real "Neo"
    from registering. Measured separation is wide -- speech lands above 1s
    of detected voice, impacts and noise at exactly zero -- so guard that
    the two classes stay on their own sides."""
    from neo.voice.keyword_spotter import _WAKE_MIN_SPEECH_MS as WAKE_MIN_SPEECH_MS

    window = 1.4
    short_word = _voiced(0.2, total_seconds=window, offset_seconds=0.3)

    impact = np.zeros(int(window * SAMPLE_RATE), dtype=np.float32)
    decay = np.arange(3200) / SAMPLE_RATE
    impact[int(0.5 * SAMPLE_RATE) : int(0.5 * SAMPLE_RATE) + 3200] = (
        np.sin(2 * np.pi * 60 * decay) * np.exp(-decay * 25) * 0.5
    ).astype(np.float32)

    loud_noise = (
        np.random.RandomState(0).randn(int(window * SAMPLE_RATE)) * 0.08
    ).astype(np.float32)

    assert speech_seconds(short_word, SAMPLE_RATE, WAKE_MIN_SPEECH_MS) > 0.0
    assert speech_seconds(impact, SAMPLE_RATE, WAKE_MIN_SPEECH_MS) == 0.0
    assert speech_seconds(loud_noise, SAMPLE_RATE, WAKE_MIN_SPEECH_MS) == 0.0


def test_speech_seconds_handles_empty_buffer():
    assert speech_seconds(np.zeros(0, dtype=np.float32), SAMPLE_RATE) == 0.0


def test_speech_seconds_signals_unavailable_rather_than_no_speech(monkeypatch):
    """When the detector can't run we must not report 0.0 -- callers treat
    that as "no speech" and would silently swallow real commands. -1.0 lets
    them fall back to the amplitude guard instead."""
    import builtins

    real_import = builtins.__import__

    def fail_vad(name, *args, **kwargs):
        if name == "faster_whisper.vad":
            raise ImportError("no vad")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_vad)
    assert speech_seconds(np.ones(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE) == -1.0


def test_trim_silence_keeps_loud_region_and_drops_far_silence():
    sample_rate = 1000
    audio = np.zeros(3000, dtype=np.float32)
    audio[1000:1500] = 1.0  # the "word"
    result = trim_silence(audio, sample_rate, pad_seconds=0.0)
    assert result.shape[0] == 500
    assert np.all(result == 1.0)


def test_trim_silence_keeps_full_word_regardless_of_length():
    # A "slow" and a "fast" version of the same loud region: trimming must
    # keep each one's full extent rather than forcing a fixed duration that
    # would truncate the slow one or pad the fast one with extra silence.
    sample_rate = 1000
    slow = np.zeros(2000, dtype=np.float32)
    slow[500:1500] = 1.0  # 1.0s loud region
    fast = np.zeros(2000, dtype=np.float32)
    fast[500:900] = 1.0  # 0.4s loud region

    slow_trimmed = trim_silence(slow, sample_rate, pad_seconds=0.0)
    fast_trimmed = trim_silence(fast, sample_rate, pad_seconds=0.0)

    assert slow_trimmed.shape[0] == 1000
    assert fast_trimmed.shape[0] == 400


def test_trim_silence_returns_original_when_all_silent():
    audio = np.zeros(100, dtype=np.float32)
    assert trim_silence(audio, sample_rate=1000).shape[0] == 100


def test_normalize_features_resamples_to_fixed_frame_count():
    short = np.random.RandomState(0).randn(10, 13)
    long = np.random.RandomState(1).randn(90, 13)

    assert normalize_features(short, target_frames=40).shape == (40, 13)
    assert normalize_features(long, target_frames=40).shape == (40, 13)


def test_normalize_features_handles_empty_input():
    empty = np.zeros((0, 13))
    result = normalize_features(empty, target_frames=40)
    assert result.shape == (40, 13)


def test_extract_mfcc_shape():
    sample_rate = 16000
    audio = (np.random.RandomState(0).randn(sample_rate) * 0.1).astype(np.float32)
    mfcc = extract_mfcc(audio, sample_rate, n_mfcc=13)
    assert mfcc.shape[1] == 13
    assert mfcc.shape[0] > 0


def test_extract_mfcc_empty_audio():
    mfcc = extract_mfcc(np.zeros(0, dtype=np.float32), 16000)
    assert mfcc.shape == (0, 13)


def test_dtw_distance_zero_for_identical_sequences():
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert dtw_distance(a, a) == 0.0


def test_dtw_distance_positive_for_different_sequences():
    a = np.array([[0.0, 0.0], [0.0, 0.0]])
    b = np.array([[10.0, 10.0], [10.0, 10.0]])
    assert dtw_distance(a, b) > 0


def test_dtw_distance_infinite_for_empty_sequence():
    a = np.zeros((0, 3))
    b = np.array([[1.0, 2.0, 3.0]])
    assert dtw_distance(a, b) == float("inf")
