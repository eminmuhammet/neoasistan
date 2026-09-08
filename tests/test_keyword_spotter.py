import numpy as np
import pytest

from neo.voice import keyword_spotter as ks
from neo.voice.audio_features import trim_silence
from neo.voice.keyword_spotter import KeywordSpotter


@pytest.fixture(autouse=True)
def amplitude_endpointing(monkeypatch):
    """Endpoint on amplitude instead of running the real VAD.

    These tests exercise the spotter's decision logic -- threshold
    calibration, the duration gate, isolation, persistence -- on synthetic
    tones. Silero VAD reasonably refuses to call those speech, so leaving it
    in the loop would make every assertion depend on its judgement of
    artificial audio rather than on the logic under test. The real detector
    is covered against unambiguous inputs in test_audio_features.py.
    """

    def fake_span(audio, sample_rate=16000, min_speech_ms=100):
        segment = trim_silence(audio, sample_rate)
        return segment, segment.size / float(sample_rate)

    monkeypatch.setattr(ks, "speech_span", fake_span)
    monkeypatch.setattr(ks, "extract_speech_segment", lambda a, sr=16000: trim_silence(a, sr))


def _tone(freq: float, seconds: float, sample_rate: int = 16000, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _isolated_tone(freq: float, seconds: float = 0.6, total: float = 1.8) -> np.ndarray:
    """A tone with silence around it -- the shape of a wake word actually
    said on its own.

    A buffer filled wall to wall is deliberately *not* a match: that is what
    the middle of a sentence looks like, and treating it as a candidate is
    what had NEO waking up during ordinary conversation.
    """
    sample_rate = 16000
    buffer = np.zeros(int(total * sample_rate), dtype=np.float32)
    word = _tone(freq, seconds, sample_rate)
    start = int(0.5 * sample_rate)
    buffer[start : start + len(word)] = word
    return buffer


def test_no_match_without_enough_templates(tmp_path):
    spotter = KeywordSpotter(tmp_path / "templates.npz")
    assert spotter.is_match(_isolated_tone(440)) is False


def test_enrolled_word_matches_itself(tmp_path):
    spotter = KeywordSpotter(tmp_path / "templates.npz")
    for _ in range(3):
        assert spotter.enroll(_isolated_tone(440)) is True

    assert spotter.has_enough_templates()
    assert spotter.is_match(_isolated_tone(440)) is True


def test_enroll_rejects_near_silent_recording(tmp_path):
    spotter = KeywordSpotter(tmp_path / "templates.npz")
    silence = np.zeros(16000, dtype=np.float32)

    assert spotter.enroll(silence) is False
    assert spotter.template_count == 0


def test_different_sound_does_not_match(tmp_path):
    spotter = KeywordSpotter(tmp_path / "templates.npz")
    for _ in range(3):
        spotter.enroll(_isolated_tone(440))

    assert spotter.is_match(_isolated_tone(1200)) is False


def test_templates_persist_across_instances(tmp_path):
    path = tmp_path / "templates.npz"
    spotter = KeywordSpotter(path)
    for _ in range(3):
        spotter.enroll(_isolated_tone(440))

    reloaded = KeywordSpotter(path)
    assert reloaded.template_count == 3
    assert reloaded.is_match(_isolated_tone(440)) is True


def test_clear_removes_templates_and_file(tmp_path):
    path = tmp_path / "templates.npz"
    spotter = KeywordSpotter(path)
    spotter.enroll(_isolated_tone(440))
    spotter.clear()

    assert spotter.template_count == 0
    assert not path.exists()


# -- negative calibration --------------------------------------------------


def _word(formants, seconds, jitter=0.0, sample_rate=16000):
    """Crude vowel-sequence synthesizer -- a pitch-varying harmonic source
    shaped by a moving formant -- so two 'words' differ in MFCC trajectory
    the way real ones do, rather than only in pitch."""
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    f0 = 125 * (1 + 0.12 * np.sin(2 * np.pi * 1.3 * t)) * (1 + jitter)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    sig = sum(np.sin(k * phase) / k for k in range(1, 9))

    env = np.zeros(n)
    for idx, f in zip(np.array_split(np.arange(n), len(formants)), formants):
        env[idx] = f
    sig = sig * (0.6 + 0.4 * np.sin(2 * np.pi * np.cumsum(env) / sample_rate))
    sig *= np.hanning(n)
    return (sig / np.abs(sig).max() * 0.3).astype(np.float32)


def _padded(word, total=2.2, offset=0.6, sample_rate=16000):
    buf = np.zeros(int(total * sample_rate), dtype=np.float32)
    buf[int(offset * sample_rate) : int(offset * sample_rate) + len(word)] = word
    return buf


WAKE_TAKES = [
    _padded(_word([600, 900, 500, 1400], 1.00, jitter=0.00)),
    _padded(_word([610, 880, 510, 1380], 1.06, jitter=0.03)),
    _padded(_word([590, 920, 495, 1420], 0.95, jitter=-0.03)),
]

OTHER_WORDS = [
    _word([500, 1600, 700, 1100], 0.90),
    _word([700, 1200, 600, 1500], 0.85),
    _word([800, 1000], 0.55),
    _word([450, 1300, 850, 900], 1.05),
]


# Words far enough from the wake takes that the two groups genuinely
# separate. OTHER_WORDS deliberately sit close, which is the realistic case
# and the one the weak-separation tests below use.
FAR_WORDS = [
    _word([1800, 300], 0.50),
    _word([250, 2200, 400], 0.70),
    _word([2000, 2400, 1900], 0.60),
]

# Takes recorded consistently, so the spread stays small and FAR_WORDS end
# up a long way outside it.
CONSISTENT_TAKES = [
    _padded(_word([600, 900, 500, 1400], 1.00, jitter=j)) for j in (0.0, 0.005, -0.005)
]


def _enrolled(path, with_negatives, takes=None, negatives=None):
    spotter = KeywordSpotter(path)
    for take in takes or WAKE_TAKES:
        assert spotter.enroll(take) is True
    if with_negatives:
        spotter.enroll_negative(
            np.concatenate(
                [_padded(w, total=1.8, offset=0.3) for w in (negatives or OTHER_WORDS)]
            )
        )
    return spotter


def _well_separated(path):
    return _enrolled(path, True, takes=CONSISTENT_TAKES, negatives=FAR_WORDS)


def test_negative_calibration_tightens_the_threshold(tmp_path):
    """The bug this fixes: a threshold derived only from how much the user's
    own takes vary has no evidence about what it must exclude, so ordinary
    speech scored under it and NEO woke up mid-conversation."""
    loose = _enrolled(tmp_path / "a.npz", with_negatives=False)
    tight = _enrolled(tmp_path / "b.npz", with_negatives=True)

    assert tight.threshold < loose.threshold
    assert tight.negative_count > 0
    assert loose.is_calibrated is False


def test_calibration_is_only_claimed_when_the_groups_really_separate(tmp_path):
    """`is_calibrated` drives the "your enrollment is weak, re-record"
    warning, so it has to mean the evidence is real.

    Any separation above zero used to count. Live enrollment measured a
    separation of 1.2 against a spread of 29.5 -- the wake takes and
    ordinary speech were touching -- and it still reported calibrated, so
    nothing ever suggested re-recording while the wake word kept failing.
    """
    weak = _enrolled(tmp_path / "weak.npz", with_negatives=True)
    strong = _well_separated(tmp_path / "strong.npz")

    assert weak.is_calibrated is False
    assert strong.is_calibrated is True


def test_a_well_separated_spotter_rejects_other_words(tmp_path):
    """When the negatives genuinely stand apart, the acoustic gate is worth
    something and stays tight."""
    spotter = _well_separated(tmp_path / "t.npz")
    for word in FAR_WORDS:
        assert spotter.is_match(_padded(word)) is False


def test_a_weak_calibration_keeps_room_for_the_users_own_takes(tmp_path):
    """The threshold has to clear the enrolled spread even when the negatives
    sit right on top of it.

    `spread` is the widest of only three pairs, which underestimates how
    much a voice varies. Placing the threshold on it meant the fourth time
    the user said the phrase, ordinary variation fell outside and the wake
    word was rejected here -- before confirmation ever ran, and with nothing
    written to the log. Observed live as "it never wakes on the first try".

    Lookalikes getting through this gate is the accepted cost: a spotter
    this poorly separated was never rejecting them reliably anyway, and the
    recognizer downstream separates the words cleanly.
    """
    spotter = _enrolled(tmp_path / "t.npz", with_negatives=True)

    assert spotter.is_calibrated is False
    assert spotter.threshold > spotter._spread


def test_calibration_keeps_matching_the_real_wake_phrase(tmp_path):
    """Tightening must not cost recall -- including on a fresh utterance that
    was never enrolled."""
    spotter = _enrolled(tmp_path / "t.npz", with_negatives=True)
    for take in WAKE_TAKES:
        assert spotter.is_match(take) is True
    assert spotter.is_match(_padded(_word([605, 895, 505, 1390], 1.02, jitter=0.015))) is True


def test_sentence_length_audio_is_rejected_on_duration(tmp_path):
    """DTW compares time-normalized features, so a whole sentence can score
    like a single word. Length is the signal that survives normalization."""
    spotter = _enrolled(tmp_path / "t.npz", with_negatives=True)
    sentence = _word([650, 1150, 520, 1350, 700, 1000, 600, 1200], 3.4)
    assert spotter.is_match(_padded(sentence, total=4.6, offset=0.3)) is False


def test_detection_window_grows_with_a_longer_wake_phrase(tmp_path):
    short = KeywordSpotter(tmp_path / "short.npz")
    for _ in range(3):
        short.enroll(_padded(_word([600, 900], 0.35)))

    long = _enrolled(tmp_path / "long.npz", with_negatives=True)

    assert long.detection_window_seconds > short.detection_window_seconds


def test_negatives_and_durations_persist_across_instances(tmp_path):
    path = tmp_path / "t.npz"
    spotter = _enrolled(path, with_negatives=True)
    expected = spotter.threshold

    reloaded = KeywordSpotter(path)
    assert reloaded.negative_count == spotter.negative_count
    assert reloaded.is_calibrated == spotter.is_calibrated
    assert reloaded.threshold == expected


def test_enrolling_over_previous_takes_starts_clean(tmp_path):
    """The bug this guards: enrolling a new phrase on top of templates that
    carried no durations kept both sets. Three "Neo" takes sitting alongside
    three "Neo uyan" takes pushed the spread to 50.4, which left a threshold
    loose enough to accept ordinary speech -- observed live as six enrolled
    samples and a separation of -24.2."""
    path = tmp_path / "t.npz"
    spotter = KeywordSpotter(path)
    for _ in range(3):
        spotter.enroll(_isolated_tone(440))

    # simulate the durations/templates desync that made the mixing possible
    spotter._durations = []
    spotter.enroll(_isolated_tone(1200))

    assert spotter.template_count == 1
    assert len(spotter._durations) == spotter.template_count


def test_durations_stay_aligned_with_templates(tmp_path):
    spotter = _enrolled(tmp_path / "t.npz", with_negatives=True)
    assert len(spotter._durations) == spotter.template_count


def test_templates_from_an_older_feature_version_are_discarded(tmp_path):
    """Endpointing changed, so old templates describe a different slice of
    audio than new recordings would -- comparing the two is meaningless.
    Dropping them forces one re-enrollment; keeping them would silently
    degrade matching with no signal to the user that anything was wrong."""
    path = tmp_path / "legacy.npz"
    legacy = KeywordSpotter(path)
    for take in WAKE_TAKES:
        legacy.enroll(take)

    # rewrite in the old layout: features only, no version stamp
    data = np.load(path, allow_pickle=True)
    np.savez(path, **{f"t{i}": data[f"pos_{i}"] for i in range(3)})

    reloaded = KeywordSpotter(path)
    assert reloaded.template_count == 0
    assert reloaded.has_enough_templates() is False
    assert reloaded.is_match(WAKE_TAKES[0]) is False


def test_current_version_templates_survive_a_reload(tmp_path):
    path = tmp_path / "t.npz"
    spotter = KeywordSpotter(path)
    for take in WAKE_TAKES:
        spotter.enroll(take)

    reloaded = KeywordSpotter(path)
    assert reloaded.template_count == 3
