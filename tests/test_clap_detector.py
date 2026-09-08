"""ClapDetector needs no audio hardware or trained model to test: a clap
is just a short burst of broadband noise, easy to synthesize deterministically.

Timing is driven by the detector's own internal clock (advanced by the
duration of audio fed to it), not wall-clock time, so these tests build
exact silence/clap/silence sequences and assert exactly when detection
fires.
"""

from __future__ import annotations

import numpy as np

from neo.voice.clap_detector import ClapDetector, ClapDetectorConfig

SR = 16000


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)


def _clap(seconds: float = 0.02, amplitude: float = 0.8, seed: int = 0) -> np.ndarray:
    """White noise with a fast attack/decay envelope -- broadband, brief,
    exactly what a real clap looks like on a waveform."""
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    noise = rng.uniform(-1.0, 1.0, n).astype(np.float32)
    env = np.hanning(n).astype(np.float32)
    return noise * env * amplitude


def _tone(seconds: float, freq: float = 220.0, amplitude: float = 0.8) -> np.ndarray:
    """A sustained low-frequency tone -- loud, but neither broadband nor
    a brief onset-only transient, the two things separating a clap from a
    voice or a hum."""
    n = int(SR * seconds)
    t = np.arange(n, dtype=np.float32) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _feed_all(detector: ClapDetector, *segments: np.ndarray) -> bool:
    fired = False
    for seg in segments:
        if detector.feed(seg):
            fired = True
    return fired


def test_two_well_timed_claps_are_detected():
    d = ClapDetector()
    fired = _feed_all(
        d,
        _silence(0.3), _clap(seed=1), _silence(0.3), _clap(seed=2), _silence(0.3),
    )
    assert fired is True


def test_a_single_clap_alone_is_not_detected():
    d = ClapDetector()
    fired = _feed_all(d, _silence(0.3), _clap(seed=1), _silence(1.0))
    assert fired is False


def test_claps_too_far_apart_are_not_a_gesture():
    d = ClapDetector(ClapDetectorConfig(max_gap_seconds=0.7))
    fired = _feed_all(
        d,
        _silence(0.3), _clap(seed=1), _silence(2.0), _clap(seed=2), _silence(0.3),
    )
    assert fired is False


def test_claps_too_close_together_are_not_a_gesture():
    """Faster than a human can actually clap twice -- more likely one clap's
    own onset plus an echo, or two unrelated bangs a beat apart."""
    d = ClapDetector(ClapDetectorConfig(min_gap_seconds=0.12, refractory_seconds=0.01))
    fired = _feed_all(
        d,
        _silence(0.3), _clap(seed=1), _silence(0.03), _clap(seed=2), _silence(0.3),
    )
    assert fired is False


def test_sustained_loud_noise_does_not_self_trigger():
    """The failure mode this guards: without onset-only detection, several
    seconds of ordinary loud sound (music, raised voices) produces a new
    'candidate' roughly every refractory period, and eventually two of them
    land inside the double-clap gap purely by chance."""
    d = ClapDetector()
    rng = np.random.default_rng(7)
    steady_noise = (rng.uniform(-1.0, 1.0, SR * 3) * 0.9).astype(np.float32)
    fired = _feed_all(d, _silence(0.2), steady_noise)
    assert fired is False


def test_a_sustained_tone_is_not_mistaken_for_a_clap():
    """Loud and has a sharp start, but is neither broadband nor brief --
    the two things that separate a clap from someone speaking loudly or a
    hum starting up."""
    d = ClapDetector()
    fired = _feed_all(d, _silence(0.2), _tone(1.5), _silence(0.2))
    assert fired is False


def test_quiet_room_noise_never_counts_as_a_candidate():
    d = ClapDetector()
    rng = np.random.default_rng(3)
    hiss = (rng.uniform(-1.0, 1.0, SR * 2) * 0.005).astype(np.float32)  # below absolute_floor
    fired = _feed_all(d, hiss)
    assert fired is False


def test_a_gesture_is_consumed_and_a_third_clap_starts_fresh():
    """After a double clap fires, the very next clap shouldn't immediately
    pair with the second clap of the gesture that just fired."""
    d = ClapDetector()
    first_pair = _feed_all(
        d, _silence(0.3), _clap(seed=1), _silence(0.3), _clap(seed=2), _silence(0.05)
    )
    assert first_pair is True

    # A lone third clap, shortly after, must not immediately re-fire.
    immediate = d.feed(_clap(seed=3))
    assert immediate is False
