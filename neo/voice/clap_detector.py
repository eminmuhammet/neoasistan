from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClapDetectorConfig:
    sample_rate: int = 16000
    # Short enough to catch a clap's brief transient without smearing it
    # across a whole ~300ms microphone chunk; long enough for an FFT bin at
    # high_freq_cutoff_hz to be meaningful.
    frame_seconds: float = 0.02
    background_seconds: float = 0.5
    # A frame must be this many times louder than the recent ambient level
    # to be a candidate at all.
    energy_ratio: float = 7.0
    # RMS floor so a perfectly quiet room's own noise floor -- which is
    # loud "relative to itself" -- never counts as a candidate.
    absolute_floor: float = 0.015
    # Minimum fraction of a frame's spectral energy that must sit above
    # high_freq_cutoff_hz. A clap is a broadband, near-white-noise burst;
    # most speech and low-frequency bumps (a footstep, a door) are not,
    # which is what keeps a raised voice or a dropped book from registering
    # as a clap on its own.
    high_freq_ratio: float = 0.30
    high_freq_cutoff_hz: float = 2000.0
    # Guards against the same clap's decay tail flickering back above
    # threshold and being counted as a second, separate transient. The
    # onset-only detection below (see _process_frame) is the primary
    # defense; this is a small extra margin.
    refractory_seconds: float = 0.10
    # The two claps of a deliberate double-clap gesture, timed by hand,
    # land in this range in practice -- fast enough to be one gesture,
    # slow enough that a single clap's own ringing never qualifies.
    min_gap_seconds: float = 0.12
    max_gap_seconds: float = 0.70
    candidate_memory_seconds: float = 1.0


class ClapDetector:
    """Detects a deliberate double clap: two short, sharp, broadband sound
    transients within a fraction of a second of each other -- the same
    gesture "clap on/clap off" switches have used for decades, chosen here
    because it needs no enrollment, no training, and works in any language,
    unlike a spoken wake word.

    Feed it audio in whatever chunk size the microphone delivers; it slices
    each chunk into its own short analysis frames internally and keeps a
    running internal clock advanced by the audio duration fed to it (not
    wall-clock time), which is what makes it deterministic to test.
    """

    def __init__(self, config: ClapDetectorConfig | None = None) -> None:
        self._config = config or ClapDetectorConfig()
        self._frame_samples = max(1, int(self._config.sample_rate * self._config.frame_seconds))
        self._background_energy = self._config.absolute_floor
        self._background_alpha = self._frame_samples / (
            self._config.sample_rate * self._config.background_seconds
        )
        self._was_loud = False
        self._refractory_until = 0.0
        self._candidate_times: list[float] = []
        self._clock = 0.0
        self._window = np.hanning(self._frame_samples).astype(np.float32)
        freqs = np.fft.rfftfreq(self._frame_samples, d=1.0 / self._config.sample_rate)
        self._high_freq_mask = freqs >= self._config.high_freq_cutoff_hz

    def feed(self, chunk: np.ndarray) -> bool:
        """Feed a chunk of mono float32 audio (-1..1) at the configured
        sample rate. Returns True the instant a double clap completes."""
        fired = False
        n = self._frame_samples
        for start in range(0, len(chunk) - n + 1, n):
            frame = chunk[start:start + n]
            self._clock += n / self._config.sample_rate
            if self._process_frame(frame):
                fired = True
        return fired

    def _process_frame(self, frame: np.ndarray) -> bool:
        energy = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))

        is_loud_now = (
            energy >= self._config.absolute_floor
            and energy >= self._background_energy * self._config.energy_ratio
        )
        # Onset only: a rising edge from quiet to loud, not every frame for
        # as long as it stays loud. Without this, a few seconds of ordinary
        # loud noise (music, raised voices) would produce a new "candidate"
        # roughly every refractory period and eventually two of them would
        # land inside the double-clap gap purely by chance.
        is_onset = is_loud_now and not self._was_loud
        self._was_loud = is_loud_now

        # Background tracks the ambient level, but only while nothing loud
        # is happening -- letting a clap itself pull the average up would
        # make the second clap of the same gesture harder to detect than
        # the first.
        if not is_loud_now:
            self._background_energy += self._background_alpha * (energy - self._background_energy)

        candidate = (
            is_onset
            and self._clock >= self._refractory_until
            and self._is_broadband(frame)
        )
        if not candidate:
            return False

        self._refractory_until = self._clock + self._config.refractory_seconds
        self._candidate_times = [
            t for t in self._candidate_times
            if self._clock - t <= self._config.candidate_memory_seconds
        ]

        fired = any(
            self._config.min_gap_seconds <= (self._clock - t) <= self._config.max_gap_seconds
            for t in self._candidate_times
        )
        if fired:
            self._candidate_times = []  # gesture consumed; start clean
        else:
            self._candidate_times.append(self._clock)
        return fired

    def _is_broadband(self, frame: np.ndarray) -> bool:
        spectrum = np.abs(np.fft.rfft(frame * self._window))
        power = spectrum ** 2
        total = power.sum()
        if total <= 0:
            return False
        return (power[self._high_freq_mask].sum() / total) >= self._config.high_freq_ratio
