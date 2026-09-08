from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .audio_features import (
    dtw_distance,
    extract_mfcc,
    extract_speech_segment,
    normalize_features,
    speech_span,
    trim_silence,
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MIN_TEMPLATES = 3

# Bumped whenever the stored feature layout changes (frame count, delta
# features, endpointing). Templates from an older version describe audio in a
# shape the current matcher can't compare against, so they are dropped rather
# than half-working.
FEATURE_VERSION = 2
_MIN_WORD_SECONDS = 0.15
_MIN_ENROLL_PEAK = 0.02

# Fallback used only for legacy template files that carry no negative
# samples. Calibrating purely from how much the user's own repetitions of the
# wake word differ answers "how much does my voice vary?" and never "how far
# away is everything else I say?", which is why a threshold derived this way
# let ordinary short words through.
_POSITIVE_ONLY_MARGIN = 1.5

# How far below the nearest non-wake-word sample the threshold is placed when
# the two groups separate cleanly. Sitting at the midpoint would split the
# gap evenly; biasing toward the wake word costs an occasional repeat instead
# of an assistant that wakes up mid-conversation.
_SEPARATION_BIAS = 0.35

# The threshold must clear the enrolled spread by a real margin, even when
# the negatives sit right on top of it.
#
# `spread` is the widest distance between any two enrolled takes, and with
# MIN_TEMPLATES = 3 that is the widest of three pairs -- a small sample of
# how much a voice varies, and one that systematically underestimates it.
# Setting the threshold at or barely above that number means the fourth
# time the user says the phrase, ordinary variation puts it outside, and
# the wake word is rejected here without ever reaching confirmation. Live
# calibration on this user read spread 29.6, nearest negative 30.8 -- a
# separation of 1.2, about 4% -- so the threshold landed at 30.0, right on
# the spread, and genuine attempts failed roughly half the time with
# nothing logged.
#
# When the two groups overlap this much the acoustic matcher has already
# shown it cannot tell them apart (see this class's docstring), so it is
# not the thing being protected by a tight threshold. Its remaining job is
# to keep recognition from running on silence, and the words themselves are
# confirmed afterwards by the recognizer, which separates them cleanly.
_MIN_SPREAD_HEADROOM = 1.25

# Below this, the separation between the two groups is a rounding error
# rather than evidence, and the widened threshold above applies. Above it,
# the negatives really do stand apart and the tight threshold is kept --
# loosening a calibration that works would only invite false wake-ups.
_WEAK_SEPARATION_RATIO = 0.25

# A wake word is one short word said on its own. These bound how far the
# candidate's spoken length may drift from the enrolled recordings before it
# is rejected regardless of how well DTW scores it -- duration is exactly
# what `normalize_features` deliberately throws away, so it has to be
# checked separately.
_MIN_DURATION_RATIO = 0.45
_MAX_DURATION_RATIO = 2.2

# The detection window must hold the whole wake phrase and still leave
# silence around it -- both so the phrase is never clipped mid-word and so
# the "is this an isolated phrase or the middle of a sentence?" check has
# room to tell the difference. Scales with whatever the user enrolled, so a
# two-word phrase like "Neo uyan" gets a wider window than a bare "Neo"
# without anything being hardcoded to a particular wake word.
_WINDOW_HEADROOM = 1.8
_MIN_WINDOW_SECONDS = 1.4

# A wake phrase is one short utterance, so VAD's default minimum segment
# length would sit right at the edge of discarding it.
_WAKE_MIN_SPEECH_MS = 100

# An isolated phrase leaves silence around it inside the detection window;
# continuous speech fills essentially all of it. Set high enough that saying
# the phrase and running straight into the command still wakes NEO -- that
# window still opens with silence, because detection fires on the earliest
# matching window, not a later one mid-sentence.
_MAX_SPEECH_FRACTION = 0.90


class KeywordSpotter:
    """Personalized wake-word detector: instead of guessing from generic
    speech-to-text (which kept mishearing "Neo" as unrelated Turkish words),
    the user enrolls a few recordings of themselves saying the wake word and
    later audio is matched against those recordings directly (MFCC + DTW
    acoustic similarity), fully offline.

    Enrollment records two things: several takes of the wake word, and a
    stretch of ordinary speech that is *not* the wake word. The match
    threshold is then placed in the gap between "how close my own repetitions
    of 'Neo' are to each other" and "how close my ordinary speech gets" --
    with only the first of those, as an earlier version had, the threshold
    was a guess with no evidence about what it needed to exclude, and
    ordinary conversation kept scoring under it.
    """

    def __init__(self, templates_path: Path, sample_rate: int = SAMPLE_RATE) -> None:
        self._path = templates_path
        self._sample_rate = sample_rate
        self._templates: list[np.ndarray] = []
        self._durations: list[float] = []
        self._negatives: list[np.ndarray] = []
        self._threshold = float("inf")
        self._separation: float | None = None
        self._spread: float | None = None
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = np.load(self._path, allow_pickle=True)
            keys = set(data.files)
            stored_version = int(data["version"][0]) if "version" in keys else 0
            if stored_version != FEATURE_VERSION:
                # The feature layout changed (frame count, delta features), so
                # old templates can't be compared against new audio at all.
                # Discard them and say so, rather than loading vectors of the
                # wrong shape and failing at match time.
                logger.info(
                    "Uyandırma şablonları eski biçimde (v%s), yeniden öğretmen gerekiyor",
                    stored_version,
                )
                self._templates = []
                self._durations = []
                self._negatives = []
                self._recalibrate()
                return
            if any(key.startswith("pos_") for key in keys):
                self._templates = [data[k] for k in sorted(keys) if k.startswith("pos_")]
                self._negatives = [data[k] for k in sorted(keys) if k.startswith("neg_")]
                self._durations = (
                    [float(x) for x in data["durations"]] if "durations" in keys else []
                )
            else:
                # Legacy layout (t0/t1/...): wake-word takes only, no
                # durations and nothing to calibrate against. Still usable,
                # but the duration and negative checks stay off until the
                # user re-enrolls.
                self._templates = [data[k] for k in sorted(keys)]
                self._negatives = []
                self._durations = []
                logger.info(
                    "Eski uyandırma şablonları yüklendi: olumsuz örnek yok, "
                    "yeniden öğretmek doğruluğu artırır."
                )
        except Exception:
            logger.exception("Wake-word templates could not be loaded")
            self._templates = []
            self._durations = []
            self._negatives = []
        self._recalibrate()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            f"pos_{i}": t for i, t in enumerate(self._templates)
        }
        payload.update({f"neg_{i}": t for i, t in enumerate(self._negatives)})
        payload["durations"] = np.asarray(self._durations, dtype=np.float64)
        payload["version"] = np.asarray([FEATURE_VERSION], dtype=np.int64)
        np.savez(self._path, **payload)

    # -- calibration -------------------------------------------------------

    def _recalibrate(self) -> None:
        if len(self._templates) < 2:
            self._threshold = float("inf")
            self._separation = None
            self._spread = None
            return

        spread = max(
            dtw_distance(self._templates[i], self._templates[j])
            for i in range(len(self._templates))
            for j in range(i + 1, len(self._templates))
        )

        self._spread = spread
        if not self._negatives:
            self._threshold = spread * _POSITIVE_ONLY_MARGIN
            self._separation = None
            return

        # Closest any non-wake-word sample gets to the enrolled word. The
        # threshold has to sit below this or ordinary speech matches.
        nearest_negative = min(
            dtw_distance(negative, template)
            for negative in self._negatives
            for template in self._templates
        )

        self._separation = nearest_negative - spread
        if nearest_negative > spread:
            tight = spread + (nearest_negative - spread) * _SEPARATION_BIAS
            if self._separation < spread * _WEAK_SEPARATION_RATIO:
                # The groups very nearly overlap, so this threshold is not
                # separating anything -- but sitting on the spread does
                # reject the user's own next attempt. Give genuine takes
                # room and leave the verdict to confirmation.
                self._threshold = max(tight, spread * _MIN_SPREAD_HEADROOM)
            else:
                # The negatives genuinely stand apart, so the acoustic gate
                # is worth something here. Keep it tight.
                self._threshold = tight
        else:
            # The two groups overlap: some ordinary speech resembles the wake
            # word more than the user's own takes resemble each other. No
            # threshold separates them, so sit just under the nearest
            # negative -- fewer false wake-ups, at the cost of missed ones --
            # and let the caller warn that re-enrolling would help.
            self._threshold = nearest_negative * 0.9
            logger.warning(
                "Uyandırma kalibrasyonu zayıf: örnekler ayrışmıyor "
                "(en yakın olumsuz=%.2f, örnek yayılımı=%.2f)",
                nearest_negative,
                spread,
            )

    # -- state -------------------------------------------------------------

    @property
    def template_count(self) -> int:
        return len(self._templates)

    @property
    def negative_count(self) -> int:
        return len(self._negatives)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def separation(self) -> float | None:
        """Gap between the nearest non-wake-word sample and the spread of the
        enrolled takes. Positive means the two groups separate cleanly;
        None when there is nothing to compare against."""
        return self._separation

    @property
    def is_calibrated(self) -> bool:
        """True when the threshold rests on actual negative evidence rather
        than on a margin picked out of the air.

        A separation merely greater than zero is not that evidence. This
        user's enrollment measured 1.2 against a spread of 29.5 -- the two
        groups were touching -- and the old test still called it calibrated,
        so nothing ever suggested re-recording while the wake word kept
        failing. It has to be a real fraction of the spread to count.
        """
        if not self._negatives or self._separation is None or self._spread is None:
            return False
        return self._separation >= self._spread * _WEAK_SEPARATION_RATIO

    @property
    def enrolled_duration(self) -> float | None:
        """Longest enrolled take, in seconds. None for legacy template files
        recorded before durations were stored."""
        return max(self._durations) if self._durations else None

    @property
    def detection_window_seconds(self) -> float:
        """How much audio the detector should compare at once, sized to the
        phrase the user actually enrolled."""
        enrolled = self.enrolled_duration
        if enrolled is None:
            return _MIN_WINDOW_SECONDS
        return max(_MIN_WINDOW_SECONDS, enrolled * _WINDOW_HEADROOM)

    def has_enough_templates(self, minimum: int = MIN_TEMPLATES) -> bool:
        return len(self._templates) >= minimum

    # -- features ----------------------------------------------------------

    def _features(self, audio: np.ndarray) -> np.ndarray:
        # Must endpoint the same way `enroll` does, or the fixed-frame
        # features describe differently-bounded audio and the distances stop
        # being comparable.
        spoken = extract_speech_segment(audio, self._sample_rate)
        return normalize_features(extract_mfcc(spoken, self._sample_rate))

    # -- enrollment --------------------------------------------------------

    def enroll(self, audio: np.ndarray) -> bool:
        """Adds `audio` as a new wake-word template. Returns False (and
        enrolls nothing) if barely any sound was detected -- e.g. the user's
        timing was off during the guided recording -- so the caller can ask
        for a retake instead of silently poisoning the calibration with a
        bad, near-silent template."""
        # Templates loaded from a pre-durations file can't be reconciled with
        # new ones -- and mixing takes of two different phrases wrecks the
        # calibration outright (a real case: three "Neo" takes left over from
        # before plus three "Neo uyan" takes pushed the spread to 50.4 and
        # left the threshold accepting everything). Start clean instead.
        if len(self._durations) != len(self._templates):
            logger.info("Önceki uyandırma şablonları temizlendi (süre bilgisi yok)")
            self._templates = []
            self._durations = []
            self._negatives = []

        spoken = extract_speech_segment(audio, self._sample_rate)
        peak = float(np.abs(spoken).max()) if spoken.size else 0.0
        logger.info(
            "enroll: raw=%d samples, spoken=%d samples (%.2fs), peak=%.4f",
            audio.size, spoken.size, spoken.size / float(self._sample_rate), peak,
        )
        if spoken.size < int(_MIN_WORD_SECONDS * self._sample_rate) or peak < _MIN_ENROLL_PEAK:
            logger.warning(
                "enroll reddedildi: spoken_samples=%d (min=%d), peak=%.4f (min=%.4f)",
                spoken.size, int(_MIN_WORD_SECONDS * self._sample_rate), peak, _MIN_ENROLL_PEAK,
            )
            return False
        self._templates.append(normalize_features(extract_mfcc(spoken, self._sample_rate)))
        self._durations.append(spoken.size / float(self._sample_rate))
        self._recalibrate()
        self._save()
        return True

    def enroll_negative(self, audio: np.ndarray, window_seconds: float | None = None) -> int:
        """Adds ordinary (non-wake-word) speech as calibration evidence.

        The recording is cut into wake-word-sized windows because that is the
        shape the detector actually sees at runtime -- a single long sample
        would be compared as one unit and tell us nothing about how a passing
        word scores. Returns how many windows were kept.
        """
        if window_seconds is None:
            window_seconds = self.detection_window_seconds
        window = int(window_seconds * self._sample_rate)
        step = max(1, window // 2)
        added = 0
        for start in range(0, max(0, audio.size - window) + 1, step):
            chunk = audio[start : start + window]
            spoken = extract_speech_segment(chunk, self._sample_rate)
            peak = float(np.abs(spoken).max()) if spoken.size else 0.0
            if spoken.size < int(_MIN_WORD_SECONDS * self._sample_rate) or peak < _MIN_ENROLL_PEAK:
                continue
            self._negatives.append(normalize_features(extract_mfcc(spoken, self._sample_rate)))
            added += 1
        if added:
            self._recalibrate()
            self._save()
        return added

    def clear(self) -> None:
        self._templates = []
        self._durations = []
        self._negatives = []
        self._threshold = float("inf")
        self._separation = None
        if self._path.exists():
            self._path.unlink()

    # -- matching ----------------------------------------------------------

    def score(self, audio: np.ndarray) -> float:
        """Lower means more similar to the enrolled wake word. Returns +inf
        when there are no templates to compare against."""
        if not self._templates:
            return float("inf")
        features = self._features(audio)
        if features.shape[0] == 0:
            return float("inf")
        return min(dtw_distance(features, template) for template in self._templates)

    def _duration_plausible(self, duration: float) -> bool:
        """Whether the candidate lasts roughly as long as the enrolled phrase.

        DTW sees time-normalized features, so a whole spoken sentence can
        score as well as a single word. Length is the signal that survives
        that normalization, and it is what separates "Neo uyan" from "Neo'yu
        çalıştırır mısın" landing in the same detection window.
        """
        if not self._durations:
            return True
        return (
            min(self._durations) * _MIN_DURATION_RATIO
            <= duration
            <= max(self._durations) * _MAX_DURATION_RATIO
        )

    def is_match(self, audio: np.ndarray) -> bool:
        """All the acoustic gating lives here rather than in the listening
        loop, because every check needs the enrolled recordings as reference
        and they all read the same single VAD pass.

        Every rejection says which of the five gates stopped it. Without
        that, a wake word that "just doesn't work" gave the logs nothing at
        all to look at -- the loop simply went quiet -- and diagnosing it
        was guesswork. The lines are one per candidate window, and only
        windows loud enough to reach here produce any.
        """
        if not self.has_enough_templates() or audio.size == 0:
            return False

        span = speech_span(audio, self._sample_rate, _WAKE_MIN_SPEECH_MS)
        if span is None:
            # VAD unavailable: fall back to amplitude endpointing rather than
            # refusing to wake up at all.
            spoken, detected = trim_silence(audio, self._sample_rate), None
        else:
            spoken, detected = span

        if spoken.size == 0:
            logger.info("Uyandırma elendi [kapı=sessiz: konuşma bulunamadı]")
            return False

        window_seconds = audio.size / self._sample_rate
        spoken_seconds = spoken.size / float(self._sample_rate)

        if detected is not None:
            if detected == 0.0:
                # DTW compares shapes, so a door bump or key press can land
                # close enough to a template to "match".
                logger.info("Uyandırma elendi [kapı=VAD: konuşma yok, gürültü]")
                return False
            # A wake phrase is said on its own. A window filled wall to wall
            # with speech is the middle of a sentence, not someone calling.
            fraction = detected / window_seconds
            if fraction > _MAX_SPEECH_FRACTION:
                logger.info(
                    "Uyandırma elendi [kapı=süreklilik: pencerenin %%%.0f'i konuşma "
                    "(üst sınır %%%.0f) -- cümle ortası sayıldı]",
                    fraction * 100, _MAX_SPEECH_FRACTION * 100,
                )
                return False

        if not self._duration_plausible(spoken_seconds):
            logger.info(
                "Uyandırma elendi [kapı=süre: %.2f sn, kabul aralığı %.2f-%.2f sn "
                "(öğretilen: %s)]",
                spoken_seconds,
                min(self._durations) * _MIN_DURATION_RATIO if self._durations else 0.0,
                max(self._durations) * _MAX_DURATION_RATIO if self._durations else 0.0,
                ", ".join(f"{d:.2f}" for d in self._durations) or "yok",
            )
            return False

        features = normalize_features(extract_mfcc(spoken, self._sample_rate))
        if features.shape[0] == 0:
            logger.info("Uyandırma elendi [kapı=öznitelik: MFCC çıkarılamadı]")
            return False
        distance = min(dtw_distance(features, template) for template in self._templates)
        if distance > self._threshold:
            logger.info(
                "Uyandırma elendi [kapı=DTW: uzaklık %.1f > eşik %.1f, süre %.2f sn]",
                distance, self._threshold, spoken_seconds,
            )
            return False

        logger.info(
            "Uyandırma akustik eşleşme: uzaklık %.1f <= eşik %.1f (süre %.2f sn), "
            "doğrulamaya gidiyor",
            distance, self._threshold, spoken_seconds,
        )
        return True
