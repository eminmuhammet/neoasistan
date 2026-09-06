from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def has_enough_speech(buffer: np.ndarray, energy_threshold: float, min_voiced_samples: int) -> bool:
    """Whisper reliably hallucinates fluent-sounding (fabricated) text --
    often YouTube-subtitle phrases like "abone olmayı unutmayın" -- when fed
    audio that's mostly silence/noise, and those hallucinations often carry
    high confidence, so logprob/no-speech thresholds inside Whisper don't
    reliably catch them. Checking there was actually enough loud content
    *before* transcribing at all is a more reliable guard, so this is used
    ahead of every STT call, not just the continuous wake-word path.

    This only measures *loudness*, though, which a single transient (a key
    press, a door) passes just as easily as a spoken word -- use it as the
    cheap first filter and let `speech_seconds` make the real call.
    """
    voiced = int(np.count_nonzero(np.abs(buffer) >= energy_threshold))
    return voiced >= min_voiced_samples


def speech_seconds(
    audio: np.ndarray, sample_rate: int = 16000, min_speech_ms: int = 200
) -> float:
    """How many seconds of `audio` an actual voice-activity detector counts
    as speech.

    This runs the same Silero VAD that faster-whisper already applies
    internally, just *before* transcription instead of during it. That
    ordering is the whole point: `logs/neo.log` around 20:36 shows five
    consecutive transcriptions of 7.8s buffers, each logging "VAD filter
    removed 00:07.800 of audio" -- Whisper was handed pure silence, paid the
    full price of a small-model pass over it, and returned nothing. The
    amplitude guard above had waved that audio through because a transient
    cleared its threshold. VAD costs milliseconds against seconds for
    transcription, so asking it first is close to free.

    Returns -1.0 when the detector isn't available, so callers can tell
    "no speech" apart from "couldn't check" and fall back rather than
    silently swallowing the user's command.
    """
    if audio.size == 0:
        return 0.0
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError:
        return -1.0

    try:
        # threshold mirrors what stt.py passes to Whisper, so this gate and
        # Whisper's internal one agree on what counts as speech.
        # min_speech_duration_ms drops isolated clicks/bumps; the small
        # speech_pad keeps the measurement close to the real speech length
        # instead of inflating every blip by the 400ms default padding.
        options = VadOptions(
            threshold=0.5,
            min_speech_duration_ms=min_speech_ms,
            speech_pad_ms=100,
        )
        segments = get_speech_timestamps(
            np.asarray(audio, dtype=np.float32), options, sampling_rate=sample_rate
        )
    except Exception:
        logger.exception("VAD çalıştırılamadı")
        return -1.0

    total = sum(int(seg["end"]) - int(seg["start"]) for seg in segments)
    return total / float(sample_rate)


def trim_silence(
    audio: np.ndarray, sample_rate: int, threshold_ratio: float = 0.12, pad_seconds: float = 0.05
) -> np.ndarray:
    """Trims leading/trailing silence around the loudest part of `audio`,
    keeping whatever length the actual sound turns out to need.

    An earlier version cropped/padded every recording to one fixed duration,
    which was fine for same-speed utterances but silently truncated words
    spoken slowly (losing the tail) or padded fast ones with extra silence
    (diluting the features) -- speaking-rate differences ended up mattering
    more than word identity. Endpointing on where the sound actually is,
    then letting `normalize_features` resample *that* to a fixed frame
    count, keeps the full word regardless of how fast it was said.
    """
    if audio.size == 0:
        return audio
    abs_audio = np.abs(audio)
    peak = float(abs_audio.max())
    if peak <= 1e-6:
        return audio

    above = np.where(abs_audio >= peak * threshold_ratio)[0]
    if above.size == 0:
        return audio

    pad = int(pad_seconds * sample_rate)
    start = max(0, int(above[0]) - pad)
    end = min(audio.size, int(above[-1]) + 1 + pad)
    return audio[start:end]


def speech_span(
    audio: np.ndarray,
    sample_rate: int = 16000,
    min_speech_ms: int = 100,
    min_silence_ms: int = 200,
) -> tuple[np.ndarray, float] | None:
    """One VAD pass answering both questions a caller usually has: *where*
    the speech is (the span from the first detected word to the last) and
    *how much* of it there is.

    Returns None when VAD can't run, so callers can fall back rather than
    mistake "couldn't check" for "no speech".
    """
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32), 0.0
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        segments = get_speech_timestamps(
            np.asarray(audio, dtype=np.float32),
            # min_silence_duration_ms defaults to 2000 in faster-whisper,
            # which merges anything less than two seconds apart into a single
            # span. On a 2.2s enrollment clip that swallowed the breath before
            # the phrase and the noise after it, so a ~1s "Neo uyan" was
            # stored as a 2.1s template that was mostly not the phrase --
            # measured spread between the user's own takes (45.5) ended up as
            # large as the distance to unrelated speech (45.4). A short
            # silence window lets real pauses actually separate.
            VadOptions(
                threshold=0.5,
                min_speech_duration_ms=min_speech_ms,
                min_silence_duration_ms=min_silence_ms,
                speech_pad_ms=100,
            ),
            sampling_rate=sample_rate,
        )
    except ImportError:
        return None
    except Exception:
        logger.exception("VAD çalıştırılamadı")
        return None

    if not segments:
        return np.zeros(0, dtype=np.float32), 0.0

    total = sum(int(s["end"]) - int(s["start"]) for s in segments) / float(sample_rate)
    start, end = _dominant_cluster(segments, sample_rate)
    return audio[max(0, start) : min(audio.size, end)], total


# Words inside one phrase sit close together; a cough, a chair or a stray
# "ıı" lands further out. Anything separated by more than this starts a new
# group, so the span returned is the utterance rather than everything between
# the first and last noise in the clip.
_CLUSTER_GAP_SECONDS = 0.45


def _dominant_cluster(segments: list[dict], sample_rate: int) -> tuple[int, int]:
    """Start/end of the group of speech segments holding the most speech."""
    groups: list[list[dict]] = [[segments[0]]]
    for segment in segments[1:]:
        gap = (int(segment["start"]) - int(groups[-1][-1]["end"])) / float(sample_rate)
        if gap > _CLUSTER_GAP_SECONDS:
            groups.append([segment])
        else:
            groups[-1].append(segment)

    best = max(
        groups,
        key=lambda group: sum(int(s["end"]) - int(s["start"]) for s in group),
    )
    return int(best[0]["start"]), int(best[-1]["end"])


def extract_speech_segment(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """The spoken part of `audio`, from the start of the first detected word
    to the end of the last.

    `trim_silence` endpoints on amplitude relative to the loudest sample,
    which quietly fails on a microphone with a noticeable noise floor: on
    this machine an enrollment recording of a ~1s phrase came back
    "trimmed" to 2.184s -- the entire buffer -- because the room noise sat
    above 12% of the peak. The enrolled duration then drove the detection
    window out to nearly 4 seconds, where a window is always full of speech
    and the checks that depend on the phrase being isolated stop meaning
    anything. VAD endpoints on what is actually speech, so noise between
    words doesn't extend the segment.

    Falls back to `trim_silence` when VAD is unavailable or finds nothing.
    """
    span = speech_span(audio, sample_rate)
    if span is None or span[0].size == 0:
        return trim_silence(audio, sample_rate)
    return span[0]


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_filters: int, n_fft: int, sample_rate: int) -> np.ndarray:
    low_mel, high_mel = _hz_to_mel(np.array([0.0, sample_rate / 2]))
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(1, n_filters + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center > left:
            filters[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filters[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return filters


def _dct(x: np.ndarray, n_coeff: int) -> np.ndarray:
    n_filters = x.shape[1]
    basis = np.cos(
        np.pi / n_filters * (np.arange(n_filters)[:, None] + 0.5) * np.arange(n_coeff)[None, :]
    )
    return x @ basis


def extract_mfcc(
    audio: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 160,
    n_filters: int = 26,
) -> np.ndarray:
    """A small self-contained MFCC implementation (pre-emphasis -> framed FFT
    power spectrum -> mel filterbank -> log -> DCT). Not meant to numerically
    match librosa/python_speech_features -- only to be a consistent,
    discriminative feature for comparing our own recordings against each
    other via DTW, without adding a heavy dependency."""
    if audio.size == 0:
        return np.zeros((0, n_mfcc))

    emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1]).astype(np.float64)

    frame_length = n_fft
    num_frames = max(1, 1 + (len(emphasized) - frame_length) // hop_length)
    frames = np.zeros((num_frames, frame_length))
    window = np.hamming(frame_length)
    for i in range(num_frames):
        start = i * hop_length
        segment = emphasized[start : start + frame_length]
        frames[i, : len(segment)] = segment
    frames *= window

    magnitude = np.abs(np.fft.rfft(frames, n=n_fft, axis=1))
    power = (magnitude**2) / n_fft

    filterbank = _mel_filterbank(n_filters, n_fft, sample_rate)
    mel_energy = power @ filterbank.T
    mel_energy = np.where(mel_energy <= 0, np.finfo(np.float64).eps, mel_energy)
    log_mel = np.log(mel_energy)

    return _dct(log_mel, n_mfcc)


def normalize_features(mfcc: np.ndarray, target_frames: int = 40) -> np.ndarray:
    """Cepstral mean normalization (removes per-recording DC-ish offset)
    plus resampling the time axis to a fixed number of frames.

    Without this, two recordings of the same word at different speaking
    rates end up as very different-length MFCC sequences, and comparing
    them costs more (even under DTW's time warping) than comparing two
    *different* words of similar duration -- speaking-rate differences were
    completely swamping word identity. Resampling to a fixed frame count
    normalizes away rate/duration first, so DTW only has to absorb small
    residual misalignment.
    """
    if mfcc.shape[0] == 0:
        return np.zeros((target_frames, mfcc.shape[1] if mfcc.ndim == 2 else 13))

    centered = mfcc - mfcc.mean(axis=0, keepdims=True)
    original_frames = centered.shape[0]
    if original_frames == target_frames:
        return centered

    x_old = np.linspace(0.0, 1.0, original_frames)
    x_new = np.linspace(0.0, 1.0, target_frames)
    resampled = np.empty((target_frames, centered.shape[1]))
    for c in range(centered.shape[1]):
        resampled[:, c] = np.interp(x_new, x_old, centered[:, c])
    return resampled


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Dynamic time warping distance between two (frames, features)
    sequences, normalized by path length so it's roughly comparable across
    slightly different-length recordings."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        row_diff = np.linalg.norm(b - ai, axis=1)
        for j in range(1, m + 1):
            cost[i, j] = row_diff[j - 1] + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m] / (n + m))
