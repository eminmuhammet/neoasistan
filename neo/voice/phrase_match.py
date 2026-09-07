from __future__ import annotations

import re
from difflib import SequenceMatcher

# Turkish characters folded to ASCII so a transcript written "uyan" and one
# written "uyân" compare equal, and so casing differences don't matter.
_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")
_NON_WORD = re.compile(r"[^a-z0-9\s]")


def normalize(text: str) -> str:
    lowered = text.replace("I", "ı").replace("İ", "i").lower()
    stripped = _NON_WORD.sub(" ", lowered.translate(_FOLD))
    return " ".join(stripped.split())


def similarity(candidate: str, phrase: str) -> float:
    """How close a transcript is to the wake phrase, 0..1.

    Speech-to-text will not reproduce a short wake phrase exactly -- live
    logs show "Neo" alone coming back as "Ne?", "Ne o?" and "No.", which is
    why matching on an exact string failed and the acoustic matcher was
    written in the first place. A two-word phrase gives the recognizer far
    more to work with, so approximate agreement becomes a usable signal
    where exact agreement never was.
    """
    a, b = normalize(candidate), normalize(phrase)
    if not a or not b:
        return 0.0
    whole = SequenceMatcher(None, a, b).ratio()

    # The phrase may sit inside a longer transcript ("neo uyan hava nasıl"),
    # so also score the best matching window of the right length.
    words = a.split()
    target_len = len(b.split())
    best_window = 0.0
    for size in {target_len, target_len + 1}:
        for start in range(0, max(1, len(words) - size + 1)):
            window = " ".join(words[start : start + size])
            best_window = max(best_window, SequenceMatcher(None, window, b).ratio())

    return max(whole, best_window, _best_trailing_suffix_match(a, b))


def _best_trailing_suffix_match(candidate: str, phrase: str) -> float:
    """Credit for a transcript that matches only the tail of the phrase.

    Observed live (logs/neo.log, 2026-09-07 13:23:45): a genuine "Neo uyan"
    came back from Whisper as just 'uyan.' -- the softer, quieter first
    word got trimmed by VAD or simply spoken too quietly to transcribe, and
    the bare second word scored only 0.67 (whole-string) against the full
    two-word phrase, below the 0.78 acceptance threshold. That forced the
    user to repeat themselves for something they'd already said correctly.

    Matching the candidate against a trailing chunk of the phrase (rather
    than the whole phrase) removes the length penalty a dropped leading
    word would otherwise cause, discounted by how much of the phrase that
    chunk actually covers so a full match still always wins outright and a
    single trailing word of a long phrase can't pass on its own.
    """
    phrase_words = phrase.split()
    if len(phrase_words) < 2:
        return 0.0
    best = 0.0
    for start in range(1, len(phrase_words)):
        remaining = len(phrase_words) - start
        if remaining < max(1, len(phrase_words) // 2):
            continue
        suffix = " ".join(phrase_words[start:])
        ratio = SequenceMatcher(None, candidate, suffix).ratio()
        coverage = remaining / len(phrase_words)
        best = max(best, ratio * (0.6 + 0.4 * coverage))
    return best


def matches_wake_phrase(transcript: str, phrase: str, threshold: float = 0.78) -> bool:
    """Whether a transcript is a plausible rendering of the wake phrase.

    Default measured against real transcripts, not clean text: live
    attempts at saying "Neo uyan" came back as 'Ne yok, uyan.' (0.84) and
    'Ne o ya?' (0.80); background noise came back as 'Ne oluyor?' (0.59).
    An earlier 0.85 -- derived from tidy strings -- would have rejected
    every genuine attempt in that log. See WAKE_PHRASE_THRESHOLD in
    wake_word.py, which is the value actually used at runtime; this default
    exists so the function is sensible when called on its own (as the tests
    do) and must be kept in sync with it.
    """
    return similarity(transcript, phrase) >= threshold
