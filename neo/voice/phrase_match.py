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

    return max(whole, best_window)


def matches_wake_phrase(transcript: str, phrase: str, threshold: float = 0.85) -> bool:
    """Whether a transcript is a plausible rendering of the wake phrase.

    Default measured against real transcripts: see WAKE_PHRASE_THRESHOLD in
    wake_word.py for the scores this number comes from.
    """
    return similarity(transcript, phrase) >= threshold
