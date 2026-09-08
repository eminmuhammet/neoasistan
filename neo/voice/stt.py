from __future__ import annotations

import asyncio
import gc
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

IDLE_UNLOAD_SECONDS = 300.0

# Whisper was trained on a lot of subtitled video, and on marginal audio it
# falls back to the boilerplate that padded those subtitle tracks. These come
# out fluent and high-confidence, so no threshold inside the model filters
# them -- but they're a small, closed set, and one of them ('Altyazı M.K.')
# was observed in logs/neo.log at 20:36:00 going all the way to the Claude
# API as if the user had said it. Nobody asks an assistant to subscribe to a
# channel, so dropping these outright costs nothing.
_HALLUCINATION_PHRASES = (
    "altyazi mk",
    "altyazi m k",
    "altyazi",
    "abone olmayi unutmayin",
    "abone olun",
    "kanalima abone olun",
    "izlediginiz icin tesekkurler",
    "izlediginiz icin tesekkur ederim",
    "altyazi ve ceviri",
    "turkce altyazi",
    "bu videoyu begendiyseniz",
    "altyazi eklenmistir",
)

_TR_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _fold(text: str) -> str:
    """Turkish-aware normalization for blocklist comparison: lowercase,
    strip diacritics and punctuation, collapse whitespace."""
    lowered = text.replace("I", "ı").replace("İ", "i").lower()
    stripped = _PUNCT.sub(" ", lowered.translate(_TR_FOLD))
    return " ".join(stripped.split())


def is_hallucination(text: str) -> bool:
    """True when a transcript is one of Whisper's known subtitle artefacts
    rather than something the user actually said."""
    folded = _fold(text)
    if not folded:
        return True
    return any(
        folded == phrase or folded.startswith(phrase + " ") or folded.endswith(" " + phrase)
        for phrase in _HALLUCINATION_PHRASES
    )


class STTUnavailableError(Exception):
    """Raised when speech-to-text cannot be performed."""


class WhisperSTT:
    """Offline, local speech-to-text via faster-whisper. The model is
    downloaded from Hugging Face and loaded lazily on first use, and
    released again after a stretch of inactivity -- the loaded "small"
    model holds several hundred MB, which is a lot to keep resident in an
    assistant that idles in the background most of the day."""

    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8") -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None
        self._last_used = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def unload_if_idle(self, idle_seconds: float = IDLE_UNLOAD_SECONDS) -> bool:
        """Frees the model when it hasn't been used recently. Returns True if
        something was actually released."""
        if self._model is None:
            return False
        if time.monotonic() - self._last_used < idle_seconds:
            return False
        self._model = None
        gc.collect()
        logger.info("Whisper modeli boşta olduğu için bellekten kaldırıldı")
        return True

    def _ensure_model(self):
        self._last_used = time.monotonic()
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise STTUnavailableError(
                "faster-whisper kurulu değil, sesli komut algılanamıyor."
            ) from exc
        try:
            self._model = WhisperModel(
                self._model_size, device=self._device, compute_type=self._compute_type
            )
        except Exception as exc:
            logger.exception("Whisper modeli yüklenemedi")
            raise STTUnavailableError("Konuşma tanıma modeli yüklenemedi.") from exc
        return self._model

    def _transcribe_sync(self, audio: Any) -> str:
        model = self._ensure_model()
        # condition_on_previous_text=False: don't let the model bias new
        # segments on earlier (possibly hallucinated) text within this call.
        # A stricter VAD threshold reduces how often near-silence gets
        # treated as speech in the first place -- Whisper is prone to
        # hallucinating fluent, plausible-sounding (but fabricated) Turkish
        # text -- often YouTube-subtitle-style phrases -- when fed audio
        # that's mostly silence/noise, and high-confidence hallucinations
        # aren't reliably caught by logprob/no-speech thresholds alone.
        # no_repeat_ngram_size + a mild repetition_penalty guard against
        # another Whisper failure mode observed live: getting stuck looping
        # a phrase ("Bu yollara... Bu yollara... Bu yollara...") on
        # marginal/noisy audio instead of producing one clean transcript.
        segments, _info = model.transcribe(
            audio,
            language="tr",
            vad_filter=True,
            vad_parameters={"threshold": 0.5, "min_speech_duration_ms": 250},
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if text and is_hallucination(text):
            logger.info("Transkript halüsinasyon olarak elendi: %r", text)
            return ""
        return text

    async def preload(self) -> None:
        """Loads the model now so the first real use doesn't wait for it.

        Loading takes seconds, and for the wake-word confirmer that wait
        lands squarely between the user saying the phrase and NEO reacting --
        which reads as "it didn't hear me".
        """
        try:
            await asyncio.to_thread(self._ensure_model)
        except STTUnavailableError:
            logger.info("Model önceden yüklenemedi, ilk kullanımda denenecek")

    async def transcribe(self, audio: Any) -> str:
        try:
            return await asyncio.to_thread(self._transcribe_sync, audio)
        except STTUnavailableError:
            raise
        except Exception as exc:
            logger.exception("STT failed")
            raise STTUnavailableError("Ses metne çevrilemedi.") from exc
