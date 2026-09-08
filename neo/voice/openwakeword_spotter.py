from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "hey_jarvis"
# Established default from openWakeWord's own examples (and what Home
# Assistant's integration uses). Checked against this project's own
# synthesized audio before adopting it: genuine "Hey Jarvis" utterances
# scored 0.999, unrelated Turkish and English speech both scored 0.000 --
# an enormous margin either side of 0.5, unlike the ~4%-wide gap the old
# DTW-based matcher was working with (see keyword_spotter.py's history).
DEFAULT_THRESHOLD = 0.5
# Keeps one utterance from firing on_wake() repeatedly while the model's
# score stays elevated across several consecutive 80ms frames.
DEFAULT_DEBOUNCE_SECONDS = 1.25

SAMPLE_RATE = 16000
# openWakeWord's native analysis frame; feed() accepts any chunk size (the
# model buffers internally), this is just documentation of what it wants.
FRAME_SAMPLES = 1280  # 80 ms @ 16 kHz


class WakeWordModelUnavailableError(Exception):
    """Raised when the ONNX model can't be loaded (missing dependency, or
    the weights haven't been downloaded yet and there's no network)."""


class OpenWakeWordSpotter:
    """A trained neural wake-word detector, replacing the DTW-against-the-
    user's-own-recordings approach in keyword_spotter.py.

    That approach needed the user to enroll several takes of the phrase,
    and on this project's own voice it still couldn't reliably separate
    the wake phrase from ordinary speech -- the nearest negative sample
    sat within about 4% of the enrolled spread. A model trained on
    thousands of synthetic voices generalizes across speakers and accents
    by construction, and needs no per-user setup.

    Only English wake words ship pretrained (alexa, hey_mycroft,
    hey_jarvis, hey_rhasspy, timer, weather) -- there is no "Neo" model,
    and training a custom one is a separate, much larger undertaking
    (thousands of synthetic TTS samples, meaningful GPU time). "Hey
    Jarvis" is the closest fit available today.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
        models_dir: Path | None = None,
    ) -> None:
        self._model_name = model_name
        self._threshold = threshold
        self._debounce_seconds = debounce_seconds
        # Where the ONNX weights are downloaded to and loaded from. Left
        # None only for quick scripts/tests; main.py always passes
        # settings.data_dir -- openWakeWord's own default is a path inside
        # the installed package, which under PyInstaller resolves inside
        # the bundle's `_internal` folder and gets wiped on every update
        # (the same class of bug settings.user_data_root()'s docstring
        # describes for the calendar and wake-word-enrollment data this
        # module replaces).
        self._models_dir = models_dir
        # Loaded lazily so simply constructing this object never touches
        # disk or network -- important for tests and for the many call
        # sites (e.g. building the tool registry) that don't actually need
        # wake-word detection.
        self._model = None
        # The key openWakeWord's own `predict()` uses in its returned
        # scores dict. Defaults to `model_name`, which is correct whenever
        # the model is loaded by bare name; when this wrapper loads it from
        # an explicit file path instead (see _ensure_model), the real key
        # is resolved from the loaded model and may differ, so it's
        # refined there rather than assumed to stay in sync.
        self._score_key: str = model_name

    def _target_dir(self) -> str:
        if self._models_dir is not None:
            self._models_dir.mkdir(parents=True, exist_ok=True)
            return str(self._models_dir)
        # openWakeWord's own default: a `resources/models` folder next to
        # the installed package.
        import openwakeword

        return os.path.join(Path(openwakeword.__file__).parent, "resources", "models")

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            import openwakeword
            from openwakeword.model import Model
            from openwakeword.utils import download_models
        except ImportError as exc:
            raise WakeWordModelUnavailableError(
                "openwakeword kurulu değil, uyandırma kelimesi kullanılamıyor."
            ) from exc
        try:
            target_dir = self._target_dir()
            # A no-op once the weights are already on disk (checked by
            # filename); only reaches the network on the very first run.
            download_models(model_names=[self._model_name], target_directory=target_dir)

            # Pass explicit file paths rather than the bare model name.
            # Otherwise Model.__init__ resolves the name through
            # `get_pretrained_model_paths()`, which -- like the download
            # above -- points at the package's own install location, not
            # wherever this spotter was told to keep its weights.
            url = openwakeword.MODELS[self._model_name]["download_url"]
            wakeword_filename = os.path.basename(url).replace(".tflite", ".onnx")
            wakeword_path = os.path.join(target_dir, wakeword_filename)

            self._model = Model(
                wakeword_models=[wakeword_path],
                inference_framework="onnx",
                melspec_model_path=os.path.join(target_dir, "melspectrogram.onnx"),
                embedding_model_path=os.path.join(target_dir, "embedding_model.onnx"),
            )
        except Exception as exc:
            logger.exception("openWakeWord modeli yüklenemedi")
            raise WakeWordModelUnavailableError(
                "Uyandırma kelimesi modeli yüklenemedi (ilk çalıştırmada internet "
                "gerekir, sonrasında tamamen çevrimdışı çalışır)."
            ) from exc

        # Loading from an explicit file path (rather than a bare name) key
        # the returned model by its filename stem (e.g. "hey_jarvis_v0.1"),
        # not necessarily self._model_name -- read back whatever key
        # actually landed in the loaded model instead of assuming they
        # match, so a future openWakeWord release changing that filename
        # scheme can't silently make feed() stop detecting anything.
        self._score_key = next(iter(self._model.models.keys()))
        return self._model

    def preload(self) -> None:
        """Loads (and, on first run, downloads) the model up front, so the
        very first wake attempt of a session isn't the one that pays for
        it -- loading measured at ~0.25s, small but still better spent
        before the user is waiting on a reply."""
        self._ensure_model()

    def feed(self, chunk: np.ndarray) -> bool:
        """Feed mono float32 audio (-1..1) at SAMPLE_RATE, any chunk size.
        Returns True at most once per utterance (debounced) when the wake
        word is heard in it."""
        model = self._ensure_model()
        pcm16 = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
        scores = model.predict(
            pcm16,
            threshold={self._score_key: self._threshold},
            debounce_time=self._debounce_seconds,
        )
        return scores.get(self._score_key, 0.0) >= self._threshold
