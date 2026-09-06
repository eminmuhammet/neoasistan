from __future__ import annotations

import logging
import queue

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class MicrophoneUnavailableError(Exception):
    """Raised when the microphone cannot be accessed."""


class PushToTalkRecorder:
    """Push-to-talk microphone capture: call start(), speak, then stop()."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.warning("Microphone status: %s", status)
        self._queue.put(indata.copy())

    def start(self) -> None:
        if self._stream is not None:
            return
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise MicrophoneUnavailableError(
                "sounddevice kurulu değil, mikrofon kullanılamıyor."
            ) from exc

        self._queue = queue.Queue()
        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            logger.exception("Mikrofon açılamadı")
            raise MicrophoneUnavailableError(
                "Mikrofona erişemiyorum. Mikrofon bağlantısını kontrol eder misin?"
            ) from exc

    def stop(self) -> np.ndarray:
        if self._stream is None:
            return np.zeros(0, dtype="float32")
        self._stream.stop()
        self._stream.close()
        self._stream = None

        chunks = []
        while not self._queue.empty():
            chunks.append(self._queue.get())
        if not chunks:
            return np.zeros(0, dtype="float32")
        return np.concatenate(chunks, axis=0).reshape(-1)
