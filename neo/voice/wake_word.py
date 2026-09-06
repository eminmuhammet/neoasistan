from __future__ import annotations

import asyncio
import logging
import queue
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import numpy as np

from .audio_features import has_enough_speech, speech_seconds
from .phrase_match import similarity
from .keyword_spotter import KeywordSpotter
from .stt import STTUnavailableError, WhisperSTT

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# The wake word is a single short word, so VAD's default minimum segment
# length would be right at the edge of discarding it. Lowered only for the
# wake check; command audio keeps the stricter default.
# Wake-word acoustic gating (speech present, phrase isolated, duration
# plausible) lives in KeywordSpotter: every one of those checks needs the
# enrolled recordings as its reference, and doing them there lets a single
# VAD pass serve all of them plus the DTW comparison.

# Measured against real transcripts rather than picked by feel. At 0.72 the
# live log shows "Ne o ya?" scoring 0.80 and waking NEO mid-conversation --
# ordinary Turkish filler that happens to share most of its letters with
# "neo uyan". Scoring the realistic set puts genuine renderings at 0.88 and
# above ("neyo uyan" 0.94, "neo uyar" 0.88) and everything else at 0.59 and
# below, with that one phrase in between.
#
# The margin is thin (0.08), so it is worth knowing the failure mode: a
# rejected real phrase costs one repeat, an accepted false one wakes NEO
# while the user is talking to someone else.
WAKE_PHRASE_THRESHOLD = 0.85

OnWake = Callable[[], Awaitable[None]]
OnCommand = Callable[[str], Awaitable[None]]
IsMuted = Callable[[], bool]


class WakeWordUnavailableError(Exception):
    """Raised when continuous listening cannot be started."""


def _should_finalize_command(
    command_buffer_len: int,
    silence_run: int,
    silence_gap_samples: int,
    step_samples: int,
    timed_out: bool,
    heard_speech: bool,
) -> bool:
    if timed_out:
        return True
    if not heard_speech:
        # The user hasn't started talking yet -- a pause here just means
        # they're taking a breath after saying "Neo", not that they're done.
        # Without this, any natural hesitation before the actual command
        # got treated as "finished speaking" and finalized an empty buffer,
        # forcing the user to say the wake word again and again.
        return False
    long_pause = silence_run >= silence_gap_samples
    return long_pause and command_buffer_len > step_samples


@dataclass
class WakeWordConfig:
    step_seconds: float = 0.3
    detect_window_seconds: float = 1.4
    # This mic's real speech turned out to run much quieter than assumed
    # (0.015 rejected the user's actual voice almost every time -- see
    # `logs/neo.log` "Komut atlandı" spam after this guard first shipped).
    # Lowered with headroom above the ~0.0007 measured ambient-noise floor.
    energy_threshold: float = 0.006
    command_timeout_seconds: float = 8.0
    end_of_command_silence_seconds: float = 1.5
    reactivation_cooldown_seconds: float = 2.0
    # Live logs after the 0.006 threshold still showed genuine attempts
    # (peaks 0.03-0.09, clearly real speech, not silence) rejected because
    # only 0.06-0.16s of the buffer counted as "voiced" -- lowered further
    # so short/quiet real utterances stop being discarded.
    min_command_voiced_seconds: float = 0.12
    # ...but a threshold that low also waves through key presses and room
    # bumps, which is how Whisper ended up transcribing pure silence every
    # ~10s. VAD makes the actual speech/not-speech call; the shortest real
    # command seen in the logs ('Alo') carried 0.72s of speech, so 0.25s
    # leaves plenty of room under it.
    min_command_speech_seconds: float = 0.25


class WakeWordListener:
    """Always-on listening loop: stays silent until it hears the enrolled
    wake word -- matched acoustically via KeywordSpotter against the user's
    own recordings, not guessed from generic speech-to-text (see that
    module's docstring for why) -- then records whatever follows until the
    user pauses or times out, and hands the whole utterance to Whisper in
    one piece, mirroring push-to-talk instead of chopping speech into fixed
    windows.

    Nothing is forwarded to the Agent/LLM until an actual command has been
    recognized, so ambient conversation never leaves the machine.
    """

    def __init__(
        self,
        spotter: KeywordSpotter,
        stt: WhisperSTT,
        config: WakeWordConfig | None = None,
        wake_phrase: str | None = None,
        confirm_stt: WhisperSTT | None = None,
    ) -> None:
        self._spotter = spotter
        self._stt = stt
        self._config = config or WakeWordConfig()
        # Acoustic matching alone could not separate the wake phrase from
        # ordinary speech on this user's voice: with clean ~1s enrollments,
        # their own three takes sat 35-44 apart while nine of ten windows of
        # unrelated speech sat 32-42 away -- the templates were no closer to
        # each other than to everything else. So DTW is demoted to a cheap
        # pre-filter and the words themselves are confirmed by recognition,
        # which is a far stronger discriminator for a two-word phrase.
        self._wake_phrase = wake_phrase
        self._confirm_stt = confirm_stt
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None
        self._running = False

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.warning("Microphone status: %s", status)
        self._queue.put(indata.copy().reshape(-1))

    def _open_stream(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise WakeWordUnavailableError(
                "sounddevice kurulu değil, sürekli dinleme kullanılamıyor."
            ) from exc
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            logger.exception("Mikrofon açılamadı")
            raise WakeWordUnavailableError(
                "Mikrofona erişemiyorum. Mikrofon bağlantısını kontrol eder misin?"
            ) from exc

    def _collect_chunk(self, num_samples: int) -> np.ndarray:
        buffer: list[np.ndarray] = []
        collected = 0
        while collected < num_samples and self._running:
            try:
                block = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            buffer.append(block)
            collected += len(block)
        if not buffer:
            return np.zeros(0, dtype="float32")
        return np.concatenate(buffer)[:num_samples]

    def _drain(self) -> None:
        """Discard whatever audio has piled up, e.g. right after unmuting."""
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    async def run(self, on_wake: OnWake, on_command: OnCommand, is_muted: IsMuted | None = None) -> None:
        if not self._spotter.has_enough_templates():
            raise WakeWordUnavailableError(
                "Uyandırma kelimesi henüz öğretilmedi. Önce 'Neo'yu öğret."
            )

        self._open_stream()
        self._running = True
        step_samples = int(self._config.step_seconds * SAMPLE_RATE)
        # Sized from what the user actually enrolled rather than a fixed
        # constant: a two-word phrase needs a wider window than a single
        # short word, and the isolation check below needs silence around the
        # phrase to have something to measure.
        window_seconds = max(
            self._config.detect_window_seconds, self._spotter.detection_window_seconds
        )
        window_samples = int(window_seconds * SAMPLE_RATE)
        logger.info("Uyandırma penceresi: %.2f sn", window_seconds)
        silence_gap_samples = int(self._config.end_of_command_silence_seconds * SAMPLE_RATE)

        rolling = np.zeros(0, dtype="float32")
        command_buffer: np.ndarray | None = None
        silence_run = 0
        heard_speech = False
        command_deadline = 0.0
        cooldown_until = 0.0
        was_muted = False

        try:
            while self._running:
                try:
                    if is_muted is not None and is_muted():
                        # NEO is speaking: don't listen to our own voice
                        # coming back through the microphone.
                        was_muted = True
                        await asyncio.to_thread(self._collect_chunk, step_samples)
                        continue

                    if was_muted:
                        was_muted = False
                        self._drain()
                        rolling = np.zeros(0, dtype="float32")
                        command_buffer = None

                    new_audio = await asyncio.to_thread(self._collect_chunk, step_samples)
                    if new_audio.size == 0:
                        continue

                    is_loud = float(np.abs(new_audio).max()) >= self._config.energy_threshold

                    if command_buffer is not None:
                        command_buffer = np.concatenate([command_buffer, new_audio])
                        heard_speech = heard_speech or is_loud
                        silence_run = 0 if is_loud else silence_run + len(new_audio)
                        finalize = _should_finalize_command(
                            len(command_buffer),
                            silence_run,
                            silence_gap_samples,
                            step_samples,
                            timed_out=time.monotonic() > command_deadline,
                            heard_speech=heard_speech,
                        )
                        if finalize:
                            finished, command_buffer = command_buffer, None
                            # Ignore the microphone briefly after handing off a
                            # command: the tail of the user's own speech (or
                            # the room settling) was occasionally similar
                            # enough to re-trigger detection immediately,
                            # producing a spurious extra "Dinliyorum."
                            cooldown_until = time.monotonic() + self._config.reactivation_cooldown_seconds
                            self._drain()
                            rolling = np.zeros(0, dtype="float32")

                            min_voiced = int(self._config.min_command_voiced_seconds * SAMPLE_RATE)
                            if not has_enough_speech(finished, self._config.energy_threshold, min_voiced):
                                voiced = int(np.count_nonzero(np.abs(finished) >= self._config.energy_threshold))
                                logger.info(
                                    "Komut atlandı: yeterli konuşma sesi yok (halüsinasyon riski) "
                                    "[peak=%.4f, voiced=%.2fs/%.2fs, threshold=%.4f]",
                                    float(np.abs(finished).max()) if finished.size else 0.0,
                                    voiced / SAMPLE_RATE,
                                    min_voiced / SAMPLE_RATE,
                                    self._config.energy_threshold,
                                )
                                continue

                            # Loudness said "maybe"; ask a real speech
                            # detector before paying for transcription.
                            detected = await asyncio.to_thread(
                                speech_seconds, finished, SAMPLE_RATE
                            )
                            if 0.0 <= detected < self._config.min_command_speech_seconds:
                                logger.info(
                                    "Komut atlandı: VAD konuşma bulamadı "
                                    "[konuşma=%.2fs, gereken=%.2fs, kayıt=%.1fs]",
                                    detected,
                                    self._config.min_command_speech_seconds,
                                    finished.size / SAMPLE_RATE,
                                )
                                continue

                            try:
                                text = await self._stt.transcribe(finished)
                            except STTUnavailableError:
                                logger.exception("Command transcription failed")
                                continue
                            if text:
                                logger.info("Komut transkripti: %r", text)
                                asyncio.ensure_future(on_command(text))
                        continue

                    rolling = np.concatenate([rolling, new_audio])[-window_samples:]
                    if not is_loud or time.monotonic() < cooldown_until:
                        continue

                    matched = await asyncio.to_thread(self._spotter.is_match, rolling)
                    if not matched:
                        continue

                    if not await self._confirm_wake_phrase(rolling):
                        continue

                    rolling = np.zeros(0, dtype="float32")
                    asyncio.ensure_future(on_wake())
                    command_buffer = np.zeros(0, dtype="float32")
                    silence_run = 0
                    heard_speech = False
                    command_deadline = time.monotonic() + self._config.command_timeout_seconds
                except Exception:
                    logger.exception("Sürekli dinleme döngüsünde beklenmeyen hata")
                    command_buffer = None
                    await asyncio.sleep(1.0)
        finally:
            self.stop()

    async def _confirm_wake_phrase(self, window: np.ndarray) -> bool:
        """Second opinion on an acoustic match: does the audio actually say
        the wake phrase?

        Runs only after the spotter has already accepted the window, so the
        recognizer is invoked on the rare candidate rather than continuously.
        When no confirmer is configured the acoustic verdict stands, so this
        can be turned off without disabling continuous listening.
        """
        if self._confirm_stt is None or not self._wake_phrase:
            return True
        try:
            transcript = await self._confirm_stt.transcribe(window)
        except STTUnavailableError:
            # Never let a broken confirmer make NEO deaf -- fall back to the
            # acoustic decision it was only ever meant to refine.
            logger.exception("Uyandırma doğrulaması yapılamadı, akustik karar geçerli")
            return True

        if not transcript:
            logger.debug("Uyandırma reddedildi: doğrulamada metin çıkmadı")
            return False

        score = similarity(transcript, self._wake_phrase)
        accepted = score >= WAKE_PHRASE_THRESHOLD
        logger.info(
            "Uyandırma doğrulama: %r -> benzerlik %.2f (%s)",
            transcript,
            score,
            "kabul" if accepted else "red",
        )
        return accepted

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
