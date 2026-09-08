from __future__ import annotations

import asyncio
import logging
import queue
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import numpy as np

from .audio_features import has_enough_speech, speech_seconds
from .clap_detector import ClapDetector
from .openwakeword_spotter import OpenWakeWordSpotter, WakeWordModelUnavailableError
from .stt import STTUnavailableError, WhisperSTT

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

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
    energy_threshold: float = 0.006
    command_timeout_seconds: float = 8.0
    # Every one of these seconds is dead air after the user has finished
    # talking, before NEO even starts thinking. 1.5s was noticeably long in
    # use; 1.0s still absorbs an ordinary mid-sentence pause.
    end_of_command_silence_seconds: float = 1.0
    # A short pause after a command finalizes before the wake detectors
    # listen again -- the tail of the user's own speech (or the room
    # settling) was occasionally similar enough to a wake trigger to fire
    # again immediately.
    reactivation_cooldown_seconds: float = 2.0
    # Live logs after the 0.006 threshold still showed genuine attempts
    # (peaks 0.03-0.09, clearly real speech, not silence) rejected because
    # only 0.06-0.16s of the buffer counted as "voiced" -- lowered further
    # so short/quiet real utterances stop being discarded.
    min_command_voiced_seconds: float = 0.08
    # ...but a threshold that low also waves through key presses and room
    # bumps, which is how Whisper ended up transcribing pure silence every
    # ~10s. VAD makes the actual speech/not-speech call; the shortest real
    # command seen in the logs ('Alo') carried 0.72s of speech, so 0.25s
    # leaves plenty of room under it.
    min_command_speech_seconds: float = 0.25
    # After NEO finishes speaking a reply, how long open_followup_window()
    # keeps the next utterance from needing the wake word repeated. Long
    # enough to think for a beat and answer back; short enough that an
    # unrelated sound half a minute later doesn't get mistaken for one.
    followup_window_seconds: float = 7.0


class WakeWordListener:
    """Always-on listening loop: stays silent until it hears "Hey Jarvis"
    (detected by a trained neural model, see openwakeword_spotter.py) or a
    deliberate double clap (see clap_detector.py), then records whatever
    follows until the user pauses or times out, and hands the whole
    utterance to Whisper in one piece, mirroring push-to-talk instead of
    chopping speech into fixed windows.

    Nothing is forwarded to the Agent/LLM until an actual command has been
    recognized, so ambient conversation never leaves the machine.
    """

    def __init__(
        self,
        wake_spotter: OpenWakeWordSpotter,
        stt: WhisperSTT,
        config: WakeWordConfig | None = None,
        clap_detector: ClapDetector | None = None,
    ) -> None:
        self._wake_spotter = wake_spotter
        self._stt = stt
        self._config = config or WakeWordConfig()
        # Optional: a caller that doesn't want the clap gesture (or wants a
        # differently-tuned one) can pass its own detector or None.
        self._clap_detector = clap_detector
        self._followup_deadline = 0.0
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None
        self._running = False

    def open_followup_window(self, seconds: float | None = None) -> None:
        """Lets the very next thing the user says be captured as a command
        directly, without the wake word, for `seconds` (default
        `WakeWordConfig.followup_window_seconds`).

        Call this right after NEO finishes speaking a reply. A natural
        back-and-forth doesn't repeat the wake phrase on every turn, and
        expecting it to would demand the same clear, deliberate diction the
        wake phrase itself needs to be recognized -- which an ordinary
        conversational reply usually doesn't have. One utterance consumes
        the window (successful or not); after that, or once `seconds` pass
        with nothing said, the wake word is required again.
        """
        self._followup_deadline = time.monotonic() + (
            seconds if seconds is not None else self._config.followup_window_seconds
        )

    def cancel_followup_window(self) -> None:
        """Closes an open follow-up window without waiting for it to expire."""
        self._followup_deadline = 0.0

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

    def _check_wake_triggers(self, chunk: np.ndarray) -> bool:
        """Feeds `chunk` to whichever wake detectors are active. Both run
        even after one fires in the same tick, so neither one's internal
        state (openWakeWord's debounce buffer, the clap detector's onset
        tracking) falls behind the audio actually heard."""
        woke = False
        try:
            if self._wake_spotter.feed(chunk):
                woke = True
        except WakeWordModelUnavailableError:
            logger.exception("Uyandırma modeli kullanılamıyor")
        if self._clap_detector is not None and self._clap_detector.feed(chunk):
            woke = True
        return woke

    async def run(self, on_wake: OnWake, on_command: OnCommand, is_muted: IsMuted | None = None) -> None:
        self._open_stream()
        self._running = True
        step_samples = int(self._config.step_seconds * SAMPLE_RATE)
        silence_gap_samples = int(self._config.end_of_command_silence_seconds * SAMPLE_RATE)

        # Loaded (and, on first run, downloaded) up front and then kept for
        # as long as continuous listening is on, since every wake-up
        # depends on it -- loading takes real time, and paying that on the
        # very first genuine attempt reads as "it didn't hear me".
        try:
            await asyncio.to_thread(self._wake_spotter.preload)
        except WakeWordModelUnavailableError as exc:
            raise WakeWordUnavailableError(str(exc)) from exc

        preload = getattr(self._stt, "preload", None)
        if preload is not None:
            asyncio.ensure_future(preload())

        rolling_energy_note = "alkış açık" if self._clap_detector is not None else "alkış kapalı"
        logger.info("Sürekli dinleme başladı (Hey Jarvis, %s)", rolling_energy_note)

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
                        if command_buffer is not None:
                            # A capture opened by the wake word is still
                            # live and the mute was almost certainly NEO's
                            # own activation chime, which plays immediately
                            # after waking. Dropping the capture here left
                            # the interface on "Dinliyor" forever while the
                            # listener quietly went back to watching for the
                            # wake word -- the user had said nothing wrong
                            # and had nothing to react to. Restart the
                            # capture instead, so the command they are about
                            # to speak is the one that gets recorded.
                            command_buffer = np.zeros(0, dtype="float32")
                            silence_run = 0
                            heard_speech = False
                            command_deadline = (
                                time.monotonic() + self._config.command_timeout_seconds
                            )

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
                                # Signal UI to reset from LISTENING back to IDLE
                                asyncio.ensure_future(on_command(""))
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
                                # Signal UI to reset from LISTENING back to IDLE
                                asyncio.ensure_future(on_command(""))
                                continue

                            try:
                                text = await self._stt.transcribe(finished)
                            except STTUnavailableError:
                                logger.exception("Command transcription failed")
                                asyncio.ensure_future(on_command(""))
                                continue
                            if text:
                                logger.info("Komut transkripti: %r", text)
                            else:
                                logger.info("Komut atlandı: transkript boş")
                            # Always reported, empty included: every path out
                            # of a command capture has to tell the UI, or it
                            # sits on "Dinliyor" until something else happens.
                            asyncio.ensure_future(on_command(text or ""))
                        continue

                    if time.monotonic() < cooldown_until:
                        continue

                    if time.monotonic() < self._followup_deadline:
                        # A follow-up window is open: what matters is only
                        # that the user said something back, not that they
                        # repeated the wake word -- so the detectors below
                        # are skipped here entirely. Downstream, the same
                        # VAD/duration checks a normal command goes through
                        # still apply, so a stray noise just quietly fails
                        # there instead of ever reaching a wake trigger.
                        if not is_loud:
                            continue
                        self._followup_deadline = 0.0  # one utterance consumes it
                        asyncio.ensure_future(on_wake())
                        command_buffer = np.zeros(0, dtype="float32")
                        silence_run = 0
                        heard_speech = False
                        command_deadline = time.monotonic() + self._config.command_timeout_seconds
                        continue

                    if self._check_wake_triggers(new_audio):
                        asyncio.ensure_future(on_wake())
                        command_buffer = np.zeros(0, dtype="float32")
                        silence_run = 0
                        heard_speech = False
                        command_deadline = time.monotonic() + self._config.command_timeout_seconds
                except Exception:
                    logger.exception("Sürekli dinleme döngüsünde beklenmeyen hata")
                    if command_buffer is not None:
                        # Same reason as the unmute path: a capture that
                        # disappears without a word leaves the UI stuck on
                        # "Dinliyor".
                        asyncio.ensure_future(on_command(""))
                    command_buffer = None
                    await asyncio.sleep(1.0)
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        self._followup_deadline = 0.0
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
