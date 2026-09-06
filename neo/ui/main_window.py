from __future__ import annotations

import asyncio
import logging
import os

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPushButton,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from ..config import autostart
from ..config.version import __version__
from ..core.updater import UpdateError, apply_update, check_for_update, download_update
from ..core.agent import Agent, extract_spoken_summary
from ..core.state import AgentState
from ..voice.audio_features import has_enough_speech, speech_seconds
from ..voice.chime import play_activation_chime
from ..voice.keyword_spotter import KeywordSpotter
from ..voice.microphone import SAMPLE_RATE as MIC_SAMPLE_RATE
from ..voice.microphone import MicrophoneUnavailableError, PushToTalkRecorder
from ..voice.stt import STTUnavailableError, WhisperSTT
from ..voice.tts import TextToSpeech, TTSUnavailableError, strip_speech_noise
from ..voice.wake_word import WakeWordListener, WakeWordUnavailableError
from .chat_view import ChatView
from .icon import build_app_icon
from .stats_panel import StatsPanel
from .status_orb import StatusOrb
from .theme import DARK_QSS

logger = logging.getLogger(__name__)

ABOUT_TEXT = """<h3>NEO</h3>
<p>Windows için kişisel yapay zekâ masaüstü asistanı.</p>
<p><b>Yapabildiklerim (API'siz):</b> saat/tarih, CPU/RAM/GPU/disk durumu,
hava durumu, takvim notları, uygulama/web sitesi açma, günaydın özeti.</p>
<p><b>Claude gerektirenler:</b> genel sohbet, web araştırması, karmaşık
istekler.</p>
<p><b>Sesli kontrol:</b> 🎙 basılı tutup konuş, ya da "🎓 Neo'yu öğret" ile
sesini öğretip "Neo" diyerek uyandır.</p>
<p style="color:#7c8b98;">Phase 1-4 tamamlandı · v0.1.0</p>
"""

ENROLLMENT_SAMPLES = 3
ENROLLMENT_PREPARE_SECONDS = 1.2
ENROLLMENT_RECORD_SECONDS = 2.2
# Long enough to yield a useful spread of "not the wake word" windows without
# turning enrollment into a chore.
NEGATIVE_RECORD_SECONDS = 10.0
ENROLLMENT_MAX_RETRIES = 3

# Same Whisper-hallucination guard used for the continuous wake-word command
# capture (see neo/voice/wake_word.py) applied to push-to-talk recordings.
# Kept in sync with WakeWordConfig's energy_threshold -- this mic's real
# speech runs quieter than a naive guess, see the comment there.
PTT_ENERGY_THRESHOLD = 0.006
PTT_MIN_VOICED_SECONDS = 0.12
PTT_MIN_SPEECH_SECONDS = 0.25
# How long a graceful shutdown gets before the process is ended outright.
# Audio and model threads don't always cooperate, and the update helper is
# blocked waiting for this process to disappear.
SHUTDOWN_GRACE_MS = 3000

STATE_LABELS = {
    AgentState.IDLE: "Hazır",
    AgentState.LISTENING: "Dinleniyor...",
    AgentState.PROCESSING: "Düşünüyor...",
    AgentState.SPEAKING: "Konuşuyor...",
    AgentState.ERROR: "Hata",
}


class MainWindow(QMainWindow):
    def __init__(
        self,
        agent: Agent,
        recorder: PushToTalkRecorder | None = None,
        stt: WhisperSTT | None = None,
        tts: TextToSpeech | None = None,
        wake_listener: WakeWordListener | None = None,
        spotter: KeywordSpotter | None = None,
        wake_phrase: str = "Neo uyan",
        update_manifest_url: str | None = None,
    ) -> None:
        super().__init__()
        self._agent = agent
        self._wake_phrase = wake_phrase
        self._update_manifest_url = update_manifest_url
        self._pending_update = None
        self._recorder = recorder
        self._stt = stt
        self._tts = tts
        self._wake_listener = wake_listener
        self._spotter = spotter
        self._wake_task: asyncio.Task | None = None
        self._active_command_task: asyncio.Task | None = None
        self._enrolling = False
        self._speaking = False
        self._quitting = False
        self._tray_notice_shown = False
        self._tray: QSystemTrayIcon | None = None
        self._app_icon = build_app_icon()
        self.setWindowTitle("NEO")
        self.setWindowIcon(self._app_icon)
        self.resize(520, 720)
        self.setStyleSheet(DARK_QSS)
        self._build_ui()
        self._build_tray_icon()
        self._set_state(AgentState.IDLE)

        # The loaded speech model holds several hundred MB; drop it again
        # once voice has been idle for a while (it reloads on next use).
        self._idle_unload_timer = QTimer(self)
        self._idle_unload_timer.setInterval(60_000)
        self._idle_unload_timer.timeout.connect(self._release_idle_resources)
        self._idle_unload_timer.start()
        # Sürekli dinleme (wake-word) varsayılan olarak KAPALI başlar: mevcut
        # DIY uygulaması yanlış tetiklenmeye açık, push-to-talk daha güvenilir.
        # Kullanıcı isterse anahtarla açabilir.

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        header_row = QHBoxLayout()
        header_row.addStretch(1)
        title = QLabel("N E O")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_row.addWidget(title)
        header_row.addStretch(1)
        info_button = QPushButton("ℹ")
        info_button.setObjectName("InfoButton")
        info_button.setToolTip("NEO hakkında")
        info_button.clicked.connect(self._on_info_clicked)
        header_row.addWidget(info_button, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addLayout(header_row)

        orb_col = QVBoxLayout()
        orb_col.setAlignment(Qt.AlignmentFlag.AlignCenter)
        orb_col.setSpacing(6)
        self._status_orb = StatusOrb()
        orb_col.addWidget(self._status_orb, alignment=Qt.AlignmentFlag.AlignCenter)
        self._status_label = QLabel()
        self._status_label.setObjectName("StatusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        orb_col.addWidget(self._status_label)
        self._research_label = QLabel()
        self._research_label.setObjectName("ResearchLabel")
        self._research_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._research_label.hide()
        orb_col.addWidget(self._research_label)
        layout.addLayout(orb_col)

        stats_panel = StatsPanel()
        stats_panel.setObjectName("StatsPanel")
        stats_layout = QVBoxLayout()
        stats_layout.setContentsMargins(12, 10, 12, 10)
        stats_layout.addWidget(stats_panel)
        stats_container = QWidget()
        stats_container.setObjectName("StatsPanel")
        stats_container.setLayout(stats_layout)
        layout.addWidget(stats_container)

        control_panel = QWidget()
        control_panel.setObjectName("ControlPanel")
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(12, 10, 12, 10)
        control_layout.setSpacing(8)

        enroll_row = QHBoxLayout()
        self._enroll_label = QLabel()
        self._enroll_label.setObjectName("StatusLabel")
        enroll_row.addWidget(self._enroll_label, stretch=1)
        self._enroll_button = QPushButton("🎓 Neo'yu öğret")
        self._enroll_button.setToolTip(
            f"Uyandırma ifadesini {ENROLLMENT_SAMPLES} kez kaydederek sesini öğretir"
        )
        self._enroll_button.clicked.connect(self._on_enroll_clicked)
        self._enroll_button.setEnabled(self._spotter is not None)
        enroll_row.addWidget(self._enroll_button)
        control_layout.addLayout(enroll_row)

        toggle_row = QHBoxLayout()
        self._wake_toggle = QPushButton()
        self._wake_toggle.setCheckable(True)
        self._wake_toggle.clicked.connect(self._on_wake_toggle_clicked)
        toggle_row.addWidget(self._wake_toggle, stretch=1)
        stop_button = QPushButton("⏹ Durdur")
        stop_button.setObjectName("StopButton")
        stop_button.setToolTip("Konuşmayı ve işlemeyi hemen durdur")
        stop_button.clicked.connect(self._on_stop_clicked)
        toggle_row.addWidget(stop_button)
        control_layout.addLayout(toggle_row)

        self._autostart_toggle = QPushButton()
        self._autostart_toggle.setCheckable(True)
        self._autostart_toggle.setToolTip(
            "Windows açıldığında NEO'yu otomatik başlatır (sadece bu kullanıcı için)"
        )
        self._autostart_toggle.clicked.connect(self._on_autostart_toggle_clicked)
        control_layout.addWidget(self._autostart_toggle)
        self._update_autostart_label()

        # Stays hidden until a check actually finds a newer version, so the
        # panel doesn't carry a button that does nothing most of the time.
        self._update_button = QPushButton()
        self._update_button.setObjectName("UpdateButton")
        self._update_button.clicked.connect(self._on_update_clicked)
        self._update_button.hide()
        control_layout.addWidget(self._update_button)

        layout.addWidget(control_panel)
        self._update_enroll_label()
        self._update_wake_toggle_label()

        self._chat_view = ChatView()
        layout.addWidget(self._chat_view, stretch=1)

        input_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("NEO'ya bir şey söyle...")
        self._input.returnPressed.connect(self._on_submit)
        send_button = QPushButton("Gönder")
        send_button.clicked.connect(self._on_submit)
        self._mic_button = QPushButton("🎙")
        self._mic_button.setFixedWidth(44)
        self._mic_button.setToolTip("Basılı tut ve konuş (push-to-talk)")
        self._mic_button.pressed.connect(self._on_mic_press)
        self._mic_button.released.connect(self._on_mic_release)
        input_row.addWidget(self._mic_button)
        input_row.addWidget(self._input, stretch=1)
        input_row.addWidget(send_button)
        layout.addLayout(input_row)

        self.setCentralWidget(root)
        self._append("NEO", "Merhaba, dinliyorum.")

    def _on_info_clicked(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.about(self, "NEO Hakkında", ABOUT_TEXT)

    # -- system tray (stay running in the background) ----------------------

    def _build_tray_icon(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self._tray = None
            return

        self._tray = QSystemTrayIcon(self._app_icon, self)
        self._tray.setToolTip("NEO")

        menu = QMenu()
        show_action = QAction("Göster", self)
        show_action.triggered.connect(self._show_from_tray)
        menu.addAction(show_action)
        quit_action = QAction("Çıkış", self)
        quit_action.triggered.connect(self._quit_from_tray)
        menu.addAction(quit_action)
        self._tray.setContextMenu(menu)

        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _on_tray_activated(self, reason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._show_from_tray()

    def _show_from_tray(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _quit_from_tray(self) -> None:
        self._quitting = True
        self.close()

    async def confirm_action(self, tool_name: str, description: str) -> bool:
        """Wired into PermissionManager as the confirm callback for
        MEDIUM/HIGH risk tools (lock/shutdown/restart computer, etc.) --
        without a real "Yes" click here, PermissionManager denies by
        default, so a risky tool can never run unattended."""
        from PySide6.QtWidgets import QMessageBox

        self._append("NEO", f"Onayın gerekiyor: '{tool_name}'. (Ekrandaki pencereye bak)")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Onay Gerekiyor")
        box.setText(f"NEO şunu yapmak istiyor: '{tool_name}'")
        box.setInformativeText(description or "Bu işlem geri alınamayabilir. Onaylıyor musun?")
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)

        # open() + await, not exec(). exec() spins a nested Qt event loop
        # while this coroutine is suspended, and qasync refuses to run other
        # tasks inside it: the stats timer firing mid-dialog raised
        # "Cannot enter into task ... while another task is being executed"
        # and killed the task that was waiting on the answer. That took down
        # an update mid-install, and the same hazard applied to every
        # MEDIUM/HIGH confirmation (lock, shutdown, calendar delete).
        future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()

        def _resolve(_code: int) -> None:
            if not future.done():
                clicked = box.standardButton(box.clickedButton())
                future.set_result(clicked == QMessageBox.StandardButton.Yes)

        box.finished.connect(_resolve)
        box.open()

        try:
            approved = await future
        except asyncio.CancelledError:
            box.close()
            raise

        self._append("NEO", "Onaylandı." if approved else "Onaylanmadı, işlem iptal edildi.")
        return approved

    def _set_state(self, state: AgentState) -> None:
        self._status_orb.set_state(state)
        self._status_label.setText(STATE_LABELS[state])

    def _release_idle_resources(self) -> None:
        # Never unload mid-conversation or while continuous listening is on.
        if self._stt is None or self._speaking or self._wake_task is not None:
            return
        if self._active_command_task is not None and not self._active_command_task.done():
            return
        self._stt.unload_if_idle()

    # -- updates -----------------------------------------------------------

    async def check_for_update(self, announce_when_current: bool = False) -> None:
        """Looks for a newer release and offers it, without interrupting.

        A failed check is deliberately quiet: no manifest URL configured, no
        internet, or a server hiccup are all normal states for an assistant
        that has to keep working offline, and none of them are worth a
        message the user didn't ask for.
        """
        if not self._update_manifest_url:
            if announce_when_current:
                self._append("NEO", "Güncelleme adresi tanımlı değil.")
            return
        try:
            info = await check_for_update(self._update_manifest_url)
        except UpdateError as exc:
            logger.info("Güncelleme kontrolü başarısız: %s", exc)
            if announce_when_current:
                self._append("NEO", str(exc))
            return

        if info is None:
            logger.info("Güncelleme yok, en son sürüm çalışıyor (%s)", __version__)
            if announce_when_current:
                self._append("NEO", f"En son sürümü kullanıyorsun ({__version__}).")
            return

        self._pending_update = info
        self._update_button.setText(f"⬇ Güncelleme hazır: {info.version}")
        self._update_button.show()
        notes = f" {info.notes}" if info.notes else ""
        self._append(
            "NEO",
            f"Yeni bir sürüm var: {info.version} (şu an {__version__}).{notes} "
            "Kurmak için yukarıdaki güncelleme düğmesine basabilirsin.",
        )

    def _on_update_clicked(self) -> None:
        if self._pending_update is not None:
            asyncio.ensure_future(self._install_update())

    async def _install_update(self) -> None:
        info = self._pending_update
        if info is None:
            return

        # Replacing the installation and restarting is not something to do
        # behind the user's back, so it goes through the same confirmation
        # path as any other high-risk action.
        approved = await self.confirm_action(
            "NEO'yu güncelle",
            f"Sürüm {info.version} indirilip kurulacak ve NEO yeniden başlatılacak.",
        )
        if not approved:
            return

        self._update_button.setEnabled(False)
        self._update_button.setText("İndiriliyor...")
        try:
            package = await download_update(info)
        except UpdateError as exc:
            self._append("NEO", f"Güncelleme kurulamadı: {exc}")
            self._update_button.setEnabled(True)
            self._update_button.setText(f"⬇ Güncelleme hazır: {info.version}")
            return

        self._append("NEO", "Güncelleme doğrulandı, kuruluyor. NEO birazdan yeniden başlayacak.")
        try:
            apply_update(package)
        except Exception:
            logger.exception("Güncelleme uygulanamadı")
            self._append("NEO", "Güncelleme uygulanamadı, mevcut sürüm çalışmaya devam ediyor.")
            self._update_button.setEnabled(True)
            return

        # The helper waits for this process to exit before swapping files,
        # so shutting down properly is part of the update working at all.
        self._shutdown()

    def _shutdown(self) -> None:
        """Ends the process for real.

        QApplication.quit() alone was not enough: under qasync the asyncio
        loop is the one running the show (`loop.run_forever()` in main), so
        Qt quitting left the process alive -- the updater's helper then
        copied over files that were still locked, and NEO sat on screen
        saying it was about to restart while nothing happened.
        """
        self._quitting = True

        # Armed first, and every step below is individually guarded: the
        # helper script is already waiting for this process to exit, so a
        # failure anywhere in the shutdown must not be able to leave NEO
        # running. An earlier version put this last and a NameError on the
        # first line meant it was never scheduled -- NEO closed its window
        # but the process stayed up and the update never applied.
        QTimer.singleShot(SHUTDOWN_GRACE_MS, lambda: os._exit(0))

        for step in (
            self.close,  # closeEvent cleanup: TTS, mic, wake loop
            lambda: QApplication.instance().quit(),
            lambda: asyncio.get_event_loop().call_soon(asyncio.get_event_loop().stop),
        ):
            try:
                step()
            except Exception:
                logger.exception("Kapanış adımı başarısız, çıkışa devam ediliyor")

    # -- autostart ---------------------------------------------------------

    def _update_autostart_label(self) -> None:
        enabled = autostart.is_enabled()
        self._autostart_toggle.setChecked(enabled)
        self._autostart_toggle.setText(
            f"Windows açılışında başlat: {'Açık' if enabled else 'Kapalı'}"
        )

    def _on_autostart_toggle_clicked(self) -> None:
        ok = autostart.disable() if autostart.is_enabled() else autostart.enable()
        if not ok:
            self._append("NEO", "Otomatik başlatma ayarı değiştirilemedi.")
        self._update_autostart_label()

    def _update_research_indicator(self) -> None:
        active = getattr(self._agent, "research_mode", False)
        self._research_label.setText("🔬 ARAŞTIRMA MODU")
        self._research_label.setVisible(bool(active))

    def _append(self, sender: str, text: str) -> None:
        self._chat_view.append(sender, text)

    # -- text chat -----------------------------------------------------

    def _on_submit(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        asyncio.ensure_future(self._handle_recognized_text(text))

    # -- manual stop -------------------------------------------------------

    def _on_stop_clicked(self) -> None:
        if self._tts is not None:
            self._tts.stop()
        if self._active_command_task is not None and not self._active_command_task.done():
            self._active_command_task.cancel()
        self._set_state(AgentState.IDLE)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """The window's [X] button minimizes to the system tray instead of
        quitting -- NEO is meant to keep running in the background (wake
        word, calendar reminders, etc.) rather than stop just because the
        window is out of sight. Use the tray menu's "Çıkış" to actually
        quit, which stops audio/mic activity right away first -- previously,
        closing while NEO was mid-reply left the underlying TTS/mic threads
        running in the background, so it kept talking after the window was
        gone."""
        if self._tray is not None and not self._quitting:
            event.ignore()
            self.hide()
            if not self._tray_notice_shown:
                self._tray_notice_shown = True
                self._tray.showMessage(
                    "NEO arka planda çalışıyor",
                    "Tamamen kapatmak için tepsi simgesine sağ tıklayıp 'Çıkış' seç.",
                    self._app_icon,
                    4000,
                )
            return

        if self._tts is not None:
            self._tts.stop()
        if self._wake_listener is not None:
            self._wake_listener.stop()
        if self._recorder is not None:
            try:
                self._recorder.stop()
            except Exception:
                logger.exception("Recorder could not be stopped on close")
        if self._active_command_task is not None and not self._active_command_task.done():
            self._active_command_task.cancel()
        if self._tray is not None:
            self._tray.hide()
        super().closeEvent(event)

    # -- push-to-talk ----------------------------------------------------

    def _on_mic_press(self) -> None:
        if self._recorder is None:
            self._append("NEO", "Mikrofon bu sürümde yapılandırılmadı.")
            return
        try:
            self._recorder.start()
        except MicrophoneUnavailableError as exc:
            self._append("NEO", str(exc))
            self._set_state(AgentState.ERROR)
            return
        self._set_state(AgentState.LISTENING)

    def _on_mic_release(self) -> None:
        if self._recorder is None:
            return
        audio = self._recorder.stop()
        if audio.size == 0:
            self._set_state(AgentState.IDLE)
            return
        asyncio.ensure_future(self._transcribe_and_handle(audio))

    async def _transcribe_and_handle(self, audio) -> None:
        if self._stt is None:
            self._append("NEO", "Ses tanıma bu sürümde yapılandırılmadı.")
            self._set_state(AgentState.IDLE)
            return

        self._set_state(AgentState.PROCESSING)

        min_voiced = int(PTT_MIN_VOICED_SECONDS * MIC_SAMPLE_RATE)
        heard = has_enough_speech(audio, PTT_ENERGY_THRESHOLD, min_voiced)
        if heard:
            # Loudness alone can't tell a spoken word from a key press, so
            # confirm with a real voice-activity detector before handing
            # anything to Whisper -- which hallucinates fluent, fabricated
            # text when fed audio that turns out not to contain speech.
            detected = await asyncio.to_thread(speech_seconds, audio, MIC_SAMPLE_RATE)
            heard = detected < 0.0 or detected >= PTT_MIN_SPEECH_SECONDS
        if not heard:
            self._append("NEO", "Bir şey duyamadım, tekrar dener misin?")
            self._set_state(AgentState.IDLE)
            return

        try:
            text = await self._stt.transcribe(audio)
        except STTUnavailableError as exc:
            self._append("NEO", str(exc))
            self._set_state(AgentState.ERROR)
            return

        if not text:
            self._append("NEO", "Bir şey duyamadım, tekrar dener misin?")
            self._set_state(AgentState.IDLE)
            return

        await self._handle_recognized_text(text)

    # -- wake word enrollment ("Neo'yu öğret") ------------------------------

    def _update_enroll_label(self) -> None:
        if self._spotter is None:
            self._enroll_label.setText("Uyandırma kelimesi: kullanılamıyor")
            return
        count = self._spotter.template_count
        if count >= ENROLLMENT_SAMPLES:
            # "Not calibrated" covers two different situations and the fix
            # differs, so say which one it is rather than making the user
            # guess why NEO keeps waking up.
            if self._spotter.is_calibrated:
                suffix = f" — ayrışma {self._spotter.separation:.1f}"
            elif self._spotter.negative_count == 0:
                suffix = " — kalibrasyon örneği yok"
            else:
                suffix = " — ayrışma zayıf, tekrar öğret"
            self._enroll_label.setText(f"'{self._wake_phrase}' öğretildi ({count} örnek){suffix}")
        else:
            self._enroll_label.setText(
                f"'{self._wake_phrase}' öğretilmedi ({count}/{ENROLLMENT_SAMPLES})"
            )

    def _on_enroll_clicked(self) -> None:
        if self._spotter is None or self._enrolling or self._recorder is None:
            return
        asyncio.ensure_future(self._run_enrollment())

    async def _run_enrollment(self) -> None:
        self._enrolling = True
        self._enroll_button.setEnabled(False)
        if self._wake_task is not None:
            self._stop_wake_loop()
        try:
            # Start from nothing: enrolling on top of earlier takes silently
            # mixes two different phrases into one template set, which blows
            # the calibration apart (a real case left three "Neo" takes
            # alongside three "Neo uyan" ones, pushing the spread to 50.4 and
            # leaving a threshold that accepted ordinary speech).
            self._spotter.clear()
            self._update_enroll_label()
            self._append(
                "NEO",
                f"Hadi başlayalım: {ENROLLMENT_SAMPLES} kere '{self._wake_phrase}' "
                "diyeceksin. Her seferinde aynı hızda ve doğal söyle.",
            )
            for i in range(1, ENROLLMENT_SAMPLES + 1):
                enrolled = False
                for attempt in range(1, ENROLLMENT_MAX_RETRIES + 1):
                    self._append(
                        "NEO",
                        f"Örnek {i}/{ENROLLMENT_SAMPLES}: şimdi '{self._wake_phrase}' de...",
                    )
                    self._set_state(AgentState.LISTENING)
                    await asyncio.sleep(ENROLLMENT_PREPARE_SECONDS)
                    try:
                        self._recorder.start()
                    except MicrophoneUnavailableError as exc:
                        self._append("NEO", str(exc))
                        self._set_state(AgentState.ERROR)
                        return
                    await asyncio.sleep(ENROLLMENT_RECORD_SECONDS)
                    audio = self._recorder.stop()
                    enrolled = audio.size > 0 and self._spotter.enroll(audio)
                    if enrolled:
                        break
                    self._append("NEO", "Pek bir şey duyamadım, bu örneği tekrar alalım.")
                if not enrolled:
                    self._append("NEO", "Bu örneği bir türlü alamadım, mikrofonunu kontrol edip tekrar dener misin?")
                    self._set_state(AgentState.ERROR)
                    return
                self._update_enroll_label()

            if not await self._enroll_negative_speech():
                return

            # Show the measured phrase length: when enrollment silently
            # captured 2.1s of breath and room noise around a ~1s phrase,
            # nothing on screen revealed it and the wake word just "felt
            # unreliable" for days.
            duration = self._spotter.enrolled_duration
            detail = f" Ölçülen ifade uzunluğu: {duration:.1f} sn." if duration else ""

            if self._spotter.is_calibrated:
                self._append(
                    "NEO",
                    "Öğrenme tamamlandı! Sesin diğer konuşmandan net ayrışıyor "
                    f"(ayrışma payı {self._spotter.separation:.1f}).{detail} "
                    "Artık sürekli dinlemeyi açabilirsin.",
                )
            else:
                # Never claim this worked when the numbers say it didn't --
                # the user would spend the evening wondering why NEO keeps
                # waking up mid-sentence.
                self._append(
                    "NEO",
                    f"Öğrenme tamamlandı ama '{self._wake_phrase}' ile normal "
                    "konuşman yeterince ayrışmadı "
                    f"(ayrışma {self._spotter.separation:.1f}, pozitif olmalı)."
                    f"{detail} Yanlış tetiklenme olabilir.",
                )
            self._set_state(AgentState.IDLE)
        finally:
            self._enrolling = False
            self._enroll_button.setEnabled(True)
            self._update_wake_toggle_label()

    async def _enroll_negative_speech(self) -> bool:
        """Records a stretch of ordinary talking so the detector learns what
        *isn't* the wake word.

        Without this the match threshold was derived only from how much the
        user's own "Neo" takes vary, which says nothing about how close their
        normal speech gets -- and it turned out to get very close, so NEO
        woke up on ordinary conversation.
        """
        self._append(
            "NEO",
            f"Son adım: {NEGATIVE_RECORD_SECONDS:.0f} saniye boyunca normal konuş — "
            f"aklına ne gelirse, ama '{self._wake_phrase}' deme. Bu, seni "
            "dinlerken hangi seslerin uyandırma ifadesi olmadığını öğreniyor.",
        )
        self._set_state(AgentState.LISTENING)
        await asyncio.sleep(ENROLLMENT_PREPARE_SECONDS)
        try:
            self._recorder.start()
        except MicrophoneUnavailableError as exc:
            self._append("NEO", str(exc))
            self._set_state(AgentState.ERROR)
            return False

        await asyncio.sleep(NEGATIVE_RECORD_SECONDS)
        audio = self._recorder.stop()
        added = self._spotter.enroll_negative(audio) if audio.size else 0
        if added == 0:
            self._append(
                "NEO",
                "Bu kayıtta konuşma duyamadım, o yüzden ayrıştırmayı kalibre "
                "edemedim. Öğretmeyi tekrar çalıştırıp bu adımda biraz "
                "konuşursan yanlış tetiklenmeler ciddi şekilde azalır.",
            )
            self._set_state(AgentState.IDLE)
            return False
        logger.info("Uyandırma kalibrasyonu: %d olumsuz pencere kaydedildi", added)
        return True

    # -- wake word ("Neo") -------------------------------------------------

    def _update_wake_toggle_label(self) -> None:
        if self._wake_listener is None:
            self._wake_toggle.setText("Sürekli dinleme: kullanılamıyor")
            self._wake_toggle.setEnabled(False)
            return
        ready = self._spotter is not None and self._spotter.has_enough_templates()
        self._wake_toggle.setEnabled(ready and not self._enrolling)
        active = self._wake_task is not None
        self._wake_toggle.setChecked(active)
        if not ready:
            self._wake_toggle.setText("Sürekli dinleme: önce sesini öğret")
        else:
            self._wake_toggle.setText(f"Sürekli dinleme: {'Açık' if active else 'Kapalı'}")

    def _on_wake_toggle_clicked(self) -> None:
        if self._wake_task is not None:
            self._stop_wake_loop()
        else:
            self._start_wake_loop()

    def set_listening(self, active: bool) -> None:
        """Entry point for voice control ("dinlemeyi durdur"). Runs on the
        GUI thread -- the agent is driven by qasync's loop, which is the Qt
        loop -- so touching widgets here is safe."""
        if active:
            self._start_wake_loop()
        else:
            self._stop_wake_loop()

    def _start_wake_loop(self) -> None:
        if self._wake_listener is None or self._wake_task is not None:
            return
        if self._spotter is None or not self._spotter.has_enough_templates():
            return
        self._wake_task = asyncio.ensure_future(self._run_wake_loop())
        self._update_wake_toggle_label()

    def _stop_wake_loop(self) -> None:
        if self._wake_listener is not None:
            self._wake_listener.stop()
        self._wake_task = None
        self._update_wake_toggle_label()
        if self._status_label.text() != STATE_LABELS[AgentState.ERROR]:
            self._set_state(AgentState.IDLE)

    async def _run_wake_loop(self) -> None:
        try:
            await self._wake_listener.run(
                self._on_wake_detected, self._on_wake_command, is_muted=lambda: self._speaking
            )
        except WakeWordUnavailableError as exc:
            self._append("NEO", str(exc))
            self._set_state(AgentState.ERROR)
        finally:
            self._wake_task = None
            self._update_wake_toggle_label()

    async def _on_wake_detected(self) -> None:
        asyncio.ensure_future(play_activation_chime())
        if self._active_command_task is not None and not self._active_command_task.done():
            self._active_command_task.cancel()
            self._append("NEO", "(cevap kesildi)")
        # Waking up is not a chat message: the chime and the orb switching to
        # "Dinleniyor..." already say it. Writing "Dinliyorum." into the
        # transcript every time meant a stretch of false wake-ups buried the
        # real conversation under a wall of identical bubbles.
        self._set_state(AgentState.LISTENING)

    async def _on_wake_command(self, text: str) -> None:
        self._active_command_task = asyncio.current_task()
        await self._handle_recognized_text(text)

    # -- shared: recognized speech -> agent -> (optional) speech reply ----

    async def _handle_recognized_text(self, text: str) -> None:
        self._append("Sen", text)
        self._set_state(AgentState.PROCESSING)
        try:
            reply = await self._agent.handle_message(text)
        except asyncio.CancelledError:
            raise
        except Exception:
            reply = "Beklenmeyen bir hata oluştu, lütfen tekrar dener misin?"
            self._append("NEO", reply)
            self._set_state(AgentState.ERROR)
            return
        self._append("NEO", reply)
        self._update_research_indicator()

        if self._tts is not None:
            self._set_state(AgentState.SPEAKING)
            self._speaking = True
            try:
                # Research-mode reports are long; only the key-points summary
                # gets read aloud while the full text stays on screen. Emoji
                # are stripped from the spoken version only -- the bubble
                # above still shows them.
                spoken = strip_speech_noise(extract_spoken_summary(reply))
                if spoken:
                    await self._tts.speak(spoken)
            except TTSUnavailableError as exc:
                self._append("NEO", str(exc))
            except asyncio.CancelledError:
                self._tts.stop()
                raise
            finally:
                self._speaking = False

        self._set_state(AgentState.IDLE)
