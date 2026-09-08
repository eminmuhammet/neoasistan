from __future__ import annotations

import asyncio
import logging
import math
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
from ..core.access_mode import AccessMode
from ..core.agent import Agent, extract_spoken_summary
from ..memory.audit_store import AuditStore
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
from .neuro_visual import NeuroVisual
from .stats_panel import StatsPanel
from .theme import DARK_QSS

logger = logging.getLogger(__name__)

def build_about_text(wake_phrase: str = "Neo uyan") -> str:
    """Assembled from the live values rather than written out by hand.

    The version used to be a literal "v0.1.0" in this text. After an update
    actually installed and NEO restarted at 0.1.5, this dialog still said
    0.1.0 -- so the one screen a user checks to confirm an update reported
    that nothing had happened.
    """
    return f"""<h3>NEO</h3>
<p>Windows için kişisel yapay zekâ masaüstü asistanı. Doğal dille konuş,
gerisini o halleder.</p>

<p><b>Sesli kontrol</b><br>
🎙 düğmesini basılı tutup konuş, ya da "🎓 Neo'yu öğret" ile sesini
öğretip "{wake_phrase}" diyerek uyandır. Uyandıktan sonra komutunu
söylemen yeterli, tuşa basman gerekmez.</p>

<p><b>Bilgisayarında yapabildiklerim</b><br>
• Saat, tarih<br>
• CPU / RAM / GPU / disk / ağ durumu ve sistem bilgisi<br>
• Uygulama ve web sitesi açma, klasör açma, dosya bulma<br>
• Belge okuma; klasör indeksleyip içerikte arama<br>
• Ses seviyesi, sessize alma, medya oynat/duraklat<br>
• Ekran görüntüsü alma<br>
• Bilgisayarı kilitleme, yeniden başlatma, kapatma</p>

<p><b>Günlük işler</b><br>
• Takvim notu ekleme, okuma, güncelleme, silme (Google Takvim ile eşitlenir)<br>
• Gmail: son postaları okuma, taslak oluşturma, e-posta gönderme<br>
• Hava durumu<br>
• Tercihlerini hatırlama ("beni sabah 8'de uyandır" gibi şeyleri saklar)<br>
• Görev zamanlama, zamanlanmışları listeleme ve iptal etme<br>
• Çok adımlı görevleri planlayıp yürütme<br>
• Yaptıklarımın kaydını sana gösterme</p>

<p><b>Yetki kipleri</b><br>
Normal kipte yalnızca güvenli işleri yaparım. Fare/klavye kontrolü gibi
yüksek riskli işler "yardımcı kipi" gerektirir ve her çağrıda ayrıca
onayını isterim. Otomasyonu her an
<b>Ctrl+Alt+Shift+Q</b> ile durdurabilirsin.</p>

<p><b>İnternet gerektirenler</b><br>
Genel sohbet, web araştırması ve karmaşık istekler. Saat, sistem durumu,
uygulama açma gibi işler internetsiz de çalışır. Konuşma tanıma ve
uyandırma kelimesi tamamen bilgisayarında, çevrimdışı çalışır.</p>

<p style="color:#7c8b98;">Sürüm {__version__} · Emin İLHAN</p>
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
    AgentState.IDLE:       "●  HAZIR",
    AgentState.LISTENING:  "◉  DİNLİYOR",
    AgentState.PROCESSING: "◈  İŞLİYOR",
    AgentState.SPEAKING:   "◎  KONUŞUYOR",
    AgentState.ERROR:      "✕  HATA",
}

_STATE_LABEL_COLOR = {
    AgentState.IDLE:       "#35e08a",
    AgentState.LISTENING:  "#6effa8",
    AgentState.PROCESSING: "#f2b134",
    AgentState.SPEAKING:   "#35e08a",
    AgentState.ERROR:      "#e05050",
}


class MainWindow(QMainWindow):
    _LEFT_PANEL_COMPACT_WIDTH = 340
    _LEFT_PANEL_FOCUS_WIDTH = 520

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
        self._voice_phase = 0.0
        self._voice_timer = QTimer(self)
        self._voice_timer.setInterval(33)
        self._voice_timer.timeout.connect(self._update_voice_visual)
        self._tray_notice_shown = False
        self._tray: QSystemTrayIcon | None = None
        self._app_icon = build_app_icon()
        self.setWindowTitle("NEO")
        self.setWindowIcon(self._app_icon)
        self._apply_window_size()
        self.setStyleSheet(DARK_QSS)
        self._build_ui()
        # Delay tray creation until the event loop is running — creating
        # QSystemTrayIcon before the first event loop tick causes it to
        # silently fail to register on some Windows configurations.
        QTimer.singleShot(500, self._build_tray_icon)
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

    def _apply_window_size(self) -> None:
        self.setMinimumSize(1000, 680)
        self.showMaximized()

    def _build_ui(self) -> None:
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # ── Header bar (full width) ───────────────────────────────────────
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(28, 14, 28, 8)
        header_layout.setSpacing(10)

        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        title = QLabel("N E O")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle = QLabel("KİŞİSEL YAPAY ZEKA ASİSTANI")
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_col.addWidget(title)
        title_col.addWidget(subtitle)

        self._access_mode_label = QLabel()
        self._access_mode_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self._access_mode_label.hide()

        header_layout.addWidget(self._access_mode_label)
        header_layout.addStretch(1)
        header_layout.addLayout(title_col)
        header_layout.addStretch(1)
        self._chat_toggle_button = QPushButton("💬")
        self._chat_toggle_button.setObjectName("InfoButton")
        self._chat_toggle_button.setCheckable(True)
        self._chat_toggle_button.setChecked(False)
        self._chat_toggle_button.setToolTip("Sohbet panelini göster/gizle")
        self._chat_toggle_button.clicked.connect(self._on_chat_toggle_clicked)
        header_layout.addWidget(self._chat_toggle_button)
        outer_layout.addWidget(header)

        # ── Content row: [left filler] [sphere] [chat panel] ─────────────
        content = QWidget()
        self._root_layout = content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Left panel — subtitle area when chat is closed, balances sphere centre
        self._left_filler = QWidget()
        left_layout = QVBoxLayout(self._left_filler)
        left_layout.setContentsMargins(24, 0, 24, 0)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._subtitle_label = QLabel()
        self._subtitle_label.setObjectName("SpeechSubtitle")
        self._subtitle_label.setWordWrap(True)
        self._subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._subtitle_label.hide()
        left_layout.addWidget(self._subtitle_label)
        content_layout.addWidget(self._left_filler, stretch=1)

        # Sphere column (always focus-size, always centred)
        sphere_col = QWidget()
        self._left_panel = sphere_col          # kept for compat refs
        sphere_col_layout = QVBoxLayout(sphere_col)
        sphere_col_layout.setContentsMargins(0, 0, 0, 0)
        sphere_col_layout.setSpacing(8)
        sphere_col_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._status_orb = NeuroVisual()
        self._status_orb.set_focus(True)       # always 460 px
        sphere_col_layout.addWidget(self._status_orb, alignment=Qt.AlignmentFlag.AlignCenter)

        self._status_label = QLabel()
        self._status_label.setObjectName("StatusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sphere_col_layout.addWidget(self._status_label)

        self._research_label = QLabel()
        self._research_label.setObjectName("ResearchLabel")
        self._research_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._research_label.hide()
        sphere_col_layout.addWidget(self._research_label)

        self._task_label = QLabel()
        self._task_label.setObjectName("StatusLabel")
        self._task_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._task_label.setStyleSheet("color: #6effa8; font-size: 11px;")
        self._task_label.hide()
        sphere_col_layout.addWidget(self._task_label)

        content_layout.addWidget(sphere_col, stretch=0)

        # Right filler — mirrors left filler so sphere stays centred when chat hidden
        self._right_filler = QWidget()
        content_layout.addWidget(self._right_filler, stretch=1)

        # Right: chat panel (hidden by default, replaces right filler space)
        self._right_panel = right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 0, 20, 0)
        right_layout.setSpacing(12)
        self._chat_view = ChatView()
        right_layout.addWidget(self._chat_view, stretch=1)
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
        right_layout.addLayout(input_row)
        right_panel.hide()
        content_layout.addWidget(right_panel, stretch=1)

        outer_layout.addWidget(content, stretch=1)

        # ── Control bar (full width, all controls in one horizontal row) ──
        control_bar = QWidget()
        control_bar.setObjectName("ControlPanel")
        ctrl = QHBoxLayout(control_bar)
        ctrl.setContentsMargins(28, 10, 28, 10)
        ctrl.setSpacing(14)

        # — Wake word section —
        self._enroll_label = QLabel()
        self._enroll_label.setObjectName("StatusLabel")
        self._enroll_label.setWordWrap(False)
        ctrl.addWidget(self._enroll_label, stretch=2)

        self._enroll_button = QPushButton("🎓 Neo'yu öğret")
        self._enroll_button.setToolTip(
            f"Uyandırma ifadesini {ENROLLMENT_SAMPLES} kez kaydederek sesini öğretir"
        )
        self._enroll_button.clicked.connect(self._on_enroll_clicked)
        self._enroll_button.setEnabled(self._spotter is not None)
        ctrl.addWidget(self._enroll_button)

        ctrl.addSpacing(24)

        # — Continuous listen + stop —
        self._wake_toggle = QPushButton()
        self._wake_toggle.setCheckable(True)
        self._wake_toggle.clicked.connect(self._on_wake_toggle_clicked)
        ctrl.addWidget(self._wake_toggle, stretch=2)

        stop_button = QPushButton("Durdur")
        stop_button.setObjectName("StopButton")
        stop_button.setToolTip("Konuşmayı ve işlemeyi hemen durdur")
        stop_button.clicked.connect(self._on_stop_clicked)
        ctrl.addWidget(stop_button)

        ctrl.addSpacing(24)

        # — Autostart —
        self._autostart_toggle = QPushButton()
        self._autostart_toggle.setCheckable(True)
        self._autostart_toggle.setToolTip(
            "Windows açıldığında NEO'yu otomatik başlatır (sadece bu kullanıcı için)"
        )
        self._autostart_toggle.clicked.connect(self._on_autostart_toggle_clicked)
        # No stretch: a set-once preference should not be the widest, loudest
        # thing on the bar. The voice controls get the room instead.
        ctrl.addWidget(self._autostart_toggle)

        self._update_button = QPushButton()
        self._update_button.setObjectName("UpdateButton")
        self._update_button.clicked.connect(self._on_update_clicked)
        self._update_button.hide()
        ctrl.addWidget(self._update_button)

        # ── Notification toast (above control bar, auto-dismiss) ─────────
        self._notif_label = QLabel()
        self._notif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._notif_label.setStyleSheet(
            "color: #d9ece0; background-color: #0f2a1e; border: 1px solid #35e08a44;"
            "border-radius: 8px; padding: 6px 16px; font-size: 12px;"
        )
        self._notif_label.hide()
        outer_layout.addWidget(self._notif_label)

        outer_layout.addWidget(control_bar)

        # ── Stats footer (full width) ─────────────────────────────────────
        stats_footer = QWidget()
        stats_footer.setObjectName("StatsFooter")
        sf_layout = QHBoxLayout(stats_footer)
        sf_layout.setContentsMargins(28, 8, 28, 10)
        sf_layout.setSpacing(0)
        stats_panel = StatsPanel()
        stats_panel.setObjectName("StatsPanel")
        sf_layout.addStretch(1)
        sf_layout.addWidget(stats_panel)
        sf_layout.addStretch(1)
        outer_layout.addWidget(stats_footer)

        self._update_enroll_label()
        self._update_wake_toggle_label()
        self._update_autostart_label()

        self.setCentralWidget(outer)
        self._append("NEO", "Merhaba, dinliyorum.")

    def _on_chat_toggle_clicked(self) -> None:
        """Show/hide the conversation column; sphere stays centred when hidden."""
        show_chat = self._chat_toggle_button.isChecked()
        self._right_panel.setVisible(show_chat)
        self._right_filler.setVisible(not show_chat)
        self._root_layout.setStretchFactor(self._left_filler, 1)
        self._root_layout.setStretchFactor(self._right_filler, 1 if not show_chat else 0)
        self._root_layout.setStretchFactor(self._right_panel, 1 if show_chat else 0)
        # Hide subtitle when chat opens (it's already visible in chat)
        if show_chat:
            self._subtitle_label.hide()

    def _on_info_clicked(self) -> None:
        # QMessageBox.about() calls exec() internally, which spins a nested
        # Qt event loop. qasync refuses to run other tasks inside that loop,
        # so clicking this while a voice command was in flight raised
        # "Cannot enter into task ... while another task is being executed"
        # and killed whatever task was running -- observed live destroying
        # the task handling a Claude reply, which is why TTS never spoke it.
        # This was the same hazard confirm_action had, fixed the same way:
        # a non-modal dialog via open() instead of a blocking exec().
        from PySide6.QtWidgets import QMessageBox

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("NEO Hakkında")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(build_about_text(self._wake_phrase))
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        # Nothing awaits this dialog's answer, so it only needs to not block
        # the loop -- open() and letting it clean itself up is enough.
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        box.open()

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
        self._status_label.setStyleSheet(
            f"color: {_STATE_LABEL_COLOR[state]};"
            "font-size: 11px; font-weight: 700; letter-spacing: 3px;"
        )
        if state == AgentState.SPEAKING:
            self._voice_phase = 0.0
            self._voice_timer.start()
        else:
            self._voice_timer.stop()
            self._status_orb.set_audio_level(0.0)

    def _update_voice_visual(self) -> None:
        """Generate a speech-like amplitude envelope and feed it to the orb."""
        import math as _m, random as _r
        # Sentence envelope (slow) × syllable pulse (fast) × noise
        self._voice_phase += 0.033
        p = self._voice_phase
        level = (
            (0.55 + 0.45 * _m.sin(p * 1.3)) *      # sentence envelope
            (0.50 + 0.50 * abs(_m.sin(p * 8.7))) *  # syllable pulses
            _r.uniform(0.70, 1.00)                   # breath noise
        )
        self._status_orb.set_audio_level(float(level))

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
            f"⚙ Açılışta başlat: {'Açık' if enabled else 'Kapalı'}"
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
        # Show NEO's reply as subtitle on the left when chat panel is closed
        if sender == "NEO" and not self._chat_toggle_button.isChecked():
            # Trim to ~200 chars so it doesn't flood the panel
            display = text if len(text) <= 200 else text[:197] + "…"
            self._subtitle_label.setText(display)
            self._subtitle_label.show()

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
                # The number alone told the user nothing actionable, and
                # "pozitif olmalı" was outright wrong once the bar became a
                # fraction of the spread rather than zero: a separation of
                # 5.3 is positive and still far too small. Say what the two
                # numbers mean and what actually helps.
                spread = getattr(self._spotter, "_spread", None)
                measured = (
                    f"(ayrışma {self._spotter.separation:.1f}, "
                    f"kendi kayıtlarının yayılımı {spread:.1f})"
                    if spread else f"(ayrışma {self._spotter.separation:.1f})"
                )
                self._append(
                    "NEO",
                    f"Öğrenme tamamlandı ama '{self._wake_phrase}' ile normal "
                    f"konuşman yeterince ayrışmadı {measured}.{detail} "
                    "Uyandırma yine de çalışır — kararı ses tanıma veriyor — "
                    "ama yanlış tetiklenme olabilir. Tekrar öğretirken "
                    "ifadeyi her seferinde aynı hız ve tonda söylemek "
                    "ayrışmayı en çok artıran şey.",
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
        if self._active_command_task is not None and not self._active_command_task.done():
            self._active_command_task.cancel()
            self._append("NEO", "(cevap kesildi)")
        # Waking up is not a chat message: the cue and the orb switching to
        # "Dinleniyor..." already say it. Writing "Dinliyorum." into the
        # transcript every time meant a stretch of false wake-ups buried the
        # real conversation under a wall of identical bubbles.
        self._set_state(AgentState.LISTENING)

        # Muted while the cue plays: it is speech now, and the wake loop
        # would otherwise capture NEO's own "Dinliyorum efendim" and hand it
        # straight back as the user's command.
        self._speaking = True
        try:
            await play_activation_chime()
        finally:
            self._speaking = False

    async def _on_wake_command(self, text: str) -> None:
        self._active_command_task = asyncio.current_task()
        if not text:
            # Komut VAD tarafından reddedildi — LISTENING'den IDLE'a dön
            self._set_state(AgentState.IDLE)
            return
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
            self._speaking = True
            try:
                # Research-mode reports are long; only the key-points summary
                # gets read aloud while the full text stays on screen. Emoji
                # are stripped from the spoken version only -- the bubble
                # above still shows them.
                spoken = strip_speech_noise(extract_spoken_summary(reply))
                if spoken:
                    # Set SPEAKING state right before audio starts, not before
                    # text processing — avoids the animation playing during the
                    # strip/extract phase when nothing is audible yet.
                    self._set_state(AgentState.SPEAKING)
                    await self._tts.speak(spoken)
            except TTSUnavailableError as exc:
                self._append("NEO", str(exc))
            except asyncio.CancelledError:
                self._tts.stop()
                raise
            finally:
                self._speaking = False

        self._set_state(AgentState.IDLE)
        # Hide subtitle a moment after speech ends so the last words are readable
        QTimer.singleShot(3000, self._subtitle_label.hide)

    # ── Backend integration API ────────────────────────────────────────────

    def set_access_mode(self, mode: AccessMode) -> None:
        """Show/update the current access-mode badge in the header."""
        label = self._access_mode_label
        if mode is AccessMode.HELPER:
            label.setText("● YARDIMCI MOD")
            label.setStyleSheet("color: #f2b134; font-size: 10px; font-weight: 700; letter-spacing: 2px;")
        else:
            label.setText("● ASİSTAN MOD")
            label.setStyleSheet("color: #35e08a; font-size: 10px; font-weight: 700; letter-spacing: 2px;")
        label.setVisible(True)

    def set_task_progress(self, task_name: str, step: int, total: int, status: str = "") -> None:
        """Show a one-line task progress indicator below the sphere."""
        if total <= 0 or step >= total:
            self._task_label.setVisible(False)
            return
        pct = math.floor(100 * step / total)
        text = f"◈  {task_name}  {step}/{total} ({pct}%)"
        if status:
            text += f"  — {status}"
        self._task_label.setText(text)
        self._task_label.setVisible(True)

    def show_proactive_notification(self, text: str, duration_ms: int = 4000) -> None:
        """Toast a short notification above the control bar, auto-dismiss after duration_ms."""
        self._notif_label.setText(text)
        self._notif_label.setVisible(True)
        QTimer.singleShot(duration_ms, lambda: self._notif_label.setVisible(False))

    def set_audit_store(self, audit_store: AuditStore) -> None:
        """Receive the audit store reference post-construction."""
        self._audit_store = audit_store
