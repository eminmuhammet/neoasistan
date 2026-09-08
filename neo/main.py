from __future__ import annotations

import asyncio
import logging
import sys

from PySide6.QtWidgets import QApplication
from qasync import QEventLoop

from .config.credentials import PasswordStore
from .config.settings import Settings, load_settings
from .core.access_mode import AccessModeManager
from .core.agent import Agent
from .core.local_llm_client import LocalLLMClient
from .core.automation_panic import register_panic_hotkey
from .core.disk_watch import DiskSpaceWatcher
from .core.permissions import PermissionManager
from .core.planner import TaskPlanner
from .core.scheduler import Scheduler
from .logging_setup import setup_logging
from .memory.audit_store import AuditStore
from .memory.calendar_store import CalendarStore
from .memory.conversation_store import ConversationStore
from .memory.document_index import DocumentIndex
from .memory.preference_store import PreferenceStore
from .memory.scheduled_job_store import ScheduledJobStore
from .memory.task_store import TaskStore
from .tools.applications import OpenApplicationTool, OpenWebsiteTool, PlayOnSpotifyTool
from .tools.audit import GetRecentActivityTool
from .tools.base import ToolRegistry
from .tools.calendar import (
    AddCalendarNoteTool,
    DeleteCalendarNoteTool,
    GetCalendarNotesTool,
    UpdateCalendarNoteTool,
)
from .tools.document_search import IndexFolderTool, ReadDocumentTool, SearchDocumentsTool
from .tools.filesystem import FindFileTool, OpenFolderTool
from .tools.media import GetVolumeTool, MediaControlTool, SetMuteTool, SetVolumeTool
from .tools.gmail import CreateEmailDraftTool, ReadRecentEmailsTool, SendEmailTool
from .tools.gmail_client import GmailClient
from .tools.google_calendar import GoogleCalendarSync
from .tools.preferences import (
    ForgetPreferenceTool,
    RecallPreferencesTool,
    RememberPreferenceTool,
)
from .tools.computer_control import (
    ClickTool,
    DragTool,
    MoveMouseTool,
    PressKeysTool,
    TypeTextTool,
)
from .tools.planning import RunTaskTool
from .tools.power import LockComputerTool, RestartComputerTool, ShutdownComputerTool
from .tools.scheduling import (
    CancelScheduledTaskTool,
    ListScheduledTasksTool,
    ScheduleTaskTool,
)
from .tools.screen import CaptureScreenTool
from .tools.system_info import (
    GetCpuUsageTool,
    GetDiskUsageTool,
    GetGpuStatusTool,
    GetNetworkStatusTool,
    GetRamUsageTool,
    GetSystemInfoTool,
)
from .tools.time_tools import GetDateTool, GetTimeTool
from .tools.weather import GetWeatherTool
from .tools.web_search import WebSearchTool
from .tools.fetch_page import FetchPageTool
from .ui.main_window import MainWindow
from .ui.mode_unlock_dialog import request_helper_mode_unlock
from .voice import chime
from .voice.keyword_spotter import KeywordSpotter
from .voice.microphone import PushToTalkRecorder
from .voice.stt import WhisperSTT
from .voice.tts import EdgeTTS, FallbackTTS, SapiTTS
from .voice.wake_word import WakeWordListener

logger = logging.getLogger(__name__)


def build_registry(settings: Settings) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(GetTimeTool())
    registry.register(GetDateTool())
    registry.register(OpenApplicationTool())
    registry.register(OpenWebsiteTool())
    registry.register(PlayOnSpotifyTool())
    registry.register(GetSystemInfoTool())
    registry.register(GetCpuUsageTool())
    registry.register(GetRamUsageTool())
    registry.register(GetDiskUsageTool())
    registry.register(GetGpuStatusTool())
    registry.register(GetNetworkStatusTool())
    registry.register(GetWeatherTool(settings.default_city))
    registry.register(WebSearchTool())
    registry.register(FetchPageTool())

    calendar_store = CalendarStore(settings.data_dir / "calendar.db")
    google_sync = GoogleCalendarSync(
        credentials_path=settings.data_dir.parent / "credentials.json",
        token_path=settings.data_dir / "google_token.json",
    )
    registry.register(AddCalendarNoteTool(calendar_store, google_sync=google_sync))
    registry.register(GetCalendarNotesTool(calendar_store))
    registry.register(DeleteCalendarNoteTool(calendar_store, google_sync=google_sync))
    registry.register(UpdateCalendarNoteTool(calendar_store, google_sync=google_sync))

    # Same credentials.json as calendar sync (same Google Cloud app/client
    # id), but its own token file -- the calendar token was authorized for
    # calendar scopes only and would fail with insufficient-scope if reused
    # here (see NEO_V2_PLAN.md item 11).
    gmail_client = GmailClient(
        credentials_path=settings.data_dir.parent / "credentials.json",
        token_path=settings.data_dir / "gmail_token.json",
    )
    registry.register(ReadRecentEmailsTool(gmail_client))
    registry.register(CreateEmailDraftTool(gmail_client))
    registry.register(SendEmailTool(gmail_client))

    registry.register(FindFileTool())
    registry.register(OpenFolderTool())
    registry.register(ReadDocumentTool())

    document_index = DocumentIndex(settings.data_dir / "document_index.db")
    registry.register(IndexFolderTool(document_index))
    registry.register(SearchDocumentsTool(document_index))

    registry.register(GetVolumeTool())
    registry.register(SetVolumeTool())
    registry.register(SetMuteTool())
    registry.register(MediaControlTool())

    registry.register(LockComputerTool())
    registry.register(ShutdownComputerTool())
    registry.register(RestartComputerTool())
    registry.register(CaptureScreenTool())

    # HIGH risk (see NEO_V2_PLAN.md item 7): PermissionManager already
    # refuses HIGH-risk tools outright while in ASSISTANT mode, so these
    # only ever run in helper mode, and even there each call still asks
    # for the user's confirmation -- the tool's raw input (coordinates,
    # keys, text) is exactly what the confirmation dialog previews.
    registry.register(ClickTool())
    registry.register(MoveMouseTool())
    registry.register(DragTool())
    registry.register(TypeTextTool())
    registry.register(PressKeysTool())
    return registry


def _acquire_single_instance_lock():
    """Windows named-mutex ile tek örnek garantisi.
    İkinci başlatma girişimi varolan pencereyi öne getirir ve çıkar."""
    import ctypes
    mutex = ctypes.windll.kernel32.CreateMutexW(None, True, "Global\\NEO_SingleInstance_v1")
    if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        return None
    return mutex  # tutulmazsa GC mutex'i serbest bırakır


def main() -> int:
    _mutex = _acquire_single_instance_lock()
    if _mutex is None:
        # Zaten çalışıyor — sadece çık
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0,
            "NEO zaten çalışıyor. Sistem tepsisindeki simgeye tıklayın.",
            "NEO",
            0x40,  # MB_ICONINFORMATION
        )
        return 0

    settings = load_settings()
    setup_logging(settings.log_dir, settings.log_level)
    logger.info("NEO başlatılıyor (model=%s)", settings.model)

    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    registry = build_registry(settings)

    # Panic hotkey for computer-control automation (Ctrl+Alt+Shift+Q),
    # independent of pyautogui's own move-to-corner FAILSAFE. Registered
    # once at startup regardless of mode, since helper mode can be
    # unlocked at any point during the session. A failure here (e.g. the
    # global keyboard hook can't be installed) must never crash startup --
    # pyautogui's FAILSAFE remains as the fallback panic mechanism.
    try:
        register_panic_hotkey()
    except Exception:
        logger.exception("Panik tuşu (Ctrl+Alt+Shift+Q) kaydedilemedi")

    # Assistant/helper authority mode (see core/access_mode.py): NEO starts
    # in the restricted assistant mode every launch. The password itself
    # never touches this module -- PasswordStore only ever returns whether
    # a candidate matched.
    mode_manager = AccessModeManager()
    password_store = PasswordStore(settings.data_dir / "access.json")
    permissions = PermissionManager(mode_manager=mode_manager)

    conversation_store = ConversationStore(settings.conversation_dir)
    logger.info("Konuşma geçmişi klasörü: %s", conversation_store.directory)

    preference_store = PreferenceStore(settings.data_dir / "preferences.db")
    registry.register(RememberPreferenceTool(preference_store))
    registry.register(RecallPreferencesTool(preference_store))
    registry.register(ForgetPreferenceTool(preference_store))

    # Güvenlik merkezi backend (see NEO_V2_PLAN.md item 9): every tool call
    # the agent makes gets logged here, allowed or denied. No UI panel yet
    # (security_panel.py's home, main_window.py, is under active redesign
    # in a parallel work stream) -- get_recent_activity lets NEO answer
    # "son 24 saatte ne yaptın" directly in chat in the meantime.
    audit_store = AuditStore(settings.data_dir / "audit.db")
    registry.register(GetRecentActivityTool(audit_store))

    async def _audit_hook(tool_name, risk, tool_input, success, error) -> None:
        await asyncio.to_thread(
            audit_store.log, tool_name, risk.value if risk else "unknown", tool_input, success, error
        )

    registry.set_audit_hook(_audit_hook)

    local_llm_client = None
    if settings.enable_local_llm_fallback:
        local_llm_client = LocalLLMClient(settings.ollama_url, settings.ollama_model)

    agent = Agent(
        settings,
        registry,
        permissions=permissions,
        conversation_store=conversation_store,
        preference_store=preference_store,
        mode_manager=mode_manager,
        audit_store=audit_store,
        local_llm_client=local_llm_client,
    )

    # run_task needs a reference to the already-constructed agent (it drives
    # steps through agent.run_subtask, reusing the exact same tool loop and
    # permission checks as a normal message), so it is registered onto the
    # registry after the agent exists rather than while build_registry() is
    # building it -- the registry object itself is the same one agent holds
    # a reference to, so mutating it here still reaches agent's tool loop.
    task_store = TaskStore(settings.data_dir / "tasks.db")
    planner = TaskPlanner(agent, task_store)
    registry.register(RunTaskTool(planner))

    # Proaktif NEO (see core/scheduler.py, core/disk_watch.py): a fired job
    # or a critical disk-space alert is written to conversation history so
    # it's never lost even if no one sees the on-screen toast. The
    # Scheduler/DiskSpaceWatcher instances themselves are constructed after
    # `window` below (they also call window.show_proactive_notification),
    # since they don't need to exist this early -- only the store and tools
    # do.
    scheduled_job_store = ScheduledJobStore(settings.data_dir / "scheduled_jobs.db")
    registry.register(ScheduleTaskTool(scheduled_job_store))
    registry.register(ListScheduledTasksTool(scheduled_job_store))
    registry.register(CancelScheduledTaskTool(scheduled_job_store))

    if scheduled_job_store.find_by_name("Sabah brifingi") is None:
        from datetime import datetime

        from .core.scheduler import compute_next_run

        first_run = compute_next_run("daily", "08:00", datetime.now())
        scheduled_job_store.create_job(
            "Sabah brifingi",
            "Bugünkü takvim notlarını ve hava durumunu efendime kısaca özetle.",
            "daily",
            "08:00",
            first_run.isoformat(),
        )

    recorder = PushToTalkRecorder()
    stt = WhisperSTT(model_size=settings.whisper_model, device=settings.whisper_device)
    tts = FallbackTTS(EdgeTTS(), SapiTTS())
    spotter = KeywordSpotter(settings.data_dir / "wake_word_templates.npz")
    confirm_stt = (
        WhisperSTT(model_size=settings.wake_confirm_model, device=settings.whisper_device)
        if settings.wake_confirm_model
        else None
    )
    wake_listener = WakeWordListener(
        spotter,
        stt,
        wake_phrase=settings.wake_phrase,
        confirm_stt=confirm_stt,
        confirm_threshold=settings.wake_sensitivity,
    )
    window = MainWindow(
        agent,
        recorder=recorder,
        stt=stt,
        tts=tts,
        wake_listener=wake_listener,
        spotter=spotter,
        wake_phrase=settings.wake_phrase,
        update_manifest_url=settings.update_manifest_url,
    )
    # The confirmation dialog for MEDIUM/HIGH risk tools lives on the GUI,
    # which needs the agent to exist first -- wire it up after the window is
    # built rather than the other way around.
    permissions.set_confirm(window.confirm_action)
    agent.set_listening_control(window.set_listening)
    agent.set_mode_unlock_control(
        lambda: request_helper_mode_unlock(password_store, mode_manager, parent=window)
    )

    # Mode indicator / task progress / audit / proactive notifications: all
    # four surfaces live on the GUI, wired post-hoc for the same reason as
    # permissions.set_confirm above -- the objects that produce this data
    # (mode_manager, planner, audit_store) exist before `window` does.
    mode_manager.set_on_change(window.set_access_mode)

    def _on_task_progress(task) -> None:
        counted_steps = [s for s in task.steps if s.status != "skipped"]
        done = sum(1 for s in counted_steps if s.status in ("done", "failed"))
        window.set_task_progress(task.goal, done, len(counted_steps), task.status)

    planner.set_on_progress(_on_task_progress)
    window.set_audit_store(audit_store)

    def _announce(text: str) -> None:
        conversation_store.add_message("assistant", text)
        window.show_proactive_notification(text)

    scheduler = Scheduler(
        agent, scheduled_job_store, on_fire=lambda job, text: _announce(text)
    )
    scheduler.start()

    disk_watcher = DiskSpaceWatcher(on_alert=_announce)
    disk_watcher.start()

    window.show()

    # Telefon/web paneli (see NEO_V2_PLAN.md item 10): opt-in, since it
    # opens a LAN-reachable port. The token is generated once and stored
    # locally; the user copies it to their phone themselves; the panel
    # never surfaces it over the network.
    if settings.enable_web_panel:
        from .web.server import WebPanelServer, load_or_create_token

        web_token = load_or_create_token(settings.data_dir / "web_token.txt")
        web_panel = WebPanelServer(agent, web_token, port=settings.web_panel_port)
        web_panel.start()
        logger.info(
            "Web paneli %d portunda başlatıldı (token data/web_token.txt dosyasında)",
            settings.web_panel_port,
        )

    # Checked in the background so a slow or unreachable update server can
    # never delay startup; a failure here is silent by design.
    loop.create_task(window.check_for_update())

    # Synthesized once and cached, so waking NEO never waits on a network
    # round trip to hear that it was heard.
    chime.configure(settings.data_dir)
    loop.create_task(chime.prepare_cues())

    with loop:
        return loop.run_forever()


if __name__ == "__main__":
    sys.exit(main())
