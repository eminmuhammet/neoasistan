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
from .core.disk_watch import DiskSpaceWatcher
from .core.permissions import PermissionManager
from .core.planner import TaskPlanner
from .core.scheduler import Scheduler
from .logging_setup import setup_logging
from .memory.calendar_store import CalendarStore
from .memory.conversation_store import ConversationStore
from .memory.preference_store import PreferenceStore
from .memory.scheduled_job_store import ScheduledJobStore
from .memory.task_store import TaskStore
from .tools.applications import OpenApplicationTool, OpenWebsiteTool
from .tools.base import ToolRegistry
from .tools.calendar import (
    AddCalendarNoteTool,
    DeleteCalendarNoteTool,
    GetCalendarNotesTool,
    UpdateCalendarNoteTool,
)
from .tools.filesystem import FindFileTool, OpenFolderTool
from .tools.google_calendar import GoogleCalendarSync
from .tools.preferences import (
    ForgetPreferenceTool,
    RecallPreferencesTool,
    RememberPreferenceTool,
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
from .voice import chime
from .tools.weather import GetWeatherTool
from .ui.main_window import MainWindow
from .ui.mode_unlock_dialog import request_helper_mode_unlock
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
    registry.register(GetSystemInfoTool())
    registry.register(GetCpuUsageTool())
    registry.register(GetRamUsageTool())
    registry.register(GetDiskUsageTool())
    registry.register(GetGpuStatusTool())
    registry.register(GetNetworkStatusTool())
    registry.register(GetWeatherTool(settings.default_city))

    calendar_store = CalendarStore(settings.data_dir / "calendar.db")
    google_sync = GoogleCalendarSync(
        credentials_path=settings.data_dir.parent / "credentials.json",
        token_path=settings.data_dir / "google_token.json",
    )
    registry.register(AddCalendarNoteTool(calendar_store, google_sync=google_sync))
    registry.register(GetCalendarNotesTool(calendar_store))
    registry.register(DeleteCalendarNoteTool(calendar_store, google_sync=google_sync))
    registry.register(UpdateCalendarNoteTool(calendar_store, google_sync=google_sync))

    registry.register(FindFileTool())
    registry.register(OpenFolderTool())

    registry.register(LockComputerTool())
    registry.register(ShutdownComputerTool())
    registry.register(RestartComputerTool())
    registry.register(CaptureScreenTool())
    return registry


def main() -> int:
    settings = load_settings()
    setup_logging(settings.log_dir, settings.log_level)
    logger.info("NEO başlatılıyor (model=%s)", settings.model)

    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    registry = build_registry(settings)

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

    agent = Agent(
        settings,
        registry,
        permissions=permissions,
        conversation_store=conversation_store,
        preference_store=preference_store,
        mode_manager=mode_manager,
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
    # it isn't lost, but there is deliberately no live on-screen popup or
    # spoken announcement here yet -- that surface lives in main_window.py,
    # which is under active redesign in a parallel work stream. Wiring
    # through the stable, public conversation_store keeps this feature
    # fully working without touching that file.
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

    scheduler = Scheduler(
        agent,
        scheduled_job_store,
        on_fire=lambda job, text: conversation_store.add_message("assistant", text),
    )
    scheduler.start()

    disk_watcher = DiskSpaceWatcher(
        on_alert=lambda text: conversation_store.add_message("assistant", text),
    )
    disk_watcher.start()

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
    window.show()

    # Checked in the background so a slow or unreachable update server can
    # never delay startup; a failure here is silent by design.
    loop.create_task(window.check_for_update())

    # Wake-up cues are synthesized once and cached, so hearing "Dinliyorum
    # efendim" doesn't cost a network round trip every time NEO is woken.
    chime.configure(settings.data_dir)
    loop.create_task(chime.prepare_cues())

    with loop:
        return loop.run_forever()


if __name__ == "__main__":
    sys.exit(main())
