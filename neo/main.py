from __future__ import annotations

import asyncio
import logging
import sys

from PySide6.QtWidgets import QApplication
from qasync import QEventLoop

from .config.settings import Settings, load_settings
from .core.agent import Agent
from .core.permissions import PermissionManager
from .logging_setup import setup_logging
from .memory.calendar_store import CalendarStore
from .memory.conversation_store import ConversationStore
from .memory.preference_store import PreferenceStore
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
from .tools.power import LockComputerTool, RestartComputerTool, ShutdownComputerTool
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
    permissions = PermissionManager()
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
