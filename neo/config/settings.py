from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Safe direction: neo.voice never imports neo.config, so no cycle.
from ..voice.wake_word import WAKE_PHRASE_THRESHOLD

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def user_data_root() -> Path:
    """Where NEO keeps things the user would hate to lose: the calendar, the
    enrolled wake word, the Google token, the API key.

    Packaged, this must live outside the installation. PyInstaller resolves
    the project root to the bundle's `_internal` folder, so data written
    there sits inside the very directory the updater replaces -- every
    update would silently wipe the user's calendar and voice enrollment.
    It also breaks the moment NEO is installed somewhere read-only.

    Running from source keeps using the project folder, so development and
    the existing checkout are unaffected.
    """
    if not is_frozen():
        return PROJECT_ROOT
    base = os.getenv("LOCALAPPDATA") or os.path.expanduser("~")
    return Path(base) / "NEO"


def _legacy_sources() -> list[Path]:
    """Where an earlier, source-run install might be.

    PROJECT_ROOT alone was wrong for the packaged build: it comes from
    __file__, which under PyInstaller resolves inside the bundle's own
    `_internal` folder. Migration therefore "found" only what a previous
    packaged run had written there and copied an empty calendar over,
    leaving the API key, Google credentials and wake-word enrollment behind.
    """
    candidates = [PROJECT_ROOT, Path.home() / "NEO"]
    seen: list[Path] = []
    for path in candidates:
        if path.is_dir() and path not in seen and (path / "run_neo.py").is_file():
            seen.append(path)
    return seen


def _migrate_legacy_data(target: Path) -> None:
    """Brings across setup the user already did, on a packaged run.

    Without this the packaged build starts blank -- no API key, no calendar,
    no enrolled wake word. Copies rather than moves, so the source install
    keeps working, and never overwrites a file the packaged build already
    has: it runs on every start and must only ever fill in what's missing.
    """
    try:
        for source_root in _legacy_sources():
            for name in (".env", "credentials.json"):
                source = source_root / name
                destination = target / name
                if source.is_file() and not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    logger.info("Önceki kurulumdan taşındı: %s", name)

            legacy_data = source_root / "data"
            if not legacy_data.is_dir():
                continue
            for item in legacy_data.iterdir():
                destination = target / "data" / item.name
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                if item.is_dir():
                    shutil.copytree(item, destination)
                else:
                    shutil.copy2(item, destination)
                logger.info("Önceki kurulumdan taşındı: data/%s", item.name)
    except Exception:
        logger.exception("Eski veriler taşınamadı; NEO boş yapılandırmayla başlayacak")


USER_ROOT = user_data_root()
if is_frozen():
    USER_ROOT.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_data(USER_ROOT)

# The packaged build reads its key from the user folder; a source checkout
# keeps reading the one next to the code.
load_dotenv(USER_ROOT / ".env")
if not is_frozen():
    load_dotenv(PROJECT_ROOT / ".env")


class ConfigError(Exception):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    model: str
    default_city: str
    log_level: str
    log_dir: Path
    data_dir: Path
    whisper_model: str
    whisper_device: str
    # Defaulted so adding further optional settings doesn't break every
    # caller/test that constructs Settings directly.
    conversation_dir: Path = PROJECT_ROOT / "data" / "conversations"
    update_manifest_url: str | None = None
    # Only ever shown in the enrollment prompts -- detection is acoustic and
    # learns whatever the user actually records, so this text and the audio
    # can't drift out of sync. Two words by default because a single short
    # syllable gives DTW very little to discriminate on, which is why every
    # commercial wake word is multi-syllable.
    wake_phrase: str = "Neo uyan"
    # A second, smaller model used only to confirm that an acoustic wake
    # match really said the phrase. Kept separate from `whisper_model`
    # because it stays resident while NEO idles listening, where the larger
    # command model's memory footprint isn't worth paying continuously --
    # and two words are an easy target for a small model. Empty disables
    # confirmation and leaves the acoustic verdict final.
    wake_confirm_model: str = "base"
    # How closely a confirmation transcript must match wake_phrase to
    # accept it, 0..1. Lower = fewer "I said it, why didn't it wake up"
    # repeats but more accidental wakes on unrelated speech; higher is the
    # reverse. Every wake-word engine exposes some form of this as a
    # user-tunable "sensitivity" rather than one fixed value, since the
    # right tradeoff genuinely depends on the room and microphone.
    #
    # Taken from WAKE_PHRASE_THRESHOLD rather than written out again: this
    # is the value actually passed to the listener, so when the two were
    # separate numbers, tuning the constant in wake_word.py changed nothing
    # at runtime and the user kept having to repeat the wake phrase.
    wake_sensitivity: float = WAKE_PHRASE_THRESHOLD
    # Opt-in: this opens a network port (LAN-reachable, token-protected --
    # see neo/web/server.py), which is not something a fresh install should
    # do without the user asking for it.
    enable_web_panel: bool = False
    web_panel_port: int = 8765
    # Faz 2: when Claude is unreachable (no credit, no internet, API outage)
    # on the very first LLM call of a turn -- i.e. before any tool has been
    # decided on -- Agent falls back to a local Ollama model for a plain-text
    # reply instead of surfacing the error. Off by default: it only helps if
    # Ollama is actually installed and running, and silently trying to reach
    # a port nobody's listening on just adds a timeout to every failure.
    enable_local_llm_fallback: bool = False
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    # When on (and enable_local_llm_fallback is also on), ordinary chat is
    # routed to the local model FIRST -- it has no tools, and is instructed
    # to say so (see local_llm_client.ROUTER_INSTRUCTION) rather than answer
    # anything that actually needs one, at which point Agent escalates to
    # Claude exactly as if the local model weren't there. This is the "sohbet
    # her zaman yerelde, Claude sadece iş/araç gerektiğinde" mode; the plain
    # enable_local_llm_fallback-only mode instead only reaches the local
    # model when Claude itself is unreachable.
    route_chat_to_local: bool = False

    def require_api_key(self) -> str:
        if not self.anthropic_api_key:
            raise ConfigError(
                "ANTHROPIC_API_KEY tanımlı değil. .env dosyasına Anthropic API "
                "anahtarınızı ekleyin (bkz. .env.example)."
            )
        return self.anthropic_api_key


def _default_conversation_dir() -> Path:
    """Conversation history goes into a cloud-synced folder when one exists,
    so the same history is readable from the user's other devices (phone
    included) without any extra account or server. Falls back to the local
    data dir when OneDrive isn't set up."""
    onedrive = os.getenv("OneDrive") or os.getenv("OneDriveConsumer")
    if onedrive and Path(onedrive).is_dir():
        return Path(onedrive) / "NEO"
    return USER_ROOT / "data" / "conversations"


def load_settings() -> Settings:
    configured_dir = os.getenv("NEO_CONVERSATION_DIR")
    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        model=os.getenv("NEO_MODEL", "claude-sonnet-5"),
        default_city=os.getenv("NEO_DEFAULT_CITY", "Bursa"),
        log_level=os.getenv("NEO_LOG_LEVEL", "INFO"),
        log_dir=USER_ROOT / "logs",
        data_dir=USER_ROOT / "data",
        whisper_model=os.getenv("NEO_WHISPER_MODEL", "small"),
        whisper_device=os.getenv("NEO_WHISPER_DEVICE", "cpu"),
        conversation_dir=Path(configured_dir) if configured_dir else _default_conversation_dir(),
        update_manifest_url=os.getenv("NEO_UPDATE_MANIFEST_URL") or None,
        wake_phrase=os.getenv("NEO_WAKE_PHRASE", "Neo uyan"),
        wake_confirm_model=os.getenv("NEO_WAKE_CONFIRM_MODEL", "base"),
        wake_sensitivity=float(
            os.getenv("NEO_WAKE_SENSITIVITY", str(WAKE_PHRASE_THRESHOLD))
        ),
        enable_web_panel=os.getenv("NEO_ENABLE_WEB_PANEL", "").strip().lower() in ("1", "true", "yes"),
        web_panel_port=int(os.getenv("NEO_WEB_PANEL_PORT", "8765")),
        enable_local_llm_fallback=os.getenv("NEO_ENABLE_LOCAL_LLM_FALLBACK", "").strip().lower()
        in ("1", "true", "yes"),
        ollama_url=os.getenv("NEO_OLLAMA_URL", "http://localhost:11434"),
        ollama_model=os.getenv("NEO_OLLAMA_MODEL", "qwen2.5:3b"),
        route_chat_to_local=os.getenv("NEO_ROUTE_CHAT_TO_LOCAL", "").strip().lower()
        in ("1", "true", "yes"),
    )
