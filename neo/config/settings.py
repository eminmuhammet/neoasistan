from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

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


def _migrate_legacy_data(target: Path) -> None:
    """Copies data from an earlier source-run install on first packaged run.

    Without this the packaged build starts blank -- no API key, no calendar,
    no enrolled wake word -- and the user has to redo setup they already did.
    Copies rather than moves, so the source install keeps working too.
    """
    if target.exists() or not PROJECT_ROOT.exists():
        return
    try:
        target.mkdir(parents=True, exist_ok=True)
        legacy_data = PROJECT_ROOT / "data"
        if legacy_data.is_dir():
            shutil.copytree(legacy_data, target / "data", dirs_exist_ok=True)
        for name in (".env", "credentials.json"):
            source = PROJECT_ROOT / name
            if source.is_file():
                shutil.copy2(source, target / name)
        logger.info("Önceki kurulumdan veriler taşındı: %s", target)
    except Exception:
        logger.exception("Eski veriler taşınamadı; NEO boş yapılandırmayla başlayacak")


USER_ROOT = user_data_root()
if is_frozen():
    _migrate_legacy_data(USER_ROOT)
    USER_ROOT.mkdir(parents=True, exist_ok=True)

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
    )
