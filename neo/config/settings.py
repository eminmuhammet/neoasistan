from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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
    return PROJECT_ROOT / "data" / "conversations"


def load_settings() -> Settings:
    configured_dir = os.getenv("NEO_CONVERSATION_DIR")
    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        model=os.getenv("NEO_MODEL", "claude-sonnet-5"),
        default_city=os.getenv("NEO_DEFAULT_CITY", "Bursa"),
        log_level=os.getenv("NEO_LOG_LEVEL", "INFO"),
        log_dir=PROJECT_ROOT / "logs",
        data_dir=PROJECT_ROOT / "data",
        whisper_model=os.getenv("NEO_WHISPER_MODEL", "small"),
        whisper_device=os.getenv("NEO_WHISPER_DEVICE", "cpu"),
        conversation_dir=Path(configured_dir) if configured_dir else _default_conversation_dir(),
        update_manifest_url=os.getenv("NEO_UPDATE_MANIFEST_URL") or None,
        wake_phrase=os.getenv("NEO_WAKE_PHRASE", "Neo uyan"),
        wake_confirm_model=os.getenv("NEO_WAKE_CONFIRM_MODEL", "base"),
    )
