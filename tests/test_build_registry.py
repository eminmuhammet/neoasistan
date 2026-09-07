"""build_registry() wires together the tools that don't need anything
built later in main() (agent, stores that depend on it, etc.) -- a
duplicate tool name among them (easy to introduce across many edits in one
long session) would raise inside ToolRegistry.register itself, so just
successfully building it once is a meaningful regression check. Tools
registered later in main() itself (preferences, scheduling, audit, ...)
aren't reachable from build_registry() alone since main() isn't structured
to be called in a test."""

from pathlib import Path

from neo.config.settings import Settings
from neo.main import build_registry


def _settings(tmp_path) -> Settings:
    return Settings(
        anthropic_api_key=None,
        model="test-model",
        default_city="Bursa",
        log_level="INFO",
        log_dir=tmp_path / "logs",
        data_dir=tmp_path / "data",
        whisper_model="tiny",
        whisper_device="cpu",
        conversation_dir=tmp_path / "conversations",
    )


def test_build_registry_succeeds_with_no_duplicate_tool_names(tmp_path):
    registry = build_registry(_settings(tmp_path))

    assert registry.get("get_time") is not None
    assert registry.get("click") is not None
    assert registry.get("read_document") is not None
    assert registry.get("get_volume") is not None
    assert registry.get("send_email") is not None
    assert len(registry.anthropic_tools()) > 25


def test_every_registered_tool_has_a_description_and_schema(tmp_path):
    registry = build_registry(_settings(tmp_path))

    for schema in registry.anthropic_tools():
        assert schema["description"]
        assert schema["input_schema"]["type"] == "object"
