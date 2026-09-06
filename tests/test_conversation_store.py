import asyncio
from pathlib import Path

from neo.config.settings import Settings
from neo.core.agent import Agent
from neo.memory.conversation_store import ConversationStore
from neo.tools.base import ToolRegistry
from neo.tools.time_tools import GetTimeTool


def _settings() -> Settings:
    return Settings(
        anthropic_api_key=None,
        model="test-model",
        default_city="Bursa",
        log_level="INFO",
        log_dir=Path("."),
        data_dir=Path("."),
        whisper_model="tiny",
        whisper_device="cpu",
        conversation_dir=Path("."),
    )


def test_messages_persist_and_read_back_in_order(tmp_path):
    store = ConversationStore(tmp_path / "sync")
    store.add_message("user", "merhaba")
    store.add_message("assistant", "selam")

    messages = store.recent_messages()

    assert [(m.role, m.text) for m in messages] == [("user", "merhaba"), ("assistant", "selam")]


def test_recent_messages_respects_limit_and_keeps_latest(tmp_path):
    store = ConversationStore(tmp_path / "sync")
    for i in range(10):
        store.add_message("user", f"mesaj {i}")

    messages = store.recent_messages(limit=3)

    assert [m.text for m in messages] == ["mesaj 7", "mesaj 8", "mesaj 9"]


def test_writes_a_readable_daily_transcript_for_other_devices(tmp_path):
    directory = tmp_path / "sync"
    store = ConversationStore(directory)
    store.add_message("user", "bugün hava nasıl")
    store.add_message("assistant", "Bursa: açık, 21°C.")

    transcripts = list((directory / "konusmalar").glob("*.md"))
    assert len(transcripts) == 1
    content = transcripts[0].read_text(encoding="utf-8")
    assert "bugün hava nasıl" in content
    assert "Bursa: açık, 21°C." in content
    assert "Sen" in content and "NEO" in content


def test_store_survives_reopening(tmp_path):
    directory = tmp_path / "sync"
    ConversationStore(directory).add_message("user", "kalıcı mı")

    reopened = ConversationStore(directory)

    assert [m.text for m in reopened.recent_messages()] == ["kalıcı mı"]
    assert reopened.message_count() == 1


def test_agent_records_exchanges_and_restores_them_next_time(tmp_path):
    directory = tmp_path / "sync"
    store = ConversationStore(directory)
    registry = ToolRegistry()
    registry.register(GetTimeTool())

    agent = Agent(_settings(), registry, conversation_store=store)
    asyncio.run(agent.handle_message("saat kaç"))

    saved = store.recent_messages()
    assert saved[0].role == "user" and saved[0].text == "saat kaç"
    assert saved[1].role == "assistant" and saved[1].text.startswith("Saat ")

    # A fresh Agent (as if the app restarted) should pick the history back up.
    restarted = Agent(_settings(), registry, conversation_store=ConversationStore(directory))
    assert len(restarted._context.messages) == 2
