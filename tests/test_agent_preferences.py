"""Known preferences are injected into every system prompt, the same way
_current_date_context() is -- so Claude doesn't have to remember to call
recall_preferences before it can use what it already knows."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neo.config.settings import Settings
from neo.core.agent import Agent, _preference_context
from neo.memory.preference_store import PreferenceStore
from neo.tools.base import ToolRegistry


@dataclass
class FakeBlock:
    type: str
    text: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)
    id: str = "tool_1"


@dataclass
class FakeMessage:
    content: list


class FakeLLM:
    def __init__(self, responses: list[FakeMessage]) -> None:
        self._responses = list(responses)

    def send(self, messages: list[dict], system: str, tools: list[dict], max_tokens: int = 1024) -> Any:
        self.last_system = system
        return self._responses.pop(0)


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


def test_preference_context_is_empty_with_no_store():
    assert _preference_context(None) == ""


def test_preference_context_is_empty_for_a_fresh_store(tmp_path):
    store = PreferenceStore(tmp_path / "p.db")
    assert _preference_context(store) == ""


def test_preference_context_lists_known_facts(tmp_path):
    store = PreferenceStore(tmp_path / "p.db")
    store.remember("şehir", "Bursa")
    store.remember("meslek", "mühendis")

    context = _preference_context(store)

    assert "şehir: Bursa" in context
    assert "meslek: mühendis" in context


def test_handle_message_injects_preferences_into_the_system_prompt(tmp_path):
    """The behavioral guarantee: a real handle_message() call must actually
    carry known facts to the LLM, not just have the helper function work in
    isolation."""
    store = PreferenceStore(tmp_path / "p.db")
    store.remember("şehir", "Bursa")

    registry = ToolRegistry()
    fake_llm = FakeLLM([FakeMessage([FakeBlock(type="text", text="Merhaba efendim.")])])

    agent = Agent(_settings(), registry, llm_client=fake_llm, preference_store=store)
    asyncio.run(agent.handle_message("selam"))

    assert "şehir: Bursa" in fake_llm.last_system


def test_handle_message_without_preferences_has_no_stray_section(tmp_path):
    """A fresh install (no facts learned yet) shouldn't carry a dangling
    'Kullanıcı hakkında bildiklerin:' header with nothing under it."""
    store = PreferenceStore(tmp_path / "p.db")

    registry = ToolRegistry()
    fake_llm = FakeLLM([FakeMessage([FakeBlock(type="text", text="Merhaba.")])])

    agent = Agent(_settings(), registry, llm_client=fake_llm, preference_store=store)
    asyncio.run(agent.handle_message("selam"))

    assert "Kullanıcı hakkında bildiklerin" not in fake_llm.last_system


def test_agent_without_a_preference_store_still_works(tmp_path):
    """Every existing caller that constructs Agent(...) without this new
    parameter (tests, and any future one) must keep working unchanged."""
    registry = ToolRegistry()
    fake_llm = FakeLLM([FakeMessage([FakeBlock(type="text", text="Merhaba.")])])

    agent = Agent(_settings(), registry, llm_client=fake_llm)
    reply = asyncio.run(agent.handle_message("selam"))

    assert reply == "Merhaba."


def test_the_three_preference_tools_are_registered_when_agent_uses_them(tmp_path):
    """Round trip: the agent actually calls remember_preference as a tool
    and the fact lands in the store Claude was told about."""
    from neo.tools.preferences import RememberPreferenceTool

    store = PreferenceStore(tmp_path / "p.db")
    registry = ToolRegistry()
    registry.register(RememberPreferenceTool(store))

    tool_call = FakeMessage(
        [
            FakeBlock(
                type="tool_use",
                name="remember_preference",
                input={"key": "şehir", "value": "Bursa"},
                id="t1",
            )
        ]
    )
    final = FakeMessage([FakeBlock(type="text", text="Not ettim efendim.")])
    fake_llm = FakeLLM([tool_call, final])

    agent = Agent(_settings(), registry, llm_client=fake_llm, preference_store=store)
    reply = asyncio.run(agent.handle_message("Bursa'da yaşıyorum, bunu hatırla"))

    assert reply == "Not ettim efendim."
    assert store.get("şehir").value == "Bursa"
