import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neo.config.settings import Settings
from neo.core.agent import (
    MAX_TOKENS_DEFAULT,
    MAX_TOKENS_RESEARCH,
    SPOKEN_SUMMARY_PREFIX,
    Agent,
    extract_spoken_summary,
)
from neo.tools.base import ToolRegistry
from neo.tools.time_tools import GetTimeTool


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
        self.last_system = ""
        self.last_max_tokens = 0

    def send(self, messages: list[dict], system: str, tools: list[dict], max_tokens: int = 1024) -> Any:
        self.last_system = system
        self.last_max_tokens = max_tokens
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


def test_research_mode_turns_on_without_calling_the_llm():
    agent = Agent(_settings(), ToolRegistry())

    reply = asyncio.run(agent.handle_message("neo araştırma modu"))

    assert agent.research_mode is True
    assert "araştırma modu" in reply.lower()


def test_research_mode_turns_off():
    agent = Agent(_settings(), ToolRegistry())
    asyncio.run(agent.handle_message("araştırma modu"))

    reply = asyncio.run(agent.handle_message("araştırma modunu kapat"))

    assert agent.research_mode is False
    assert "kapat" in reply.lower()


def test_research_mode_uses_bigger_token_budget_and_extra_prompt():
    registry = ToolRegistry()
    fake_llm = FakeLLM([FakeMessage([FakeBlock(type="text", text="rapor")])])
    agent = Agent(_settings(), registry, llm_client=fake_llm)
    asyncio.run(agent.handle_message("araştırma modu"))

    asyncio.run(agent.handle_message("yapay zekanın eğitime etkisi"))

    assert fake_llm.last_max_tokens == MAX_TOKENS_RESEARCH
    assert "ARAŞTIRMA MODU AKTİF" in fake_llm.last_system


def test_normal_mode_uses_default_token_budget():
    registry = ToolRegistry()
    fake_llm = FakeLLM([FakeMessage([FakeBlock(type="text", text="kısa cevap")])])
    agent = Agent(_settings(), registry, llm_client=fake_llm)

    asyncio.run(agent.handle_message("bana bir fikir ver"))

    assert fake_llm.last_max_tokens == MAX_TOKENS_DEFAULT
    assert "ARAŞTIRMA MODU AKTİF" not in fake_llm.last_system


def test_research_mode_bypasses_the_local_shortcut_layer():
    # In research mode even "saat kaç" should be researched by the LLM
    # rather than short-circuited by the local fast path.
    registry = ToolRegistry()
    registry.register(GetTimeTool())
    fake_llm = FakeLLM([FakeMessage([FakeBlock(type="text", text="detaylı rapor")])])
    agent = Agent(_settings(), registry, llm_client=fake_llm)
    asyncio.run(agent.handle_message("araştırma modu"))

    reply = asyncio.run(agent.handle_message("saat kaç"))

    assert reply == "detaylı rapor"


def test_extract_spoken_summary_pulls_marked_line():
    reply = (
        f"{SPOKEN_SUMMARY_PREFIX} En önemli nokta şudur.\n\n"
        "KONU\nUzun rapor...\n\nDETAYLI ANALİZ\nÇok uzun metin..."
    )
    assert extract_spoken_summary(reply) == "En önemli nokta şudur."


def test_extract_spoken_summary_falls_back_to_full_text():
    reply = "Saat 14.30."
    assert extract_spoken_summary(reply) == "Saat 14.30."
