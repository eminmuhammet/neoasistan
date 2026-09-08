"""Faz 3A: a one-off research request ("bunu araştırır mısın", "detaylı
incele") should get the full research-turn treatment (bigger token budget,
RESEARCH_MODE_PROMPT, skips local fast path/router) for just that message,
without the user first saying "araştırma modu" -- and without leaving
research_mode toggled on for the next, unrelated message.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from neo.config.settings import Settings
from neo.core.agent import MAX_TOKENS_DEFAULT, MAX_TOKENS_RESEARCH, Agent
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


class RecordingLLM:
    def __init__(self, responses: list[FakeMessage]) -> None:
        self._responses = list(responses)
        self.last_system = None
        self.last_max_tokens = None

    def send(self, messages, system, tools, max_tokens=1024):
        self.last_system = system
        self.last_max_tokens = max_tokens
        return self._responses.pop(0)


class RoutingLocalLLM:
    def __init__(self, needs_claude: bool = False, text: str = "yerel cevap") -> None:
        self._needs_claude = needs_claude
        self._text = text
        self.calls = 0

    async def chat_or_escalate(self, messages, system):
        self.calls += 1
        return self._needs_claude, self._text


def _settings(route_chat_to_local: bool = False) -> Settings:
    return Settings(
        anthropic_api_key="sk-test",
        model="test-model",
        default_city="Bursa",
        log_level="INFO",
        log_dir=Path("."),
        data_dir=Path("."),
        whisper_model="tiny",
        whisper_device="cpu",
        conversation_dir=Path("."),
        route_chat_to_local=route_chat_to_local,
    )


def test_a_one_off_research_phrase_gets_the_bigger_token_budget_and_research_prompt():
    registry = ToolRegistry()
    llm = RecordingLLM([FakeMessage([FakeBlock(type="text", text="Rapor burada.")])])
    agent = Agent(_settings(), registry, llm_client=llm)

    reply = asyncio.run(agent.handle_message("yapay zeka regülasyonlarını araştırır mısın"))

    assert reply == "Rapor burada."
    assert llm.last_max_tokens == MAX_TOKENS_RESEARCH
    assert "ARAŞTIRMA MODU AKTİF" in llm.last_system
    assert agent.research_mode is False  # never toggled persistently


def test_research_mode_does_not_leak_into_the_next_ordinary_message():
    registry = ToolRegistry()
    llm = RecordingLLM(
        [
            FakeMessage([FakeBlock(type="text", text="Rapor burada.")]),
            FakeMessage([FakeBlock(type="text", text="Sıradan cevap.")]),
        ]
    )
    agent = Agent(_settings(), registry, llm_client=llm)

    asyncio.run(agent.handle_message("bunu detaylı incele lütfen"))
    asyncio.run(agent.handle_message("teşekkürler"))

    assert llm.last_max_tokens == MAX_TOKENS_DEFAULT
    assert "ARAŞTIRMA MODU AKTİF" not in llm.last_system


def test_a_research_phrase_skips_local_routing_even_when_it_is_on():
    registry = ToolRegistry()
    local = RoutingLocalLLM(needs_claude=False, text="asla dönmemeli")
    llm = RecordingLLM([FakeMessage([FakeBlock(type="text", text="Rapor.")])])
    agent = Agent(_settings(route_chat_to_local=True), registry, llm_client=llm, local_llm_client=local)

    reply = asyncio.run(agent.handle_message("kuantum bilgisayarlar hakkında araştırma yap"))

    assert reply == "Rapor."
    assert local.calls == 0


def test_an_ordinary_chat_phrase_still_uses_the_default_token_budget():
    registry = ToolRegistry()
    llm = RecordingLLM([FakeMessage([FakeBlock(type="text", text="Selam.")])])
    agent = Agent(_settings(), registry, llm_client=llm)

    reply = asyncio.run(agent.handle_message("bugün canım biraz sıkkın"))

    assert reply == "Selam."
    assert llm.last_max_tokens == MAX_TOKENS_DEFAULT
