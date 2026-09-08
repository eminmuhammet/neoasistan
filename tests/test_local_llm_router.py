"""When Settings.route_chat_to_local is on, ordinary chat is meant to go to
the local model FIRST -- it has no tools and is told to say ESCALATE_MARKER
instead of guessing at anything that actually needs one (see
local_llm_client.ROUTER_INSTRUCTION). Only an escalation should ever reach
Claude; this is what keeps everyday conversation off the API entirely
rather than merely cheaper (contrast with test_local_llm_fallback.py, which
only reaches the local model when Claude itself is unreachable).
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from neo.config.settings import Settings
from neo.core.agent import Agent
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
        self.calls = 0

    def send(self, messages, system, tools, max_tokens=1024):
        self.calls += 1
        return self._responses.pop(0)


class RoutingLocalLLM:
    """Stands in for LocalLLMClient.chat_or_escalate without touching Ollama."""

    def __init__(self, needs_claude: bool, text: str = "") -> None:
        self._needs_claude = needs_claude
        self._text = text
        self.calls = 0

    async def chat_or_escalate(self, messages, system):
        self.calls += 1
        return self._needs_claude, self._text


def _settings(route_chat_to_local: bool, api_key: str | None = "sk-test") -> Settings:
    return Settings(
        anthropic_api_key=api_key,
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


def test_plain_chat_is_answered_locally_and_claude_is_never_called():
    registry = ToolRegistry()
    local = RoutingLocalLLM(needs_claude=False, text="Merhaba! Nasıl yardımcı olabilirim?")
    llm = FakeLLM([])  # would raise IndexError if popped -- proves Claude wasn't touched
    agent = Agent(_settings(True), registry, llm_client=llm, local_llm_client=local)

    reply = asyncio.run(agent.handle_message("kuantum bilgisayarlar hakkında ne düşünüyorsun"))

    assert reply == "Merhaba! Nasıl yardımcı olabilirim?"
    assert local.calls == 1
    assert llm.calls == 0


def test_plain_chat_needs_no_claude_api_key_when_routed_locally():
    """Local-only replies shouldn't require an Anthropic key at all -- that's
    the whole point of routing chat away from Claude by default."""
    registry = ToolRegistry()
    local = RoutingLocalLLM(needs_claude=False, text="Selam!")
    agent = Agent(_settings(True, api_key=None), registry, local_llm_client=local)

    reply = asyncio.run(agent.handle_message("kuantum bilgisayarlar hakkında ne düşünüyorsun"))

    assert reply == "Selam!"
    assert local.calls == 1


def test_escalation_falls_through_to_the_real_claude_tool_loop():
    registry = ToolRegistry()
    local = RoutingLocalLLM(needs_claude=True)
    final = FakeMessage([FakeBlock(type="text", text="İşte cevabın.")])
    llm = FakeLLM([final])
    agent = Agent(_settings(True), registry, llm_client=llm, local_llm_client=local)

    reply = asyncio.run(agent.handle_message("şu dosyayı bulup açar mısın"))

    assert reply == "İşte cevabın."
    assert local.calls == 1
    assert llm.calls == 1


def test_research_mode_skips_local_routing_even_when_enabled():
    registry = ToolRegistry()
    local = RoutingLocalLLM(needs_claude=False, text="asla dönmemeli")
    final = FakeMessage([FakeBlock(type="text", text="Araştırma raporu.")])
    llm = FakeLLM([final])
    agent = Agent(_settings(True), registry, llm_client=llm, local_llm_client=local)
    agent.research_mode = True

    reply = asyncio.run(agent.handle_message("bunu araştır"))

    assert reply == "Araştırma raporu."
    assert local.calls == 0


def test_routing_disabled_by_default_goes_straight_to_claude():
    registry = ToolRegistry()
    local = RoutingLocalLLM(needs_claude=False, text="asla dönmemeli")
    final = FakeMessage([FakeBlock(type="text", text="Claude cevabı.")])
    llm = FakeLLM([final])
    agent = Agent(_settings(False), registry, llm_client=llm, local_llm_client=local)

    reply = asyncio.run(agent.handle_message("kuantum bilgisayarlar hakkında ne düşünüyorsun"))

    assert reply == "Claude cevabı."
    assert local.calls == 0
