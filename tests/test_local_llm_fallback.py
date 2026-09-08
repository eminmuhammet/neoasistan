"""Faz 2: when Claude can't be reached (no credit, no internet, API outage),
Agent used to just hand the user LLMRequestError's message back verbatim.
If a local Ollama model is configured (Agent(local_llm_client=...), see
core/local_llm_client.py), the very first LLM call of a turn falls back to
it instead -- but only the first: once Claude was already mid-tool-use, a
local model with no tools has no way to pick that thread back up, so later
iterations must still surface the real error.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from neo.config.settings import Settings
from neo.core.agent import Agent
from neo.core.llm_client import LLMRequestError
from neo.core.local_llm_client import LocalLLMError, to_plain_messages
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


class FailingLLM:
    def send(self, messages, system, tools, max_tokens=1024):
        raise LLMRequestError("Claude API'sine ulaşılamadı.")


class FailingThenToolLLM:
    """Fails only from the second call onward, so a test can put a real
    tool_use response first without ever reaching a real LLM."""

    def __init__(self, first_response) -> None:
        self._first_response = first_response
        self.calls = 0

    def send(self, messages, system, tools, max_tokens=1024):
        self.calls += 1
        if self.calls == 1:
            return self._first_response
        raise LLMRequestError("Claude API'sine ulaşılamadı.")


class StubLocalLLM:
    def __init__(self, reply: str = "Yerel modelden cevap.") -> None:
        self.reply = reply
        self.calls = 0

    async def chat(self, messages, system):
        self.calls += 1
        return self.reply


class BrokenLocalLLM:
    async def chat(self, messages, system):
        raise LocalLLMError("Ollama çalışmıyor.")


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


def test_falls_back_to_local_llm_when_claude_unreachable_on_first_call():
    registry = ToolRegistry()
    local_llm = StubLocalLLM("Merhaba, ben yerel modelim.")
    agent = Agent(_settings(), registry, llm_client=FailingLLM(), local_llm_client=local_llm)

    reply = asyncio.run(agent.handle_message("kuantum bilgisayarlar hakkında ne düşünüyorsun"))

    assert reply == "Merhaba, ben yerel modelim."
    assert local_llm.calls == 1


def test_without_local_llm_configured_the_original_error_surfaces():
    registry = ToolRegistry()
    agent = Agent(_settings(), registry, llm_client=FailingLLM(), local_llm_client=None)

    reply = asyncio.run(agent.handle_message("kuantum bilgisayarlar hakkında ne düşünüyorsun"))

    assert "ulaşılamadı" in reply


def test_local_llm_failure_falls_back_to_the_original_claude_error():
    registry = ToolRegistry()
    agent = Agent(
        _settings(), registry, llm_client=FailingLLM(), local_llm_client=BrokenLocalLLM()
    )

    reply = asyncio.run(agent.handle_message("kuantum bilgisayarlar hakkında ne düşünüyorsun"))

    assert "ulaşılamadı" in reply


def test_a_failure_mid_tool_use_does_not_fall_back_to_local_llm():
    registry = ToolRegistry()
    tool_call = FakeMessage([FakeBlock(type="tool_use", name="nope", input={}, id="t1")])
    local_llm = StubLocalLLM()
    llm = FailingThenToolLLM(tool_call)
    agent = Agent(_settings(), registry, llm_client=llm, local_llm_client=local_llm)

    reply = asyncio.run(agent.handle_message("bir şey yap"))

    assert "ulaşılamadı" in reply
    assert local_llm.calls == 0


def test_to_plain_messages_drops_pure_tool_plumbing_and_flattens_text_blocks():
    messages = [
        {"role": "user", "content": "merhaba"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "get_time", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "14:00"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Saat 14:00."}]},
    ]

    plain = to_plain_messages(messages)

    assert plain == [
        {"role": "user", "content": "merhaba"},
        {"role": "assistant", "content": "Saat 14:00."},
    ]
