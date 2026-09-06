import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neo.config.settings import Settings
from neo.core.agent import Agent
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


def test_agent_executes_tool_then_returns_text():
    registry = ToolRegistry()
    registry.register(GetTimeTool())

    tool_call = FakeMessage([FakeBlock(type="tool_use", name="get_time", input={}, id="t1")])
    final = FakeMessage([FakeBlock(type="text", text="Saat şu an.")])
    fake_llm = FakeLLM([tool_call, final])

    agent = Agent(_settings(), registry, llm_client=fake_llm)
    reply = asyncio.run(agent.handle_message("bana bir tavsiye ver"))

    assert reply == "Saat şu an."


def test_agent_returns_plain_text_without_tools():
    registry = ToolRegistry()
    final = FakeMessage([FakeBlock(type="text", text="Merhaba!")])
    fake_llm = FakeLLM([final])

    agent = Agent(_settings(), registry, llm_client=fake_llm)
    reply = asyncio.run(agent.handle_message("selam"))

    assert reply == "Merhaba!"


def test_agent_without_api_key_and_no_injected_client_returns_friendly_error():
    registry = ToolRegistry()
    agent = Agent(_settings(), registry)

    reply = asyncio.run(agent.handle_message("selam"))

    assert "ANTHROPIC_API_KEY" in reply


def test_agent_answers_local_command_without_any_llm_call():
    registry = ToolRegistry()
    registry.register(GetTimeTool())

    # No API key and no injected LLM client at all -- if this reaches the LLM
    # path it would fail with a ConfigError message, not a "Saat ..." reply.
    agent = Agent(_settings(), registry)

    reply = asyncio.run(agent.handle_message("saat kaç"))

    assert reply.startswith("Saat ")
