"""Agent integration with AccessModeManager: recognizing mode-switch
commands, injecting the current mode into the system prompt, and giving
Claude an honest reason when a HIGH-risk tool gets refused outright rather
than actually asked about.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neo.config.settings import Settings
from neo.core.access_mode import AccessMode, AccessModeManager
from neo.core.agent import Agent, _mode_context
from neo.core.permissions import PermissionManager
from neo.tools.base import RiskLevel, Tool, ToolRegistry, ToolResult


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
    def __init__(self, responses):
        self._responses = list(responses)

    def send(self, messages, system, tools, max_tokens=1024):
        self.last_system = system
        self.last_messages = messages
        return self._responses.pop(0)


class HighRiskTool(Tool):
    name = "shutdown_computer"
    description = "test"
    risk = RiskLevel.HIGH
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **kwargs):
        return ToolResult(success=True, data={"done": True})


def _settings():
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


# -- voice/text mode-switch commands ---------------------------------------


def test_assistant_mode_command_drops_from_helper():
    modes = AccessModeManager()
    modes.unlock_helper_mode()
    agent = Agent(_settings(), ToolRegistry(), llm_client=FakeLLM([]), mode_manager=modes)

    reply = asyncio.run(agent.handle_message("asistan moduna dön"))

    assert modes.mode is AccessMode.ASSISTANT
    assert "asistan" in reply.lower()


def test_helper_mode_request_without_a_wired_dialog_is_honest():
    modes = AccessModeManager()
    agent = Agent(_settings(), ToolRegistry(), llm_client=FakeLLM([]), mode_manager=modes)

    reply = asyncio.run(agent.handle_message("yardımcı moduna geç"))

    assert modes.mode is AccessMode.ASSISTANT
    assert "bağlanmadı" in reply


def test_helper_mode_request_without_a_mode_manager_is_honest():
    agent = Agent(_settings(), ToolRegistry(), llm_client=FakeLLM([]))

    reply = asyncio.run(agent.handle_message("yardımcı moduna geç"))

    assert "yapılandırılmadı" in reply


def test_helper_mode_request_calls_the_unlock_callback():
    modes = AccessModeManager()
    agent = Agent(_settings(), ToolRegistry(), llm_client=FakeLLM([]), mode_manager=modes)

    async def unlock():
        modes.unlock_helper_mode()
        return True

    agent.set_mode_unlock_control(unlock)
    reply = asyncio.run(agent.handle_message("yardımcı moduna geç"))

    assert modes.mode is AccessMode.HELPER
    assert "yetkiliyim" in reply.lower()


def test_a_failed_unlock_leaves_assistant_mode_active():
    modes = AccessModeManager()
    agent = Agent(_settings(), ToolRegistry(), llm_client=FakeLLM([]), mode_manager=modes)

    async def wrong_password():
        return False

    agent.set_mode_unlock_control(wrong_password)
    reply = asyncio.run(agent.handle_message("yardımcı moduna geç"))

    assert modes.mode is AccessMode.ASSISTANT
    assert "geçilmedi" in reply


def test_mode_commands_never_reach_the_llm():
    agent = Agent(
        _settings(), ToolRegistry(), llm_client=FakeLLM([]), mode_manager=AccessModeManager()
    )
    asyncio.run(agent.handle_message("asistan moduna dön"))


# -- system prompt injection ------------------------------------------------


def test_mode_context_empty_without_a_manager():
    assert _mode_context(None) == ""


def test_mode_context_names_assistant_mode_restrictions():
    context = _mode_context(AccessModeManager())
    assert "ASİSTAN" in context


def test_mode_context_shows_helper_mode_and_countdown():
    modes = AccessModeManager()
    modes.unlock_helper_mode()
    context = _mode_context(modes)
    assert "YARDIMCI" in context


def test_handle_message_injects_mode_context_into_system_prompt():
    modes = AccessModeManager()
    fake_llm = FakeLLM([FakeMessage([FakeBlock(type="text", text="Tamam efendim.")])])
    agent = Agent(_settings(), ToolRegistry(), llm_client=fake_llm, mode_manager=modes)

    asyncio.run(agent.handle_message("selam"))

    assert "ASİSTAN" in fake_llm.last_system


# -- HIGH-risk tool denial, real message accuracy ---------------------------


def test_high_risk_tool_denied_in_assistant_mode_gets_an_honest_error():
    modes = AccessModeManager()
    registry = ToolRegistry()
    registry.register(HighRiskTool())
    permissions = PermissionManager(mode_manager=modes)

    tool_call = FakeMessage(
        [FakeBlock(type="tool_use", name="shutdown_computer", input={}, id="t1")]
    )
    final = FakeMessage([FakeBlock(type="text", text="Yapamadım.")])
    fake_llm = FakeLLM([tool_call, final])

    agent = Agent(
        _settings(), registry, permissions=permissions, llm_client=fake_llm, mode_manager=modes
    )
    asyncio.run(agent.handle_message("bilgisayarı kapat"))

    tool_result_message = agent._context.messages[-2]
    content = tool_result_message["content"][0]["content"]
    assert "sorulmadı" in content
    assert "onaylamadı" not in content


def test_high_risk_tool_denial_message_in_helper_mode_is_the_generic_one():
    modes = AccessModeManager()
    modes.unlock_helper_mode()
    registry = ToolRegistry()
    registry.register(HighRiskTool())

    async def deny(name, desc):
        return False

    permissions = PermissionManager(confirm=deny, mode_manager=modes)

    tool_call = FakeMessage(
        [FakeBlock(type="tool_use", name="shutdown_computer", input={}, id="t1")]
    )
    final = FakeMessage([FakeBlock(type="text", text="Tamam, yapmadım.")])
    fake_llm = FakeLLM([tool_call, final])

    agent = Agent(
        _settings(), registry, permissions=permissions, llm_client=fake_llm, mode_manager=modes
    )
    asyncio.run(agent.handle_message("bilgisayarı kapat"))

    tool_result_message = agent._context.messages[-2]
    content = tool_result_message["content"][0]["content"]
    assert "onaylamadı" in content


def test_high_risk_tool_actually_runs_in_helper_mode_with_confirmation():
    modes = AccessModeManager()
    modes.unlock_helper_mode()
    registry = ToolRegistry()
    registry.register(HighRiskTool())

    async def approve(name, desc):
        return True

    permissions = PermissionManager(confirm=approve, mode_manager=modes)

    tool_call = FakeMessage(
        [FakeBlock(type="tool_use", name="shutdown_computer", input={}, id="t1")]
    )
    final = FakeMessage([FakeBlock(type="text", text="Kapatıyorum.")])
    fake_llm = FakeLLM([tool_call, final])

    agent = Agent(
        _settings(), registry, permissions=permissions, llm_client=fake_llm, mode_manager=modes
    )
    reply = asyncio.run(agent.handle_message("bilgisayarı kapat"))

    assert reply == "Kapatıyorum."


def test_agent_without_a_mode_manager_still_works_exactly_as_before():
    fake_llm = FakeLLM([FakeMessage([FakeBlock(type="text", text="Merhaba.")])])
    agent = Agent(_settings(), ToolRegistry(), llm_client=fake_llm)

    reply = asyncio.run(agent.handle_message("selam"))

    assert reply == "Merhaba."
    assert "Yetki modu" not in fake_llm.last_system
