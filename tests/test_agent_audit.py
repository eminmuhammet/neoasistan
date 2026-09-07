"""Agent integration with AuditStore: a denied tool call must leave a
trail. (Successful calls are logged by ToolRegistry's audit hook instead --
see test_tool_registry_audit.py -- so every caller that goes through
registry.execute is covered, not just Agent's own tool loop.)
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from neo.config.settings import Settings
from neo.core.agent import Agent
from neo.core.permissions import PermissionManager
from neo.memory.audit_store import AuditStore
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
        return self._responses.pop(0)


class EchoTool(Tool):
    name = "echo"
    description = "test"
    risk = RiskLevel.LOW
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **kwargs):
        return ToolResult(success=True, data={"ok": True})


class AlwaysDeniedTool(Tool):
    name = "dangerous"
    description = "test"
    risk = RiskLevel.HIGH
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **kwargs):
        return ToolResult(success=True, data={})


def _settings():
    return Settings(
        anthropic_api_key=None, model="test-model", default_city="Bursa",
        log_level="INFO", log_dir=Path("."), data_dir=Path("."),
        whisper_model="tiny", whisper_device="cpu", conversation_dir=Path("."),
    )


async def _never_confirm(tool_name, description):
    return False


def test_successful_tool_call_is_not_double_logged_by_agent(tmp_path):
    """Agent no longer logs successful calls itself -- that's
    ToolRegistry's job now (its audit hook covers every caller, not just
    Agent's own tool loop). Passing audit_store here must not produce a
    duplicate entry for a call that went through registry.execute."""
    registry = ToolRegistry()
    registry.register(EchoTool())
    audit = AuditStore(tmp_path / "audit.db")
    llm = FakeLLM([
        FakeMessage([FakeBlock(type="tool_use", name="echo", input={"a": 1}, id="t1")]),
        FakeMessage([FakeBlock(type="text", text="tamam")]),
    ])
    agent = Agent(_settings(), registry, llm_client=llm, audit_store=audit)

    asyncio.run(agent.handle_message("echo yap"))

    assert audit.recent() == []


def test_denied_tool_call_is_also_logged(tmp_path):
    registry = ToolRegistry()
    registry.register(AlwaysDeniedTool())
    audit = AuditStore(tmp_path / "audit.db")
    permissions = PermissionManager(confirm=_never_confirm)
    llm = FakeLLM([
        FakeMessage([FakeBlock(type="tool_use", name="dangerous", input={}, id="t1")]),
        FakeMessage([FakeBlock(type="text", text="tamam")]),
    ])
    agent = Agent(_settings(), registry, llm_client=llm, permissions=permissions, audit_store=audit)

    asyncio.run(agent.handle_message("tehlikeli şeyi yap"))

    entries = audit.recent()
    assert len(entries) == 1
    assert entries[0].tool_name == "dangerous"
    assert entries[0].success is False


def test_no_audit_store_does_not_break_the_tool_loop():
    registry = ToolRegistry()
    registry.register(EchoTool())
    llm = FakeLLM([
        FakeMessage([FakeBlock(type="tool_use", name="echo", input={}, id="t1")]),
        FakeMessage([FakeBlock(type="text", text="tamam")]),
    ])
    agent = Agent(_settings(), registry, llm_client=llm)

    reply = asyncio.run(agent.handle_message("echo yap"))

    assert reply == "tamam"
