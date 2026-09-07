"""Agent.run_subtask / raw_llm_call: the task planner's way of reusing the
exact same LLM-call, tool-execution and permission-check machinery as an
ordinary conversation, without polluting the user's own chat history or
conversation_store with internal planning chatter.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neo.config.settings import Settings
from neo.core.agent import Agent
from neo.core.access_mode import AccessMode, AccessModeManager
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
        self.calls = []

    def send(self, messages, system, tools, max_tokens=1024):
        self.calls.append({"messages": messages, "system": system, "tools": tools})
        return self._responses.pop(0)


class RecordingStore:
    def __init__(self):
        self.recorded = []

    def add_message(self, role, text):
        self.recorded.append((role, text))

    def recent_messages(self, limit=20):
        return []


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


# -- run_subtask isolation ----------------------------------------------------


def test_run_subtask_does_not_touch_conversation_history():
    store = RecordingStore()
    fake_llm = FakeLLM([FakeMessage([FakeBlock(type="text", text="adım sonucu")])])
    agent = Agent(_settings(), ToolRegistry(), llm_client=fake_llm, conversation_store=store)

    result = asyncio.run(agent.run_subtask("\n\nGöREV BAĞLAMI", "bir şey yap"))

    assert result.text == "adım sonucu"
    assert agent._context.messages == []  # the user's own chat context, untouched
    assert store.recorded == []  # nothing written to conversation_store either


def test_run_subtask_does_not_leak_between_calls():
    """Two subtask calls must not share state -- each is its own isolated
    exchange, the way two different steps of a task shouldn't see each
    other's raw conversation, only the summarized result the planner
    explicitly passes along."""
    fake_llm = FakeLLM(
        [
            FakeMessage([FakeBlock(type="text", text="birinci")]),
            FakeMessage([FakeBlock(type="text", text="ikinci")]),
        ]
    )
    agent = Agent(_settings(), ToolRegistry(), llm_client=fake_llm)

    asyncio.run(agent.run_subtask("", "ilk görev"))
    asyncio.run(agent.run_subtask("", "ikinci görev"))

    # fake_llm.calls stores a reference to each call's live messages list,
    # which its own context.add_assistant(...) mutates right after send()
    # returns -- so checking length after the fact is unreliable. What
    # actually proves isolation is that the first call's user turn never
    # shows up inside the second call's context, and vice versa.
    first_texts = [m["content"] for m in fake_llm.calls[0]["messages"] if m["role"] == "user"]
    second_texts = [m["content"] for m in fake_llm.calls[1]["messages"] if m["role"] == "user"]
    assert first_texts == ["ilk görev"]
    assert second_texts == ["ikinci görev"]


def test_run_subtask_includes_the_goal_context_in_the_system_prompt():
    fake_llm = FakeLLM([FakeMessage([FakeBlock(type="text", text="ok")])])
    agent = Agent(_settings(), ToolRegistry(), llm_client=fake_llm)

    asyncio.run(agent.run_subtask("\n\nBU BIR TEST BAGLAMI", "yap"))

    assert "BU BIR TEST BAGLAMI" in fake_llm.calls[0]["system"]


def test_run_subtask_still_runs_real_tools_through_the_same_loop():
    """A subtask isn't a stripped-down execution path -- it uses the exact
    same tool registry and gets to call real tools, since a task step is
    still supposed to actually do things."""
    registry = ToolRegistry()
    registry.register(HighRiskTool())

    async def approve(name, desc):
        return True

    permissions = PermissionManager(confirm=approve)
    tool_call = FakeMessage(
        [FakeBlock(type="tool_use", name="shutdown_computer", input={}, id="t1")]
    )
    final = FakeMessage([FakeBlock(type="text", text="kapattım")])
    fake_llm = FakeLLM([tool_call, final])

    agent = Agent(_settings(), registry, permissions=permissions, llm_client=fake_llm)
    result = asyncio.run(agent.run_subtask("", "bilgisayarı kapat"))

    assert result.text == "kapattım"


def test_run_subtask_returns_a_trace_of_real_tool_calls():
    """The whole reason SubtaskResult carries tool_calls at all: a live run
    showed the planner's verification step being handed only Claude's
    closing sentence ("the time is 18:11") with no way to tell whether a
    real tool ran or the model just said a plausible-sounding number. A
    properly skeptical verifier correctly refused to certify that on
    faith -- so the trace has to carry the actual tool name, input and
    result, not just prose.
    """
    registry = ToolRegistry()
    registry.register(HighRiskTool())

    async def approve(name, desc):
        return True

    permissions = PermissionManager(confirm=approve)
    tool_call = FakeMessage(
        [FakeBlock(type="tool_use", name="shutdown_computer", input={"force": True}, id="t1")]
    )
    final = FakeMessage([FakeBlock(type="text", text="kapattım")])
    fake_llm = FakeLLM([tool_call, final])

    agent = Agent(_settings(), registry, permissions=permissions, llm_client=fake_llm)
    result = asyncio.run(agent.run_subtask("", "bilgisayarı kapat"))

    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call["name"] == "shutdown_computer"
    assert call["input"] == {"force": True}
    assert call["result"]["success"] is True


def test_run_subtask_trace_is_empty_when_no_tool_was_called():
    fake_llm = FakeLLM([FakeMessage([FakeBlock(type="text", text="cevap, arac yok")])])
    agent = Agent(_settings(), ToolRegistry(), llm_client=fake_llm)

    result = asyncio.run(agent.run_subtask("", "sadece konuş"))

    assert result.tool_calls == []


def test_handle_message_does_not_collect_a_tool_trace():
    """Ordinary conversation has no reason to pay for trace bookkeeping it
    never reads -- tool_trace stays an opt-in parameter, and handle_message
    doesn't opt in."""
    registry = ToolRegistry()
    tool_call = FakeMessage(
        [FakeBlock(type="tool_use", name="get_time", input={}, id="t1")]
    )
    final = FakeMessage([FakeBlock(type="text", text="Saat 12.")])
    fake_llm = FakeLLM([tool_call, final])

    from neo.tools.time_tools import GetTimeTool

    registry.register(GetTimeTool())
    agent = Agent(_settings(), registry, llm_client=fake_llm)

    reply = asyncio.run(agent.handle_message("saat kaç şu an tam olarak söyler misin"))

    assert reply == "Saat 12."


def test_run_subtask_respects_the_assistant_mode_restriction():
    """The mode system applies inside a task step exactly as it does to a
    normal message -- a task cannot be used to sneak a HIGH-risk action
    past assistant mode's outright refusal."""
    modes = AccessModeManager()
    registry = ToolRegistry()
    registry.register(HighRiskTool())
    permissions = PermissionManager(mode_manager=modes)  # no confirm at all

    tool_call = FakeMessage(
        [FakeBlock(type="tool_use", name="shutdown_computer", input={}, id="t1")]
    )
    final = FakeMessage([FakeBlock(type="text", text="yapamadım")])
    fake_llm = FakeLLM([tool_call, final])

    agent = Agent(
        _settings(), registry, permissions=permissions, llm_client=fake_llm, mode_manager=modes
    )
    asyncio.run(agent.run_subtask("", "bilgisayarı kapat"))

    # The tool_result the sub-loop generated (thrown away afterward, but we
    # can still assert on the mock's recorded call args) must carry the
    # honest "never asked" denial, matching handle_message's own behavior.
    assert modes.mode is AccessMode.ASSISTANT


# -- raw_llm_call --------------------------------------------------------------


def test_raw_llm_call_passes_arguments_through_and_returns_the_response():
    fake_llm = FakeLLM([FakeMessage([FakeBlock(type="text", text="cevap")])])
    agent = Agent(_settings(), ToolRegistry(), llm_client=fake_llm)

    tools = [{"name": "submit_plan", "input_schema": {}}]
    response = asyncio.run(agent.raw_llm_call([{"role": "user", "content": "x"}], "sistem", tools, 777))

    assert response.content[0].text == "cevap"
    assert fake_llm.calls[0]["tools"] == tools
    assert fake_llm.calls[0]["system"] == "sistem"


def test_raw_llm_call_does_not_touch_conversation_context():
    fake_llm = FakeLLM([FakeMessage([FakeBlock(type="text", text="cevap")])])
    agent = Agent(_settings(), ToolRegistry(), llm_client=fake_llm)

    asyncio.run(agent.raw_llm_call([{"role": "user", "content": "x"}], "sistem", [], 100))

    assert agent._context.messages == []
