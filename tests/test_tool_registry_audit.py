"""ToolRegistry's audit hook: the single chokepoint every caller (Agent's
tool loop, local_commands.py's fast path, the scheduler, the planner) goes
through -- this is what makes "every tool run is logged" actually true for
all of them, not just the ones that happen to go through Agent.
"""

import asyncio

from neo.core.local_commands import try_handle_locally
from neo.tools.base import RiskLevel, Tool, ToolRegistry, ToolResult


class EchoTool(Tool):
    name = "echo"
    description = "test"
    risk = RiskLevel.LOW
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **kwargs):
        return ToolResult(success=True, data={"ok": True})


class FailingTool(Tool):
    name = "fails"
    description = "test"
    risk = RiskLevel.MEDIUM
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **kwargs):
        return ToolResult(success=False, error="nope")


def _recording_hook(calls):
    async def hook(tool_name, risk, tool_input, success, error):
        calls.append((tool_name, risk, tool_input, success, error))
    return hook


def test_execute_calls_the_audit_hook_on_success():
    registry = ToolRegistry()
    registry.register(EchoTool())
    calls = []
    registry.set_audit_hook(_recording_hook(calls))

    asyncio.run(registry.execute("echo", {"a": 1}))

    assert calls == [("echo", RiskLevel.LOW, {"a": 1}, True, None)]


def test_execute_calls_the_audit_hook_on_failure():
    registry = ToolRegistry()
    registry.register(FailingTool())
    calls = []
    registry.set_audit_hook(_recording_hook(calls))

    asyncio.run(registry.execute("fails", {}))

    assert calls == [("fails", RiskLevel.MEDIUM, {}, False, "nope")]


def test_execute_without_a_hook_still_works():
    registry = ToolRegistry()
    registry.register(EchoTool())

    result = asyncio.run(registry.execute("echo", {}))

    assert result.success is True


def test_unknown_tool_is_not_sent_to_the_audit_hook():
    registry = ToolRegistry()
    calls = []
    registry.set_audit_hook(_recording_hook(calls))

    asyncio.run(registry.execute("no_such_tool", {}))

    assert calls == []


def test_local_fast_path_time_command_is_also_audited():
    """The real gap a live test caught: 'saat kaç?' is answered by
    local_commands.py's fast path, which calls registry.execute directly
    and never goes through Agent at all -- so only a hook living in
    ToolRegistry itself can see it."""
    registry = ToolRegistry()
    registry.register(EchoToolNamedGetTime())
    calls = []
    registry.set_audit_hook(_recording_hook(calls))

    from neo.core import local_commands

    original = local_commands.match_local_command

    class _FakeCommand:
        tool_name = "get_time"

        @staticmethod
        def formatter(result):
            return "test"

    local_commands.match_local_command = lambda text: _FakeCommand()
    try:
        asyncio.run(try_handle_locally("saat kaç", registry))
    finally:
        local_commands.match_local_command = original

    assert calls and calls[0][0] == "get_time"


class EchoToolNamedGetTime(Tool):
    name = "get_time"
    description = "test"
    risk = RiskLevel.LOW
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **kwargs):
        return ToolResult(success=True, data={"time": "12:00"})
