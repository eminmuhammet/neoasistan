import asyncio

from neo.tools.base import RiskLevel
from neo.tools.planning import RunTaskTool


class FakePlanner:
    def __init__(self, result):
        self._result = result
        self.calls = []

    async def run(self, goal):
        self.calls.append(goal)
        return self._result


def test_run_task_is_low_risk():
    """Orchestration itself is safe -- each step's own tool call carries
    its own risk and goes through its own permission check as it runs."""
    assert RunTaskTool(FakePlanner((True, ""))).risk is RiskLevel.LOW


def test_successful_task_returns_a_successful_tool_result():
    planner = FakePlanner((True, "Görevi tamamladım: ..."))
    result = asyncio.run(RunTaskTool(planner).run(goal="ekrana bak ve özetle"))

    assert result.success is True
    assert "tamamladım" in result.data["summary"]
    assert planner.calls == ["ekrana bak ve özetle"]


def test_failed_task_returns_an_unsuccessful_tool_result():
    """The outer conversation needs to see success=False to relay the
    failure honestly, not just infer it from parsing the summary text."""
    planner = FakePlanner((False, "Görevi tamamlayamadım: ..."))
    result = asyncio.run(RunTaskTool(planner).run(goal="imkansız bir şey"))

    assert result.success is False
    assert "tamamlayamadım" in result.data["summary"]


def test_goal_is_passed_through_unchanged():
    planner = FakePlanner((True, "ok"))
    asyncio.run(RunTaskTool(planner).run(goal="tam olarak bu cümle"))

    assert planner.calls == ["tam olarak bu cümle"]
