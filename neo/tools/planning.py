from __future__ import annotations

from ..core.planner import TaskPlanner
from .base import RiskLevel, Tool, ToolResult


class RunTaskTool(Tool):
    name = "run_task"
    description = (
        "Birden fazla adım gerektiren ve her adımdan sonra sonucun "
        "gerçekten doğru olup olmadığının kontrol edilmesi gereken bir "
        "hedefi başlatır (ör. \"ekrana bak, hatayı bul, düzelt, tekrar "
        "ekrana bakıp kontrol et\"). Basit, tek adımlık isteklerde bu aracı "
        "KULLANMA -- onları doğrudan, normal şekilde yap. 'goal' hedefi "
        "açık ve somut bir cümleyle anlat."
    )
    # Orchestration itself changes nothing on its own -- it only decides
    # what to try and checks the result. Each step's own tool call is
    # separately risk-checked exactly as it would be outside a task, so a
    # HIGH-risk action inside a task still needs its own confirmation (or
    # is refused outright in assistant mode) when that step actually runs.
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "Gerçekleştirilecek genel hedef, açık ve somut.",
            }
        },
        "required": ["goal"],
    }

    def __init__(self, planner: TaskPlanner) -> None:
        self._planner = planner

    async def run(self, goal: str, **kwargs: object) -> ToolResult:
        success, summary = await self._planner.run(goal)
        return ToolResult(success=success, data={"summary": summary})
