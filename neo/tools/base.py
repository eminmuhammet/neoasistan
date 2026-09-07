from __future__ import annotations

import abc
import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

AuditHook = Callable[[str, "RiskLevel | None", dict, bool, "str | None"], Awaitable[None]]

logger = logging.getLogger(__name__)


class RiskLevel(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ToolResult:
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {"success": self.success, **self.data}
        if self.error:
            payload["error"] = self.error
        return payload


class Tool(abc.ABC):
    name: str
    description: str
    risk: RiskLevel = RiskLevel.LOW
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    @abc.abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult:
        ...

    def to_anthropic_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._audit_hook: AuditHook | None = None

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def anthropic_tools(self) -> list[dict[str, Any]]:
        return [t.to_anthropic_schema() for t in self._tools.values()]

    def set_audit_hook(self, hook: AuditHook | None) -> None:
        """Every caller (the agent's own tool loop, local_commands.py's
        fast path, the scheduler, the planner) goes through execute() --
        wiring the audit log in here rather than in each caller is the only
        way "her araç çalıştırması kaydedilir" is actually true for all of
        them, not just the ones that happen to go through Agent."""
        self._audit_hook = hook

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Bilinmeyen araç: {name}")
        try:
            result = await tool.run(**arguments)
        except Exception:
            logger.exception("Tool execution failed: %s", name)
            result = ToolResult(
                success=False, error=f"'{name}' aracı çalıştırılırken bir hata oluştu."
            )
        if self._audit_hook is not None:
            await self._audit_hook(name, tool.risk, arguments, result.success, result.error)
        return result
