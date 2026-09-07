from __future__ import annotations

import asyncio

from ..memory.audit_store import AuditStore
from .base import RiskLevel, Tool, ToolResult


class GetRecentActivityTool(Tool):
    name = "get_recent_activity"
    description = (
        "Kullanıcı 'son 24 saatte ne yaptın', 'hangi araçları çalıştırdın' "
        "gibi bir şey sorduğunda, NEO'nun gerçekten çalıştırdığı (ya da "
        "reddedilen) araçların kaydını döner."
    )
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "hours": {
                "type": "number",
                "description": "Kaç saat geriye bakılacağı (varsayılan 24).",
            },
        },
    }

    def __init__(self, store: AuditStore) -> None:
        self._store = store

    async def run(self, hours: float = 24.0, **kwargs: object) -> ToolResult:
        entries = await asyncio.to_thread(self._store.recent, hours)
        return ToolResult(
            success=True,
            data={
                "hours": hours,
                "count": len(entries),
                "entries": [
                    {
                        "tool_name": e.tool_name,
                        "risk": e.risk,
                        "input_summary": e.input_summary,
                        "success": e.success,
                        "error": e.error,
                        "created_at": e.created_at,
                    }
                    for e in entries
                ],
            },
        )
