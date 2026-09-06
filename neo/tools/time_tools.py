from __future__ import annotations

from datetime import datetime

from .base import RiskLevel, Tool, ToolResult

TR_DAYS = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
TR_MONTHS = [
    "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
    "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
]


class GetTimeTool(Tool):
    name = "get_time"
    description = "Bilgisayarın yerel saatini döndürür."
    risk = RiskLevel.LOW
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **kwargs: object) -> ToolResult:
        now = datetime.now()
        return ToolResult(success=True, data={"time": now.strftime("%H:%M:%S")})


class GetDateTool(Tool):
    name = "get_date"
    description = "Bilgisayarın yerel tarihini döndürür."
    risk = RiskLevel.LOW
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **kwargs: object) -> ToolResult:
        now = datetime.now()
        return ToolResult(
            success=True,
            data={
                "date": now.strftime("%d.%m.%Y"),
                "weekday": TR_DAYS[now.weekday()],
                "day": now.day,
                "month": TR_MONTHS[now.month - 1],
                "year": now.year,
            },
        )
