from __future__ import annotations

import asyncio
import ctypes
import logging
import subprocess

from .base import RiskLevel, Tool, ToolResult

logger = logging.getLogger(__name__)


class LockComputerTool(Tool):
    name = "lock_computer"
    description = "Bilgisayar ekranını kilitler (oturum açık kalır, veri kaybı olmaz)."
    risk = RiskLevel.MEDIUM
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **kwargs: object) -> ToolResult:
        try:
            await asyncio.to_thread(ctypes.windll.user32.LockWorkStation)
        except Exception:
            logger.exception("Ekran kilitlenemedi")
            return ToolResult(success=False, error="Ekranı kilitleyemedim.")
        return ToolResult(success=True, data={"locked": True})


class ShutdownComputerTool(Tool):
    name = "shutdown_computer"
    description = (
        "Bilgisayarı kapatır. Geri dönüşü olmayan bir işlemdir; kaydedilmemiş "
        "işler kaybolabilir. Mutlaka kullanıcı onayı gerektirir."
    )
    risk = RiskLevel.HIGH
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **kwargs: object) -> ToolResult:
        try:
            await asyncio.to_thread(
                subprocess.run, ["shutdown", "/s", "/t", "5"], check=True, capture_output=True
            )
        except Exception:
            logger.exception("Bilgisayar kapatılamadı")
            return ToolResult(success=False, error="Bilgisayarı kapatamadım.")
        return ToolResult(success=True, data={"shutting_down": True, "seconds": 5})


class RestartComputerTool(Tool):
    name = "restart_computer"
    description = (
        "Bilgisayarı yeniden başlatır. Geri dönüşü olmayan bir işlemdir; "
        "kaydedilmemiş işler kaybolabilir. Mutlaka kullanıcı onayı gerektirir."
    )
    risk = RiskLevel.HIGH
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **kwargs: object) -> ToolResult:
        try:
            await asyncio.to_thread(
                subprocess.run, ["shutdown", "/r", "/t", "5"], check=True, capture_output=True
            )
        except Exception:
            logger.exception("Bilgisayar yeniden başlatılamadı")
            return ToolResult(success=False, error="Bilgisayarı yeniden başlatamadım.")
        return ToolResult(success=True, data={"restarting": True, "seconds": 5})
