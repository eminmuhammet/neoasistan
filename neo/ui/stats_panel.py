from __future__ import annotations

import asyncio

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QWidget

from ..tools.system_info import (
    GetCpuUsageTool,
    GetGpuStatusTool,
    GetRamUsageTool,
    get_own_usage,
)
from .radial_gauge import RadialGauge

REFRESH_INTERVAL_MS = 2500


class StatsPanel(QWidget):
    """Live CPU/RAM/GPU HUD gauges. Reads straight from the same local
    tools the Agent uses (psutil / nvidia-ml-py) on a timer -- no LLM call
    involved, so it keeps working with no API credit and costs nothing to
    refresh.

    The gauges show whole-machine load, and each one also carries NEO's own
    share underneath. Without that second number the panel is genuinely
    misleading: sitting inside NEO's own window, a "66%" RAM gauge reads as
    NEO's usage, when the process actually held 577 MB of 15.8 GB and the
    rest was everything else running.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cpu_tool = GetCpuUsageTool()
        self._ram_tool = GetRamUsageTool()
        self._gpu_tool = GetGpuStatusTool()
        self._gpu_available = True

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(12)

        self._cpu_gauge = RadialGauge("CPU", color="#23c9ff")
        self._ram_gauge = RadialGauge("RAM", color="#35e08a")
        self._gpu_gauge = RadialGauge("GPU", color="#f2b134")
        for gauge in (self._cpu_gauge, self._ram_gauge, self._gpu_gauge):
            layout.addWidget(gauge)

        self._refreshing = False   # reentrancy guard (see _schedule_refresh)

        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_INTERVAL_MS)
        self._timer.timeout.connect(self._schedule_refresh)
        self._timer.start()
        self._schedule_refresh()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()
        self._schedule_refresh()

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # Polling CPU/RAM/GPU (and repainting gauges) while the window sits
        # hidden in the tray is pure waste -- NEO is meant to idle cheaply.
        self._timer.stop()
        super().hideEvent(event)

    def _schedule_refresh(self) -> None:
        # Guard against concurrent refreshes: if the previous _refresh() coroutine
        # hasn't finished (e.g. GPU query blocked, wake-command task is running)
        # skip this tick rather than spawning a second task -- qasync raises
        # "Cannot enter into task" when two tasks collide inside its event loop.
        if self._refreshing:
            return
        self._refreshing = True
        asyncio.ensure_future(self._refresh())

    async def _refresh(self) -> None:
        try:
            cpu = await self._cpu_tool.run()
            if cpu.success:
                self._cpu_gauge.set_value(cpu.data["cpu_percent"])

            ram = await self._ram_tool.run()
            if ram.success:
                self._ram_gauge.set_value(ram.data["percent"])

            own = await asyncio.to_thread(get_own_usage)
            if own is not None:
                self._cpu_gauge.set_extra_text(f"NEO %{own['cpu_percent']:.1f}")
                self._ram_gauge.set_extra_text(f"NEO {own['memory_mb']:.0f}MB")
            else:
                # Never fill these in with a guess -- an empty slot is honest.
                self._cpu_gauge.set_extra_text("")
                self._ram_gauge.set_extra_text("")

            gpu = await self._gpu_tool.run()
            if gpu.success:
                self._gpu_available = True
                self._gpu_gauge.set_value(gpu.data["gpu_percent"])
                temp = gpu.data.get("temperature_c")
                self._gpu_gauge.set_extra_text(f"{temp}°C" if temp is not None else "")
            elif self._gpu_available:
                # Only flip to the "yok" state once, and never fake a number --
                # honest > pretty here.
                self._gpu_available = False
                self._gpu_gauge.set_value(0)
                self._gpu_gauge.set_extra_text("yok")
        finally:
            self._refreshing = False
