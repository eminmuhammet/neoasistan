from __future__ import annotations

import asyncio
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# Disk fill is slow; nothing is lost by checking every few minutes rather
# than constantly, and less frequent checks mean less background work in an
# app meant to idle cheaply.
DEFAULT_CHECK_SECONDS = 300.0
DEFAULT_THRESHOLD_PERCENT = 90.0

OnAlert = Callable[[str], None]


class DiskSpaceWatcher:
    """A "kritik uyarı" per the plan: watches disk usage and warns once a
    drive crosses above a fill threshold.

    Deliberately calls no LLM at all -- a percentage-over-threshold
    comparison needs no judgment, and this runs on a timer forever, so it
    should cost nothing. Edge-triggered: it fires once at the moment a
    drive crosses over, not on every check while it stays over, which would
    just be nagging about something already reported.
    """

    def __init__(
        self,
        on_alert: OnAlert,
        threshold_percent: float = DEFAULT_THRESHOLD_PERCENT,
        check_seconds: float = DEFAULT_CHECK_SECONDS,
    ) -> None:
        self._on_alert = on_alert
        self._threshold = threshold_percent
        self._check_seconds = check_seconds
        self._qtimer = None
        self._already_over: set[str] = set()

    def start(self) -> None:
        from PySide6.QtCore import QTimer

        if self._qtimer is not None:
            return
        self._qtimer = QTimer()
        self._qtimer.setInterval(int(self._check_seconds * 1000))
        self._qtimer.timeout.connect(lambda: asyncio.ensure_future(self.check()))
        self._qtimer.start()

    def stop(self) -> None:
        if self._qtimer is not None:
            self._qtimer.stop()
            self._qtimer = None

    async def check(self) -> None:
        from ..tools.system_info import GetDiskUsageTool

        result = await GetDiskUsageTool().run()
        if not result.success:
            # Never fabricate a warning (or its absence) when the read
            # itself failed -- just skip this check and try again next tick.
            logger.warning("Disk kullanımı okunamadı, kontrol atlandı")
            return

        currently_over: set[str] = set()
        for drive in result.data["drives"]:
            name = drive["drive"]
            if drive["percent_used"] < self._threshold:
                continue
            currently_over.add(name)
            if name not in self._already_over:
                self._on_alert(
                    f"{name} sürücüsü %{drive['percent_used']:.0f} dolu, "
                    f"sadece {drive['free_gb']} GB boş alan kaldı."
                )

        self._already_over = currently_over
