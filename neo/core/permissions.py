from __future__ import annotations

from typing import Awaitable, Callable

from ..tools.base import RiskLevel

ConfirmCallback = Callable[[str, str], Awaitable[bool]]


class PermissionManager:
    """Gates tool execution by risk tier.

    LOW risk tools run without confirmation. MEDIUM/HIGH risk tools require an
    async confirm callback (wired to a GUI dialog); if none is configured they
    are denied by default rather than silently allowed.
    """

    def __init__(self, confirm: ConfirmCallback | None = None) -> None:
        self._confirm = confirm

    def set_confirm(self, confirm: ConfirmCallback | None) -> None:
        """Wires (or clears) the confirmation callback after construction --
        useful when the callback lives on a GUI object created after the
        Agent/PermissionManager (e.g. MainWindow)."""
        self._confirm = confirm

    async def check(self, tool_name: str, risk: RiskLevel, description: str) -> bool:
        if risk == RiskLevel.LOW:
            return True
        if self._confirm is None:
            return False
        return await self._confirm(tool_name, description)
