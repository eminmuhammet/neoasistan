from __future__ import annotations

from typing import Awaitable, Callable

from .access_mode import AccessMode, AccessModeManager
from ..tools.base import RiskLevel

ConfirmCallback = Callable[[str, str], Awaitable[bool]]


class PermissionManager:
    """Gates tool execution by risk tier -- and, once a mode manager is
    wired in, by NEO's current authority level too.

    LOW risk tools always run without confirmation. Without a mode manager,
    MEDIUM and HIGH both fall back to the original rule this class shipped
    with: confirm if a callback is wired, deny otherwise. That fallback is
    deliberate, not a placeholder -- every existing caller that constructs
    PermissionManager() without a mode manager (including tests written
    before mode support existed) must keep seeing exactly that behavior.

    With a mode manager wired, the fuller table applies:

        | risk   | assistant mode      | helper mode         |
        |--------|----------------------|----------------------|
        | LOW    | free                 | free                 |
        | MEDIUM | confirm              | free                 |
        | HIGH   | refused, no dialog   | confirm              |

    HIGH-risk actions in assistant mode are refused before a confirmation
    dialog is even shown: the plan behind this is that "full authority"
    should be a deliberate mode switch, not something clickable out of habit
    on a dialog that happens to be on screen.
    """

    def __init__(
        self,
        confirm: ConfirmCallback | None = None,
        mode_manager: AccessModeManager | None = None,
    ) -> None:
        self._confirm = confirm
        self._mode_manager = mode_manager

    def set_confirm(self, confirm: ConfirmCallback | None) -> None:
        """Wires (or clears) the confirmation callback after construction --
        useful when the callback lives on a GUI object created after the
        Agent/PermissionManager (e.g. MainWindow)."""
        self._confirm = confirm

    def set_mode_manager(self, mode_manager: AccessModeManager | None) -> None:
        self._mode_manager = mode_manager

    async def check(self, tool_name: str, risk: RiskLevel, description: str) -> bool:
        if risk == RiskLevel.LOW:
            return True

        if self._mode_manager is not None:
            mode = self._mode_manager.mode
            if risk == RiskLevel.HIGH and mode is AccessMode.ASSISTANT:
                return False
            if risk == RiskLevel.MEDIUM and mode is AccessMode.HELPER:
                return True

        if self._confirm is None:
            return False
        return await self._confirm(tool_name, description)
