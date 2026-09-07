from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class AccessMode(Enum):
    ASSISTANT = "assistant"
    HELPER = "helper"


ModeChangedCallback = Callable[[AccessMode], None]

# 10 minutes, per the plan: long enough that a real working session in
# helper mode doesn't get interrupted mid-task, short enough that walking
# away from the computer doesn't leave full authority open indefinitely.
DEFAULT_IDLE_TIMEOUT_SECONDS = 600.0


class AccessModeManager:
    """Tracks NEO's current authority level.

    Starts, and always eventually falls back to, ASSISTANT -- the safe
    default where HIGH-risk actions are refused outright and MEDIUM-risk
    ones need a live confirmation. HELPER (full authority) only exists
    behind a password and drops back to ASSISTANT on its own after a period
    of inactivity, so a moment of full access doesn't stay open by
    accident once the user walks away.

    This is a guard against *accidental* access (a housemate, a curious
    kid, forgetting a session was unlocked) -- not a defense against
    someone with real access to the machine, who could edit files directly
    regardless. The real boundary is still the Windows user account.
    """

    def __init__(
        self,
        idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
        on_change: ModeChangedCallback | None = None,
    ) -> None:
        self._mode = AccessMode.ASSISTANT
        self._idle_timeout = idle_timeout_seconds
        self._last_activity = time.monotonic()
        self._on_change = on_change

    @property
    def mode(self) -> AccessMode:
        """Reading this is also when expiry is checked -- there's no
        background timer ticking on its own, just a check-on-access, which
        is enough since every real caller (PermissionManager.check,
        the system prompt builder) reads this right before it matters."""
        self._maybe_expire()
        return self._mode

    def touch(self) -> None:
        """Resets the idle countdown. Called on every interaction (a voice
        command, a typed message, a button press) rather than only on
        privileged actions, so simply talking to NEO keeps helper mode
        alive -- someone actively using it isn't "idle" just because their
        last message happened not to need elevated authority."""
        self._last_activity = time.monotonic()

    def seconds_until_drop(self) -> float | None:
        """None in assistant mode (nothing counting down). Otherwise how
        long until helper mode expires on its own."""
        if self._mode is not AccessMode.HELPER:
            return None
        return max(0.0, self._idle_timeout - self._idle_seconds())

    def set_on_change(self, callback: ModeChangedCallback | None) -> None:
        """Lets a caller wire the mode-changed notification after this
        manager already exists -- needed because the GUI that wants to
        display the current mode is built after PermissionManager/Agent,
        which both need a mode_manager instance up front. Same post-hoc
        wiring pattern as PermissionManager.set_confirm."""
        self._on_change = callback

    def unlock_helper_mode(self) -> None:
        """Switches to helper mode. Callers must have already verified the
        password themselves (see config.credentials.PasswordStore) -- this
        class only tracks *which* mode is active, not who's allowed to set
        it, the same separation ConversationStore keeps from what gets said
        into it."""
        self.touch()
        self._set_mode(AccessMode.HELPER)

    def drop_to_assistant_mode(self) -> None:
        """An explicit, immediate drop -- e.g. a "asistan moduna dön"
        command -- rather than waiting for the idle timeout."""
        self._set_mode(AccessMode.ASSISTANT)

    def _idle_seconds(self) -> float:
        return time.monotonic() - self._last_activity

    def _maybe_expire(self) -> None:
        if self._mode is AccessMode.HELPER and self._idle_seconds() >= self._idle_timeout:
            logger.info("Yardımcı modu boşta kalma nedeniyle asistan moduna düştü")
            self._set_mode(AccessMode.ASSISTANT)

    def _set_mode(self, mode: AccessMode) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        if self._on_change is not None:
            self._on_change(mode)
