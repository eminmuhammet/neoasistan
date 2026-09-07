"""PermissionManager's mode-aware behavior, once an AccessModeManager is
wired in.

    | risk   | assistant mode      | helper mode         |
    |--------|----------------------|----------------------|
    | LOW    | free                 | free                 |
    | MEDIUM | confirm              | free                 |
    | HIGH   | refused, no dialog   | confirm              |
"""

import asyncio

from neo.core.access_mode import AccessModeManager
from neo.core.permissions import PermissionManager
from neo.tools.base import RiskLevel


def _manager(mode_manager, confirm=None):
    return PermissionManager(confirm=confirm, mode_manager=mode_manager)


async def _always_yes(name, desc):
    return True


def test_low_risk_is_free_in_both_modes():
    modes = AccessModeManager()
    manager = _manager(modes)  # no confirm callback at all

    assert asyncio.run(manager.check("get_time", RiskLevel.LOW, "{}")) is True
    modes.unlock_helper_mode()
    assert asyncio.run(manager.check("get_time", RiskLevel.LOW, "{}")) is True


def test_medium_risk_in_assistant_mode_requires_confirmation():
    modes = AccessModeManager()
    manager = _manager(modes)  # no confirm wired -> deny

    assert asyncio.run(manager.check("open_application", RiskLevel.MEDIUM, "{}")) is False

    manager2 = _manager(modes, confirm=_always_yes)
    assert asyncio.run(manager2.check("open_application", RiskLevel.MEDIUM, "{}")) is True


def test_medium_risk_in_helper_mode_runs_freely():
    """The table's "Serbest" for MEDIUM+helper: no confirm callback needed
    at all, unlike assistant mode."""
    modes = AccessModeManager()
    modes.unlock_helper_mode()
    manager = _manager(modes)  # deliberately no confirm callback

    assert asyncio.run(manager.check("open_application", RiskLevel.MEDIUM, "{}")) is True


def test_high_risk_in_assistant_mode_is_refused_without_asking():
    """The plan's specific safety point: not even offered a confirmation
    dialog. Verified by wiring a confirm callback that would prove itself
    if called -- it must never be reached."""
    modes = AccessModeManager()
    called = []

    async def would_approve(name, desc):
        called.append((name, desc))
        return True

    manager = _manager(modes, confirm=would_approve)

    allowed = asyncio.run(manager.check("shutdown_computer", RiskLevel.HIGH, "{}"))

    assert allowed is False
    assert called == [], "asistan modunda HIGH risk icin onay penceresi hic acilmamali"


def test_high_risk_in_helper_mode_still_requires_confirmation():
    """"Tam yetki" means the action is *possible*, not that it runs
    unattended -- helper mode still asks before a HIGH-risk action."""
    modes = AccessModeManager()
    modes.unlock_helper_mode()

    denied = _manager(modes)  # no confirm wired -> still deny
    assert asyncio.run(denied.check("shutdown_computer", RiskLevel.HIGH, "{}")) is False

    approved = _manager(modes, confirm=_always_yes)
    assert asyncio.run(approved.check("shutdown_computer", RiskLevel.HIGH, "{}")) is True


def test_without_a_mode_manager_behavior_is_unchanged():
    """Backward compatibility is the point of this test: every existing
    caller that builds PermissionManager() with no mode manager (including
    every test written before mode support existed) must see the exact
    original rule -- confirm-or-deny for both MEDIUM and HIGH alike."""
    no_confirm = PermissionManager()
    assert asyncio.run(no_confirm.check("shutdown_computer", RiskLevel.HIGH, "{}")) is False
    assert asyncio.run(no_confirm.check("open_application", RiskLevel.MEDIUM, "{}")) is False

    with_confirm = PermissionManager(confirm=_always_yes)
    assert asyncio.run(with_confirm.check("shutdown_computer", RiskLevel.HIGH, "{}")) is True
    assert asyncio.run(with_confirm.check("open_application", RiskLevel.MEDIUM, "{}")) is True


def test_mode_manager_can_be_wired_after_construction():
    manager = PermissionManager()
    modes = AccessModeManager()

    manager.set_mode_manager(modes)

    called = []
    manager.set_confirm(lambda name, desc: called.append(1) or _always_yes(name, desc))
    allowed = asyncio.run(manager.check("shutdown_computer", RiskLevel.HIGH, "{}"))
    assert allowed is False  # assistant mode: still refused outright
