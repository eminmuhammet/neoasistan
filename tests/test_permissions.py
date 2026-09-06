import asyncio

from neo.core.permissions import PermissionManager
from neo.tools.base import RiskLevel


def test_low_risk_auto_approved():
    manager = PermissionManager()
    allowed = asyncio.run(manager.check("get_time", RiskLevel.LOW, "{}"))
    assert allowed is True


def test_high_risk_denied_without_confirm_callback():
    manager = PermissionManager()
    allowed = asyncio.run(manager.check("shutdown_computer", RiskLevel.HIGH, "{}"))
    assert allowed is False


def test_high_risk_uses_confirm_callback():
    async def always_yes(name, desc):
        return True

    manager = PermissionManager(confirm=always_yes)
    allowed = asyncio.run(manager.check("shutdown_computer", RiskLevel.HIGH, "{}"))
    assert allowed is True


def test_set_confirm_wires_callback_after_construction():
    async def always_no(name, desc):
        return False

    manager = PermissionManager()
    assert asyncio.run(manager.check("shutdown_computer", RiskLevel.HIGH, "{}")) is False

    manager.set_confirm(always_no)
    assert asyncio.run(manager.check("shutdown_computer", RiskLevel.HIGH, "{}")) is False

    manager.set_confirm(None)
    assert asyncio.run(manager.check("shutdown_computer", RiskLevel.HIGH, "{}")) is False


def test_set_confirm_can_later_approve():
    manager = PermissionManager()

    async def always_yes(name, desc):
        return True

    manager.set_confirm(always_yes)
    assert asyncio.run(manager.check("lock_computer", RiskLevel.MEDIUM, "{}")) is True
