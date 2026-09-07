import asyncio

from neo.memory.audit_store import AuditStore
from neo.tools.audit import GetRecentActivityTool
from neo.tools.base import RiskLevel


def _store(tmp_path) -> AuditStore:
    return AuditStore(tmp_path / "audit.db")


def test_returns_logged_entries(tmp_path):
    store = _store(tmp_path)
    store.log("click", "high", {"x": 1}, True, None)

    result = asyncio.run(GetRecentActivityTool(store).run())

    assert result.success is True
    assert result.data["count"] == 1
    assert result.data["entries"][0]["tool_name"] == "click"


def test_returns_empty_list_when_nothing_logged(tmp_path):
    result = asyncio.run(GetRecentActivityTool(_store(tmp_path)).run())

    assert result.success is True
    assert result.data["entries"] == []


def test_is_low_risk():
    assert GetRecentActivityTool.risk == RiskLevel.LOW
