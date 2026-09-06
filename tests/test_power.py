import asyncio
from unittest.mock import patch

from neo.tools.base import RiskLevel
from neo.tools.power import LockComputerTool, RestartComputerTool, ShutdownComputerTool


def test_lock_computer_calls_windows_api():
    tool = LockComputerTool()
    assert tool.risk == RiskLevel.MEDIUM

    with patch("ctypes.windll.user32.LockWorkStation") as mock_lock:
        result = asyncio.run(tool.run())

    mock_lock.assert_called_once()
    assert result.success


def test_lock_computer_reports_failure_gracefully():
    tool = LockComputerTool()
    with patch("ctypes.windll.user32.LockWorkStation", side_effect=OSError("boom")):
        result = asyncio.run(tool.run())
    assert not result.success


def test_shutdown_is_high_risk_and_never_runs_a_real_command():
    tool = ShutdownComputerTool()
    assert tool.risk == RiskLevel.HIGH

    with patch("subprocess.run") as mock_run:
        result = asyncio.run(tool.run())

    mock_run.assert_called_once_with(
        ["shutdown", "/s", "/t", "5"], check=True, capture_output=True
    )
    assert result.success


def test_restart_is_high_risk_and_never_runs_a_real_command():
    tool = RestartComputerTool()
    assert tool.risk == RiskLevel.HIGH

    with patch("subprocess.run") as mock_run:
        result = asyncio.run(tool.run())

    mock_run.assert_called_once_with(
        ["shutdown", "/r", "/t", "5"], check=True, capture_output=True
    )
    assert result.success


def test_shutdown_reports_failure_gracefully():
    import subprocess

    tool = ShutdownComputerTool()
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "shutdown")):
        result = asyncio.run(tool.run())
    assert not result.success
