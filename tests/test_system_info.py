import asyncio

from neo.tools.system_info import (
    GetCpuUsageTool,
    GetDiskUsageTool,
    GetGpuStatusTool,
    GetNetworkStatusTool,
    GetRamUsageTool,
    GetSystemInfoTool,
)


def test_cpu_usage_returns_percent():
    result = asyncio.run(GetCpuUsageTool().run())
    assert result.success
    assert 0 <= result.data["cpu_percent"] <= 100


def test_ram_usage_returns_expected_fields():
    result = asyncio.run(GetRamUsageTool().run())
    assert result.success
    assert result.data["total_gb"] > 0
    assert 0 <= result.data["percent"] <= 100


def test_disk_usage_returns_at_least_one_drive():
    result = asyncio.run(GetDiskUsageTool().run())
    assert result.success
    assert len(result.data["drives"]) >= 1


def test_system_info_returns_core_fields():
    result = asyncio.run(GetSystemInfoTool().run())
    assert result.success
    assert result.data["cpu_cores_logical"] >= 1
    assert result.data["ram_total_gb"] > 0


def test_gpu_status_returns_structured_result():
    # No NVIDIA GPU on the machine running this test is a valid outcome too --
    # it must be reported honestly (success=False + error), never faked.
    result = asyncio.run(GetGpuStatusTool().run())
    if result.success:
        assert "gpu_percent" in result.data
    else:
        assert result.error


def test_network_status_returns_boolean():
    result = asyncio.run(GetNetworkStatusTool().run())
    assert result.success
    assert isinstance(result.data["connected"], bool)
