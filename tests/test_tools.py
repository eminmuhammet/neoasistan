import asyncio

from neo.tools.base import ToolRegistry
from neo.tools.time_tools import GetDateTool, GetTimeTool


def test_get_time_tool_returns_success():
    result = asyncio.run(GetTimeTool().run())
    assert result.success
    assert "time" in result.data


def test_get_date_tool_returns_turkish_fields():
    result = asyncio.run(GetDateTool().run())
    assert result.success
    assert result.data["weekday"]
    assert result.data["month"]


def test_registry_executes_registered_tool():
    registry = ToolRegistry()
    registry.register(GetTimeTool())
    result = asyncio.run(registry.execute("get_time", {}))
    assert result.success


def test_registry_unknown_tool_returns_error():
    registry = ToolRegistry()
    result = asyncio.run(registry.execute("does_not_exist", {}))
    assert not result.success
    assert result.error


def test_anthropic_schema_shape():
    registry = ToolRegistry()
    registry.register(GetTimeTool())
    schemas = registry.anthropic_tools()
    assert schemas[0]["name"] == "get_time"
    assert "input_schema" in schemas[0]
