import asyncio
from unittest.mock import patch

from neo.tools.applications import OpenApplicationTool, OpenWebsiteTool


def test_open_application_unknown_returns_error():
    tool = OpenApplicationTool()
    result = asyncio.run(tool.run(name="kesinlikle-var-olmayan-uygulama-999"))
    assert not result.success


def test_open_application_launches_resolved_target():
    tool = OpenApplicationTool()
    with patch("neo.tools.applications.resolve_application", return_value="C:/fake/app.exe"):
        with patch("neo.tools.applications.os.startfile") as mock_start:
            result = asyncio.run(tool.run(name="fakeapp"))
    assert result.success
    mock_start.assert_called_once_with("C:/fake/app.exe")


def test_open_website_known_alias():
    tool = OpenWebsiteTool()
    with patch("neo.tools.applications.webbrowser.open", return_value=True) as mock_open:
        result = asyncio.run(tool.run(site="google"))
    assert result.success
    assert result.data["url"] == "https://www.google.com"
    mock_open.assert_called_once()


def test_open_website_search_query():
    tool = OpenWebsiteTool()
    with patch("neo.tools.applications.webbrowser.open", return_value=True):
        result = asyncio.run(tool.run(site="google", query="yapay zeka"))
    assert result.success
    assert "yapay+zeka" in result.data["url"]


def test_open_website_raw_url_passthrough():
    tool = OpenWebsiteTool()
    with patch("neo.tools.applications.webbrowser.open", return_value=True):
        result = asyncio.run(tool.run(site="https://example.com"))
    assert result.data["url"] == "https://example.com"


def test_open_website_reports_failure():
    tool = OpenWebsiteTool()
    with patch("neo.tools.applications.webbrowser.open", return_value=False):
        result = asyncio.run(tool.run(site="google"))
    assert not result.success
