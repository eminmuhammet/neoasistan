"""Faz 1: fetch_page lets Claude (or the local model, once escalated) read
a page web_search found, without a real HTML-parsing dependency -- see the
comment in neo/tools/fetch_page.py on why a regex strip is enough here.
"""

import asyncio
from unittest.mock import MagicMock, patch

from neo.tools.base import RiskLevel
from neo.tools.fetch_page import FetchPageTool

_SAMPLE_HTML = """
<html><head><title>Test Sayfası</title>
<style>body { color: red; }</style>
</head>
<body>
<script>console.log("ignored");</script>
<p>Burada  gerçek   içerik var.</p>
</body></html>
"""


def _fake_response(text: str, content_type: str = "text/html"):
    response = MagicMock()
    response.text = text
    response.headers = {"content-type": content_type}
    response.raise_for_status.return_value = None
    return response


def test_fetch_page_extracts_title_and_strips_tags():
    tool = FetchPageTool()
    with patch("httpx.get", return_value=_fake_response(_SAMPLE_HTML)):
        result = asyncio.run(tool.run(url="https://example.com/page"))

    assert result.success
    assert result.data["title"] == "Test Sayfası"
    assert "Burada gerçek içerik var." in result.data["content"]
    assert "console.log" not in result.data["content"]
    assert "color: red" not in result.data["content"]


def test_fetch_page_rejects_non_http_urls():
    tool = FetchPageTool()
    result = asyncio.run(tool.run(url="file:///etc/passwd"))

    assert not result.success


def test_fetch_page_rejects_non_text_content_types():
    tool = FetchPageTool()
    with patch("httpx.get", return_value=_fake_response("binary junk", content_type="image/png")):
        result = asyncio.run(tool.run(url="https://example.com/image.png"))

    assert not result.success


def test_fetch_page_network_error_returns_friendly_message():
    import httpx

    tool = FetchPageTool()
    with patch("httpx.get", side_effect=httpx.ConnectError("boom")):
        result = asyncio.run(tool.run(url="https://example.com"))

    assert not result.success
    assert result.error


def test_fetch_page_truncates_long_content():
    tool = FetchPageTool()
    long_html = "<html><body>" + ("kelime " * 5000) + "</body></html>"
    with patch("httpx.get", return_value=_fake_response(long_html)):
        result = asyncio.run(tool.run(url="https://example.com/long"))

    assert result.success
    assert result.data["truncated"] is True
    assert len(result.data["content"]) <= 6000


def test_fetch_page_is_low_risk():
    assert FetchPageTool.risk == RiskLevel.LOW
