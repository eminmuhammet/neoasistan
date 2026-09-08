"""Faz 1: replaces Anthropic's server-side (billed-per-search) web_search
tool with a free one that scrapes DuckDuckGo's HTML-only endpoint -- see
neo/tools/web_search.py and agent.py's removal of WEB_SEARCH_TOOL.
"""

import asyncio
from unittest.mock import MagicMock, patch

from neo.tools.base import RiskLevel
from neo.tools.web_search import WebSearchTool

_SAMPLE_HTML = """
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&amp;rut=1">
    Örnek Başlık Bir
  </a>
  <a class="result__snippet">Bu <b>ilk</b> sonucun özeti.</a>
</div>
<div class="result">
  <a class="result__a" href="https://example.com/b">Örnek Başlık İki</a>
  <a class="result__snippet">İkinci sonucun özeti burada.</a>
</div>
"""


def _fake_response(text: str, status_ok: bool = True):
    response = MagicMock()
    response.text = text
    response.raise_for_status.return_value = None
    return response


def test_search_returns_parsed_results_and_unwraps_redirect_urls():
    tool = WebSearchTool()
    with patch("httpx.post", return_value=_fake_response(_SAMPLE_HTML)):
        result = asyncio.run(tool.run(query="örnek arama"))

    assert result.success
    assert result.data["count"] == 2
    first = result.data["results"][0]
    assert first["title"] == "Örnek Başlık Bir"
    assert first["url"] == "https://example.com/a"
    assert "ilk" in first["snippet"]
    assert result.data["results"][1]["url"] == "https://example.com/b"


def test_search_respects_max_results():
    tool = WebSearchTool()
    with patch("httpx.post", return_value=_fake_response(_SAMPLE_HTML)):
        result = asyncio.run(tool.run(query="örnek arama", max_results=1))

    assert result.data["count"] == 1


def test_search_rejects_empty_query():
    tool = WebSearchTool()
    result = asyncio.run(tool.run(query="   "))

    assert not result.success


def test_search_with_no_matches_returns_empty_results():
    tool = WebSearchTool()
    with patch("httpx.post", return_value=_fake_response("<html>hiçbir şey yok</html>")):
        result = asyncio.run(tool.run(query="hiçbir sonuç vermeyecek şey"))

    assert result.success
    assert result.data["count"] == 0


def test_search_network_error_returns_friendly_message():
    import httpx

    tool = WebSearchTool()
    with patch("httpx.post", side_effect=httpx.ConnectError("boom")):
        result = asyncio.run(tool.run(query="örnek arama"))

    assert not result.success
    assert result.error


def test_web_search_is_low_risk():
    assert WebSearchTool.risk == RiskLevel.LOW
