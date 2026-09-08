"""Faz 1: replaces Anthropic's server-side (billed-per-search) web_search
tool with a free one. Originally scraped DuckDuckGo's HTML endpoint, but a
live check from this user's connection found every DuckDuckGo host timing
out (a known ISP-level block in Turkey) while Bing answered normally -- see
neo/tools/web_search.py's comment. Bing wraps every result href in its own
click-tracking redirect (bing.com/ck/a?...&u=a1<base64url target>&...),
which is why decoding that is its own tested function.
"""

import asyncio
from unittest.mock import MagicMock, patch

from neo.tools.base import RiskLevel
from neo.tools.web_search import WebSearchTool, _decode_bing_redirect

_SAMPLE_HTML = """
<li class="b_algo">
<h2 class=""><a target="_blank" href="https://www.bing.com/ck/a?!&amp;&amp;p=xyz&amp;u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9h&amp;ntb=1" h="ID=SERP,1">Örnek <strong>Başlık</strong> Bir</a></h2>
<div class="b_caption"><p class="b_lineclamp2">Bu <b>ilk</b> sonucun &amp; özeti.</p></div>
</li>
<li class="b_algo">
<h2 class=""><a target="_blank" href="https://example.com/b" h="ID=SERP,2">Örnek Başlık İki</a></h2>
<div class="b_caption"><p class="b_lineclamp2">İkinci sonucun özeti burada.</p></div>
</li>
"""


def _fake_response(text: str):
    response = MagicMock()
    response.text = text
    response.raise_for_status.return_value = None
    return response


def test_decode_bing_redirect_extracts_the_real_target_url():
    wrapped = "https://www.bing.com/ck/a?!&&p=xyz&u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9h&ntb=1"
    assert _decode_bing_redirect(wrapped) == "https://example.com/a"


def test_decode_bing_redirect_passes_through_a_direct_url():
    assert _decode_bing_redirect("https://example.com/b") == "https://example.com/b"


def test_search_returns_parsed_results_and_decodes_redirect_urls():
    tool = WebSearchTool()
    with patch("httpx.get", return_value=_fake_response(_SAMPLE_HTML)):
        result = asyncio.run(tool.run(query="örnek arama"))

    assert result.success
    assert result.data["count"] == 2
    first = result.data["results"][0]
    assert first["title"] == "Örnek Başlık Bir"
    assert first["url"] == "https://example.com/a"
    assert "ilk" in first["snippet"]
    assert "&" in first["snippet"]  # &amp; unescaped
    assert result.data["results"][1]["url"] == "https://example.com/b"


def test_search_respects_max_results():
    tool = WebSearchTool()
    with patch("httpx.get", return_value=_fake_response(_SAMPLE_HTML)):
        result = asyncio.run(tool.run(query="örnek arama", max_results=1))

    assert result.data["count"] == 1


def test_search_rejects_empty_query():
    tool = WebSearchTool()
    result = asyncio.run(tool.run(query="   "))

    assert not result.success


def test_search_with_no_matches_returns_empty_results():
    tool = WebSearchTool()
    with patch("httpx.get", return_value=_fake_response("<html>hiçbir şey yok</html>")):
        result = asyncio.run(tool.run(query="hiçbir sonuç vermeyecek şey"))

    assert result.success
    assert result.data["count"] == 0


def test_search_network_error_returns_friendly_message():
    import httpx

    tool = WebSearchTool()
    with patch("httpx.get", side_effect=httpx.ConnectTimeout("boom")):
        result = asyncio.run(tool.run(query="örnek arama"))

    assert not result.success
    assert result.error


def test_web_search_is_low_risk():
    assert WebSearchTool.risk == RiskLevel.LOW
