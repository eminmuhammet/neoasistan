from __future__ import annotations

import asyncio
import logging
import re

from .base import RiskLevel, Tool, ToolResult

logger = logging.getLogger(__name__)

# DuckDuckGo's HTML-only endpoint (no JS, no API key) -- built for exactly
# this: a plain HTML results page meant for text browsers/lite clients,
# unlike the JS-heavy main site which has nothing stable to scrape. This
# replaces Anthropic's server-side web_search tool (which bills per search
# on top of tokens) with a free one, see agent.py's WEB_SEARCH_TOOL removal.
_SEARCH_URL = "https://html.duckduckgo.com/html/"

_RESULT_BLOCK_RE = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'.*?<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(html_fragment: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub("", html_fragment)).strip()


def _unwrap_redirect(url: str) -> str:
    """DuckDuckGo's HTML results link through its own redirector
    (//duckduckgo.com/l/?uddg=<encoded target>&...) rather than the real
    target -- unwrapped here so the tool result gives Claude (and
    fetch_page) a URL that's actually fetchable."""
    if "uddg=" not in url:
        return url
    from urllib.parse import parse_qs, unquote, urlparse

    query = parse_qs(urlparse(url).query)
    target = query.get("uddg")
    return unquote(target[0]) if target else url


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "İnternette arama yapar ve en alakalı sonuçları (başlık, URL, kısa özet) "
        "döndürür. Bir sonucun tam içeriğini okumak için fetch_page kullan."
    )
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Arama sorgusu."},
            "max_results": {
                "type": "integer",
                "description": "Döndürülecek en fazla sonuç sayısı (varsayılan 5).",
            },
        },
        "required": ["query"],
    }

    async def run(self, query: str, max_results: int = 5, **kwargs: object) -> ToolResult:
        query = (query or "").strip()
        if not query:
            return ToolResult(success=False, error="Arama sorgusu boş olamaz.")
        return await asyncio.to_thread(self._search, query, max(1, min(max_results, 10)))

    def _search(self, query: str, max_results: int) -> ToolResult:
        import httpx

        try:
            response = httpx.post(
                _SEARCH_URL,
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (compatible; NEO-assistant/1.0)"},
                timeout=10.0,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            logger.exception("Web araması başarısız")
            return ToolResult(success=False, error="Web araması şu an yapılamıyor.")

        results = []
        for match in _RESULT_BLOCK_RE.finditer(response.text):
            results.append(
                {
                    "title": _clean_text(match.group("title")),
                    "url": _unwrap_redirect(match.group("url")),
                    "snippet": _clean_text(match.group("snippet")),
                }
            )
            if len(results) >= max_results:
                break

        if not results:
            return ToolResult(success=True, data={"query": query, "results": [], "count": 0})
        return ToolResult(
            success=True, data={"query": query, "results": results, "count": len(results)}
        )
