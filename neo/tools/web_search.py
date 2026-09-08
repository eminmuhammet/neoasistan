from __future__ import annotations

import asyncio
import base64
import html
import logging
import re
from urllib.parse import parse_qs, urlparse

from .base import RiskLevel, Tool, ToolResult

logger = logging.getLogger(__name__)

# Originally scraped DuckDuckGo's HTML-only endpoint (no JS, no API key --
# built for exactly this). Switched to Bing after a live test from this
# user's connection timed out on every DuckDuckGo host (html.duckduckgo.com,
# lite.duckduckgo.com, even the plain duckduckgo.com homepage) while Google
# and Bing both answered normally -- DuckDuckGo has a known history of ISP
# blocks in Turkey. Bing's plain (JS-off) results page still needs no API
# key and no login, so the free/local goal (see agent.py's WEB_SEARCH_TOOL
# removal) is unaffected.
_SEARCH_URL = "https://www.bing.com/search"

# Bing wraps every organic result's href in its own click-tracking redirect
# (bing.com/ck/a?...&u=a1<base64url of the real target>&...) rather than
# linking to it directly -- decoded here so the tool result (and fetch_page,
# if Claude follows up on it) gets a URL that's actually fetchable.
_RESULT_RE = re.compile(
    r'<h2[^>]*>\s*<a[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>\s*</h2>'
    r'\s*<div class="b_caption"><p[^>]*>(?P<snippet>.*?)</p>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(html_fragment: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", _TAG_RE.sub("", html_fragment))).strip()


def _decode_bing_redirect(url: str) -> str:
    url = html.unescape(url)
    parsed = urlparse(url)
    if not parsed.netloc.endswith("bing.com") or parsed.path != "/ck/a":
        return url
    encoded = parse_qs(parsed.query).get("u", [None])[0]
    if not encoded or not encoded.startswith("a1"):
        return url
    body = encoded[2:]
    body += "=" * (-len(body) % 4)
    try:
        return base64.urlsafe_b64decode(body).decode("utf-8", errors="replace")
    except Exception:
        return url


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
            response = httpx.get(
                _SEARCH_URL,
                params={"q": query},
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
                    )
                },
                timeout=10.0,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            logger.exception("Web araması başarısız")
            return ToolResult(success=False, error="Web araması şu an yapılamıyor.")

        results = []
        for match in _RESULT_RE.finditer(response.text):
            results.append(
                {
                    "title": _clean_text(match.group("title")),
                    "url": _decode_bing_redirect(match.group("url")),
                    "snippet": _clean_text(match.group("snippet")),
                }
            )
            if len(results) >= max_results:
                break

        return ToolResult(
            success=True, data={"query": query, "results": results, "count": len(results)}
        )
