from __future__ import annotations

import logging
import os
import webbrowser
from urllib.parse import quote_plus

from .app_registry import resolve_application
from .base import RiskLevel, Tool, ToolResult

logger = logging.getLogger(__name__)

_SITE_ALIASES = {
    "google": "https://www.google.com",
    "youtube": "https://www.youtube.com",
    "github": "https://github.com",
    "gmail": "https://mail.google.com",
    "twitter": "https://twitter.com",
    "x": "https://twitter.com",
    "instagram": "https://instagram.com",
    "reddit": "https://reddit.com",
    "spotify": "https://open.spotify.com",
    "netflix": "https://netflix.com",
    "wikipedia": "https://wikipedia.org",
}

_SEARCH_URLS = {
    "google": "https://www.google.com/search?q={query}",
    "youtube": "https://www.youtube.com/results?search_query={query}",
}


class OpenApplicationTool(Tool):
    name = "open_application"
    description = (
        "Bilgisayarda kurulu bir masaüstü uygulamasını adına göre bulup açar "
        "(ör. 'chrome', 'spotify', 'discord', 'visual studio code')."
    )
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Açılacak uygulamanın adı (ör. 'chrome', 'spotify').",
            }
        },
        "required": ["name"],
    }

    async def run(self, name: str, **kwargs: object) -> ToolResult:
        target = resolve_application(name)
        if target is None:
            return ToolResult(success=False, error=f"'{name}' adında bir uygulama bulamadım.")
        try:
            os.startfile(str(target))
        except OSError:
            logger.exception("Uygulama başlatılamadı: %s", name)
            return ToolResult(success=False, error=f"'{name}' başlatılırken bir hata oluştu.")
        return ToolResult(success=True, data={"launched": name})


class OpenWebsiteTool(Tool):
    name = "open_website"
    description = (
        "Varsayılan tarayıcıda bir web sitesini açar. Site adı verilebilir "
        "(ör. 'google', 'youtube', 'github') ya da tam bir URL. İsteğe bağlı "
        "'query' verilirse o sitede arama yapar."
    )
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "site": {
                "type": "string",
                "description": "Site adı (ör. 'google', 'youtube') veya tam URL.",
            },
            "query": {
                "type": "string",
                "description": "Verilirse, sitede bu terimle arama yapılır.",
            },
        },
        "required": ["site"],
    }

    async def run(self, site: str, query: str | None = None, **kwargs: object) -> ToolResult:
        key = site.strip().lower()

        if query and key in _SEARCH_URLS:
            url = _SEARCH_URLS[key].format(query=quote_plus(query))
        elif key in _SITE_ALIASES:
            url = _SITE_ALIASES[key]
        elif site.startswith(("http://", "https://")):
            url = site
        else:
            url = f"https://{site}"

        try:
            opened = webbrowser.open(url)
        except Exception:
            logger.exception("Web sitesi açılamadı: %s", url)
            opened = False

        if not opened:
            return ToolResult(success=False, error=f"'{site}' açılamadı.")
        return ToolResult(success=True, data={"url": url})
