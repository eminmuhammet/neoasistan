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


class PlayOnSpotifyTool(Tool):
    name = "play_on_spotify"
    description = (
        "Spotify'da bir şarkı/sanatçı/albüm arar ve arama sonucunu açar. "
        "Kullanıcının Spotify masaüstü uygulaması kurulu olsun olmasın "
        "çalışır (web player kullanır) -- 'X şarkısını Spotify'da aç' gibi "
        "isteklerde computer_control (fare/klavye) yerine bunu kullan; "
        "otomasyon araçları bu iş için hem yavaş hem gereksiz risklidir."
    )
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Aranacak şarkı/sanatçı/albüm adı (ör. 'Tarkan Kuzu Kuzu').",
            }
        },
        "required": ["query"],
    }

    async def run(self, query: str, **kwargs: object) -> ToolResult:
        query = (query or "").strip()
        if not query:
            return ToolResult(success=False, error="Aranacak şarkı/sanatçı adı boş olamaz.")

        url = f"https://open.spotify.com/search/{quote_plus(query)}"
        try:
            opened = webbrowser.open(url)
        except Exception:
            logger.exception("Spotify araması açılamadı: %s", query)
            opened = False

        if not opened:
            return ToolResult(success=False, error="Spotify açılamadı.")
        return ToolResult(success=True, data={"query": query, "url": url})


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
