from __future__ import annotations

import asyncio
import logging
import re

from .base import RiskLevel, Tool, ToolResult

logger = logging.getLogger(__name__)

_MAX_CONTENT_CHARS = 6000

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)


def _extract_text(html: str) -> tuple[str, str]:
    """A regex strip rather than a real HTML parser: this only needs to
    turn a page into readable text for Claude to summarize, not preserve
    structure -- adding a parsing dependency (bs4/lxml) for that would be
    the same overkill document_index.py already deliberately avoided with
    embeddings (see that file's own comment)."""
    title_match = _TITLE_RE.search(html)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""

    body = _SCRIPT_STYLE_RE.sub(" ", html)
    text = re.sub(r"\s+", " ", _TAG_RE.sub(" ", body)).strip()
    return title, text


class FetchPageTool(Tool):
    name = "fetch_page"
    description = (
        "Bir web sayfasının metin içeriğini indirip döndürür (web_search'ün "
        "bulduğu bir sonucu daha detaylı okumak için kullanılır)."
    )
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "İçeriği okunacak sayfanın tam adresi."},
        },
        "required": ["url"],
    }

    async def run(self, url: str, **kwargs: object) -> ToolResult:
        url = (url or "").strip()
        if not url.startswith(("http://", "https://")):
            return ToolResult(success=False, error="Geçerli bir http(s) adresi vermelisin.")
        return await asyncio.to_thread(self._fetch, url)

    def _fetch(self, url: str) -> ToolResult:
        import httpx

        try:
            response = httpx.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; NEO-assistant/1.0)"},
                timeout=10.0,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("Sayfa reddedildi (%s): %s", exc.response.status_code, url)
            # Distinct from a network failure on purpose: a live check found
            # Wikipedia and openai.com both returning 403 to this tool no
            # matter the User-Agent (Wikipedia's own body text points at
            # its bot policy; openai.com looks Cloudflare-gated) -- this is
            # the site refusing automated access, not a blip worth retrying
            # the same URL for. Saying so explicitly is what stops a research
            # turn from burning its whole tool-call budget hammering one
            # blocked domain (see agent.py's RESEARCH_MODE_PROMPT).
            if exc.response.status_code in (403, 429):
                return ToolResult(
                    success=False,
                    error=(
                        f"'{url}' otomatik erişimi engelliyor (HTTP {exc.response.status_code}). "
                        "Bu adresi tekrar deneme, başka bir kaynağa geç."
                    ),
                )
            return ToolResult(
                success=False, error=f"'{url}' adresine ulaşılamadı (HTTP {exc.response.status_code})."
            )
        except httpx.HTTPError:
            logger.exception("Sayfa indirilemedi: %s", url)
            return ToolResult(success=False, error=f"'{url}' adresine ulaşılamadı.")

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            return ToolResult(
                success=False,
                error="Bu sayfa metin/HTML değil, içeriği okunamıyor.",
            )

        title, text = _extract_text(response.text)
        truncated = text[:_MAX_CONTENT_CHARS]
        return ToolResult(
            success=True,
            data={
                "url": url,
                "title": title,
                "content": truncated,
                "truncated": len(text) > _MAX_CONTENT_CHARS,
            },
        )
