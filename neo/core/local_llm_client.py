from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# When routing chat to the local model first (Agent.handle_message), the
# local model has no tools of its own -- it's asked to say exactly this,
# and nothing else, when the request actually needs one, rather than
# guessing an answer to something it can't really do (open an app, read a
# file, check the real calendar). Deliberately not natural language ("I
# can't do this") so it can't be confused with an ordinary refusal Claude
# should still see verbatim.
ESCALATE_MARKER = "TOOL_GEREKLI"

ROUTER_INSTRUCTION = (
    "\n\nÖNEMLİ: Senin araç/yeteneğin yok -- sadece sohbet edebilirsin. "
    "Kullanıcının isteği gerçek bir yetenek gerektiriyorsa (dosya/ekran "
    "okuma, uygulama açma, e-posta, takvime yazma, sistem bilgisi, internet "
    "araması, çok adımlı bir görev ya da başka bir sisteme erişim), "
    f"cevap olarak SADECE şunu yaz, başka hiçbir şey ekleme: {ESCALATE_MARKER}\n"
    "Aksi halde normal, doğal bir sohbet cevabı ver."
)


class LocalLLMError(Exception):
    """Raised when the local model can't be reached or fails to answer."""


def _flatten_content(content: Any) -> str:
    """Reduces one message's content to plain text for the local model.

    Claude's context holds tool_use/tool_result blocks the local model has
    no concept of (it isn't given any tools -- see LocalLLMClient.chat).
    Messages that are only tool plumbing collapse to an empty string and are
    dropped by the caller rather than confusing a model with no tool_use
    turn to attach that content to.
    """
    if isinstance(content, str):
        return content
    parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
    return "\n".join(p for p in parts if p)


def to_plain_messages(messages: list[dict]) -> list[dict[str, str]]:
    """Converts Anthropic-format context messages into the plain
    {"role", "content"} shape Ollama's /api/chat expects, dropping any
    message that turns out to be pure tool plumbing once flattened."""
    plain = []
    for message in messages:
        text = _flatten_content(message.get("content"))
        if text:
            plain.append({"role": message["role"], "content": text})
    return plain


class LocalLLMClient:
    """Talks to a local Ollama server for plain-text chat replies.

    Deliberately no tool support: this is only ever used as a fallback for
    ordinary conversation when Claude is unreachable (no credit, no
    internet, API outage) -- see agent.py's _run_tool_loop. A request that
    actually needs a tool (calendar, system info, ...) gaining a
    confidently wrong local-model answer instead of an honest "can't reach
    Claude" would be worse than the outage itself.
    """

    def __init__(self, base_url: str, model: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    async def chat(self, messages: list[dict], system: str) -> str:
        plain_messages = to_plain_messages(messages)
        if not plain_messages:
            raise LocalLLMError("Yerel modele gönderilecek bir mesaj yok.")

        payload = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}, *plain_messages],
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self._base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            logger.warning("Yerel model (Ollama) çalışmadı: %s", exc.__class__.__name__)
            raise LocalLLMError(
                "Yerel model de yanıt veremedi. Ollama'nın çalıştığından emin olun."
            ) from exc

        text = (data.get("message") or {}).get("content", "").strip()
        if not text:
            raise LocalLLMError("Yerel model boş bir yanıt döndürdü.")
        return text

    async def chat_or_escalate(self, messages: list[dict], system: str) -> tuple[bool, str]:
        """Like chat(), but asks the local model to flag requests it can't
        actually fulfill (see ROUTER_INSTRUCTION) instead of answering them.

        Returns (needs_claude, text): when needs_claude is True, text is
        empty and the caller should run the real Claude tool loop instead.
        A LocalLLMError (server down, empty reply) is treated the same as
        an explicit escalation -- if the local model can't even be reached,
        Claude is the only path left, not a hard failure.
        """
        try:
            reply = await self.chat(messages, system + ROUTER_INSTRUCTION)
        except LocalLLMError:
            return True, ""
        if reply.strip() == ESCALATE_MARKER:
            return True, ""
        return False, reply
