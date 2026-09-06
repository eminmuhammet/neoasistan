from __future__ import annotations

import logging

from anthropic import Anthropic, APIError, AuthenticationError

from ..config.settings import Settings

logger = logging.getLogger(__name__)


class LLMRequestError(Exception):
    """Raised when the LLM API call fails."""


# A one-hour cache rather than the five-minute default. NEO is used in short
# bursts spread across a day, and with a 5m window most first-messages would
# land after expiry -- paying the *write* premium (1.25x) instead of the read
# discount, i.e. worse than no caching at all for sporadic use. A 1h write
# costs 2x but is amortized over everything said in that hour.
_CACHE_CONTROL = {"type": "ephemeral", "ttl": "1h"}


def _cacheable_system(system: str) -> list[dict]:
    """The system prompt as a cached block.

    It is byte-identical on every call within a day, but was being charged
    at full input price each time -- 1,289 tokens a call, and a message that
    uses a tool pays it two or three times over.
    """
    return [{"type": "text", "text": system, "cache_control": _CACHE_CONTROL}]


def _cacheable_tools(tools: list[dict]) -> list[dict]:
    """Marks the end of the local tool definitions as a cache breakpoint, so
    everything up to that point is read from cache on later calls.

    Measured: 18 tool schemas are 2,712 tokens resent verbatim on every
    single request -- together with the system prompt that was $1.20 per 100
    messages before the user had said a word. Cache reads cost a tenth of
    that.

    Server-side tools (web_search) are left after the breakpoint: they're
    small, and keeping them outside means a change to their config can't
    invalidate the expensive part of the prefix.
    """
    if not tools:
        return []
    local = [tool for tool in tools if "type" not in tool]
    server = [tool for tool in tools if "type" in tool]
    if not local:
        return tools
    local = [*local[:-1], {**local[-1], "cache_control": _CACHE_CONTROL}]
    return [*local, *server]


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = Anthropic(api_key=settings.require_api_key())

    def send(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
    ):
        try:
            response = self._client.messages.create(
                model=self._settings.model,
                max_tokens=max_tokens,
                system=_cacheable_system(system),
                messages=messages,
                tools=_cacheable_tools(tools or []),
            )
            self._log_usage(response)
            return response
        except AuthenticationError as exc:
            logger.error("Anthropic API anahtarı geçersiz")
            raise LLMRequestError(
                "Anthropic API anahtarınız geçersiz görünüyor. .env dosyasındaki "
                "ANTHROPIC_API_KEY değerini kontrol edin."
            ) from exc
        except APIError as exc:
            message = str(getattr(exc, "message", "") or str(exc)).lower()
            if "credit balance" in message:
                logger.error("Anthropic hesap bakiyesi yetersiz")
                raise LLMRequestError(
                    "Anthropic hesabınızda kredi bakiyesi yetersiz görünüyor. "
                    "console.anthropic.com üzerinden Plans & Billing bölümünden "
                    "kredi ekleyin."
                ) from exc
            logger.error("Anthropic API error: %s", exc.__class__.__name__)
            raise LLMRequestError(
                "Claude API'sine ulaşılamadı. İnternet bağlantınızı veya API "
                "anahtarınızı kontrol edin."
            ) from exc

    @staticmethod
    def _log_usage(response) -> None:
        """Records what each call actually cost in tokens.

        Without this the only signal that something was expensive was the
        billing page the next day -- which is how a $1.20 day went unnoticed
        until the user checked. `cache_read` should dominate once the prompt
        cache is warm; if it stays at zero, caching has silently stopped
        working.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        logger.info(
            "Claude kullanımı: giriş=%s, çıkış=%s, önbellek_yazma=%s, önbellek_okuma=%s",
            getattr(usage, "input_tokens", 0),
            getattr(usage, "output_tokens", 0),
            getattr(usage, "cache_creation_input_tokens", 0),
            getattr(usage, "cache_read_input_tokens", 0),
        )
