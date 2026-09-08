"""LocalLLMClient talks to a local Ollama server (see core/local_llm_client.py).
No tools are ever sent to it -- it's only used as a plain-chat fallback when
Claude is unreachable, and a confidently wrong tool-shaped answer from a
model that can't actually run tools would be worse than the outage itself.
"""

import asyncio

import httpx
import pytest

from neo.core.local_llm_client import ESCALATE_MARKER, LocalLLMClient, LocalLLMError


class _FakeResponse:
    def __init__(self, json_data: dict, status_code: int = 200) -> None:
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self) -> dict:
        return self._json


class _FakeAsyncClient:
    def __init__(self, response: _FakeResponse | None = None, error: Exception | None = None, **_):
        self._response = response
        self._error = error
        self.posted_payload = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, json):
        self.posted_payload = json
        if self._error is not None:
            raise self._error
        return self._response


def test_chat_returns_the_model_reply(monkeypatch):
    fake = _FakeAsyncClient(_FakeResponse({"message": {"content": "Merhaba!"}}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: fake)

    client = LocalLLMClient("http://localhost:11434", "qwen2.5:3b")
    reply = asyncio.run(client.chat([{"role": "user", "content": "selam"}], system="Sen NEO'sun."))

    assert reply == "Merhaba!"
    assert fake.posted_payload["model"] == "qwen2.5:3b"
    assert fake.posted_payload["messages"][0] == {"role": "system", "content": "Sen NEO'sun."}


def test_chat_raises_local_llm_error_when_server_unreachable(monkeypatch):
    fake = _FakeAsyncClient(error=httpx.ConnectError("connection refused"))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: fake)

    client = LocalLLMClient("http://localhost:11434", "qwen2.5:3b")

    with pytest.raises(LocalLLMError):
        asyncio.run(client.chat([{"role": "user", "content": "selam"}], system="Sen NEO'sun."))


def test_chat_raises_local_llm_error_on_empty_reply(monkeypatch):
    fake = _FakeAsyncClient(_FakeResponse({"message": {"content": "  "}}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: fake)

    client = LocalLLMClient("http://localhost:11434", "qwen2.5:3b")

    with pytest.raises(LocalLLMError):
        asyncio.run(client.chat([{"role": "user", "content": "selam"}], system="Sen NEO'sun."))


def test_chat_raises_local_llm_error_with_no_messages(monkeypatch):
    client = LocalLLMClient("http://localhost:11434", "qwen2.5:3b")

    with pytest.raises(LocalLLMError):
        asyncio.run(client.chat([], system="Sen NEO'sun."))


def test_chat_or_escalate_returns_plain_reply_when_the_model_answers_directly(monkeypatch):
    fake = _FakeAsyncClient(_FakeResponse({"message": {"content": "Gayet iyiyim!"}}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: fake)
    client = LocalLLMClient("http://localhost:11434", "qwen2.5:3b")

    needs_claude, text = asyncio.run(
        client.chat_or_escalate([{"role": "user", "content": "naber"}], system="Sen NEO'sun.")
    )

    assert needs_claude is False
    assert text == "Gayet iyiyim!"
    assert ESCALATE_MARKER in fake.posted_payload["messages"][0]["content"]


def test_chat_or_escalate_flags_requests_the_local_model_cannot_fulfill(monkeypatch):
    fake = _FakeAsyncClient(_FakeResponse({"message": {"content": ESCALATE_MARKER}}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: fake)
    client = LocalLLMClient("http://localhost:11434", "qwen2.5:3b")

    needs_claude, text = asyncio.run(
        client.chat_or_escalate(
            [{"role": "user", "content": "şu dosyayı açar mısın"}], system="Sen NEO'sun."
        )
    )

    assert needs_claude is True
    assert text == ""


def test_chat_or_escalate_treats_an_unreachable_server_as_an_escalation(monkeypatch):
    fake = _FakeAsyncClient(error=httpx.ConnectError("connection refused"))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: fake)
    client = LocalLLMClient("http://localhost:11434", "qwen2.5:3b")

    needs_claude, text = asyncio.run(
        client.chat_or_escalate([{"role": "user", "content": "naber"}], system="Sen NEO'sun.")
    )

    assert needs_claude is True
    assert text == ""
