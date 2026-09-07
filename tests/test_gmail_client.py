import asyncio
from pathlib import Path

import pytest

from neo.tools.gmail_client import GmailClient, GmailUnavailableError


def test_not_configured_when_credentials_file_is_missing(tmp_path):
    client = GmailClient(tmp_path / "nope.json", tmp_path / "token.json")

    assert client.is_configured() is False


def test_configured_when_credentials_file_exists(tmp_path):
    creds = tmp_path / "credentials.json"
    creds.write_text("{}", encoding="utf-8")
    client = GmailClient(creds, tmp_path / "token.json")

    assert client.is_configured() is True


def test_list_recent_raises_when_not_configured(tmp_path):
    client = GmailClient(tmp_path / "nope.json", tmp_path / "token.json")

    with pytest.raises(GmailUnavailableError):
        asyncio.run(client.list_recent())


def test_create_draft_raises_when_not_configured(tmp_path):
    client = GmailClient(tmp_path / "nope.json", tmp_path / "token.json")

    with pytest.raises(GmailUnavailableError):
        asyncio.run(client.create_draft("a@b.com", "konu", "gövde"))


def test_send_raises_when_not_configured(tmp_path):
    client = GmailClient(tmp_path / "nope.json", tmp_path / "token.json")

    with pytest.raises(GmailUnavailableError):
        asyncio.run(client.send("a@b.com", "konu", "gövde"))


def _configured_client(tmp_path) -> GmailClient:
    creds = tmp_path / "credentials.json"
    creds.write_text("{}", encoding="utf-8")
    return GmailClient(creds, tmp_path / "token.json")


def test_list_recent_delegates_to_sync_implementation(tmp_path, monkeypatch):
    client = _configured_client(tmp_path)
    monkeypatch.setattr(client, "_list_recent_sync", lambda n: [{"id": "1"}] * n)

    result = asyncio.run(client.list_recent(max_results=3))

    assert result == [{"id": "1"}, {"id": "1"}, {"id": "1"}]


def test_list_recent_wraps_failures(tmp_path, monkeypatch):
    def _raise(n):
        raise RuntimeError("boom")

    client = _configured_client(tmp_path)
    monkeypatch.setattr(client, "_list_recent_sync", _raise)

    with pytest.raises(GmailUnavailableError):
        asyncio.run(client.list_recent())


def test_create_draft_delegates_and_returns_id(tmp_path, monkeypatch):
    client = _configured_client(tmp_path)
    monkeypatch.setattr(client, "_create_draft_sync", lambda to, s, b: "draft-1")

    result = asyncio.run(client.create_draft("a@b.com", "konu", "gövde"))

    assert result == "draft-1"


def test_send_delegates_and_returns_id(tmp_path, monkeypatch):
    client = _configured_client(tmp_path)
    monkeypatch.setattr(client, "_send_sync", lambda to, s, b: "msg-1")

    result = asyncio.run(client.send("a@b.com", "konu", "gövde"))

    assert result == "msg-1"


def test_send_wraps_failures(tmp_path, monkeypatch):
    def _raise(to, s, b):
        raise RuntimeError("smtp down")

    client = _configured_client(tmp_path)
    monkeypatch.setattr(client, "_send_sync", _raise)

    with pytest.raises(GmailUnavailableError):
        asyncio.run(client.send("a@b.com", "konu", "gövde"))


def test_build_raw_message_is_valid_base64():
    import base64

    raw = GmailClient._build_raw_message("a@b.com", "konu", "gövde metni")
    decoded = base64.urlsafe_b64decode(raw.encode("ascii")).decode("utf-8")

    assert "a@b.com" in decoded
    assert "konu" in decoded
