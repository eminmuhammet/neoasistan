import asyncio

import pytest

from neo.tools.base import RiskLevel
from neo.tools.gmail import CreateEmailDraftTool, ReadRecentEmailsTool, SendEmailTool
from neo.tools.gmail_client import GmailUnavailableError


class FakeGmailClient:
    def __init__(self) -> None:
        self.sent: list[tuple] = []
        self.drafted: list[tuple] = []
        self.raise_unavailable = False

    async def list_recent(self, max_results=10):
        if self.raise_unavailable:
            raise GmailUnavailableError("Gmail yapılandırılmadı.")
        return [{"id": "1", "from": "a@b.com", "subject": "konu", "date": "", "snippet": ""}][:max_results]

    async def create_draft(self, to, subject, body):
        if self.raise_unavailable:
            raise GmailUnavailableError("Gmail yapılandırılmadı.")
        self.drafted.append((to, subject, body))
        return "draft-1"

    async def send(self, to, subject, body):
        if self.raise_unavailable:
            raise GmailUnavailableError("Gmail yapılandırılmadı.")
        self.sent.append((to, subject, body))
        return "msg-1"


def test_read_recent_emails_returns_messages():
    client = FakeGmailClient()

    result = asyncio.run(ReadRecentEmailsTool(client).run(max_results=1))

    assert result.success is True
    assert result.data["count"] == 1


def test_read_recent_emails_reports_unavailable_cleanly():
    client = FakeGmailClient()
    client.raise_unavailable = True

    result = asyncio.run(ReadRecentEmailsTool(client).run())

    assert result.success is False


def test_create_draft_calls_client_and_returns_id():
    client = FakeGmailClient()

    result = asyncio.run(CreateEmailDraftTool(client).run(to="a@b.com", subject="konu", body="gövde"))

    assert result.success is True
    assert result.data["draft_id"] == "draft-1"
    assert client.drafted == [("a@b.com", "konu", "gövde")]


def test_send_email_calls_client_and_returns_id():
    client = FakeGmailClient()

    result = asyncio.run(SendEmailTool(client).run(to="a@b.com", subject="konu", body="gövde"))

    assert result.success is True
    assert result.data["message_id"] == "msg-1"
    assert client.sent == [("a@b.com", "konu", "gövde")]


def test_send_email_reports_unavailable_cleanly():
    client = FakeGmailClient()
    client.raise_unavailable = True

    result = asyncio.run(SendEmailTool(client).run(to="a@b.com", subject="konu", body="gövde"))

    assert result.success is False


def test_read_and_draft_are_low_risk_but_send_is_high():
    assert ReadRecentEmailsTool.risk == RiskLevel.LOW
    assert CreateEmailDraftTool.risk == RiskLevel.LOW
    assert SendEmailTool.risk == RiskLevel.HIGH
