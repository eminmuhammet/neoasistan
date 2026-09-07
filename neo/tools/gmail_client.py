from __future__ import annotations

import asyncio
import base64
import logging
from email.mime.text import MIMEText
from pathlib import Path

logger = logging.getLogger(__name__)

# Least privilege: readonly for listing/inspecting mail, compose for
# creating drafts, send for the one action that actually reaches someone
# else's inbox. Not "https://mail.google.com/" (full access), which would
# also allow deleting mail NEO has no business touching.
_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
]


class GmailUnavailableError(Exception):
    """Raised when Gmail access isn't configured or a call to it fails."""


class GmailClient:
    """Optional Gmail integration, same one-time OAuth setup as
    GoogleCalendarSync (see neo/tools/google_calendar.py) -- reuses the
    same credentials.json (the app's own identity) but a separate token
    file, since the calendar token was authorized for calendar scopes
    only and reusing it here would fail with insufficient-scope.

    Until credentials.json exists, is_configured() is False and every
    email tool reports it as unavailable rather than erroring -- same
    pattern as calendar sync.
    """

    def __init__(self, credentials_path: Path, token_path: Path) -> None:
        self._credentials_path = credentials_path
        self._token_path = token_path
        self._service = None

    def is_configured(self) -> bool:
        return self._credentials_path.exists()

    def _get_service(self):
        if self._service is not None:
            return self._service

        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        creds = None
        if self._token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self._token_path), _SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(str(self._credentials_path), _SCOPES)
                creds = flow.run_local_server(port=0)
            self._token_path.parent.mkdir(parents=True, exist_ok=True)
            self._token_path.write_text(creds.to_json(), encoding="utf-8")

        self._service = build("gmail", "v1", credentials=creds)
        return self._service

    def _list_recent_sync(self, max_results: int) -> list[dict[str, str]]:
        service = self._get_service()
        response = (
            service.users()
            .messages()
            .list(userId="me", maxResults=max_results, labelIds=["INBOX"])
            .execute()
        )
        messages: list[dict[str, str]] = []
        for item in response.get("messages", []):
            full = (
                service.users()
                .messages()
                .get(
                    userId="me", id=item["id"], format="metadata",
                    metadataHeaders=["From", "Subject", "Date"],
                )
                .execute()
            )
            headers = {h["name"]: h["value"] for h in full["payload"]["headers"]}
            messages.append({
                "id": item["id"],
                "from": headers.get("From", ""),
                "subject": headers.get("Subject", ""),
                "date": headers.get("Date", ""),
                "snippet": full.get("snippet", ""),
            })
        return messages

    async def list_recent(self, max_results: int = 10) -> list[dict[str, str]]:
        if not self.is_configured():
            raise GmailUnavailableError("Gmail yapılandırılmadı.")
        try:
            return await asyncio.to_thread(self._list_recent_sync, max_results)
        except Exception as exc:
            logger.exception("Gmail listelenemedi")
            raise GmailUnavailableError("Gmail okunamadı.") from exc

    @staticmethod
    def _build_raw_message(to: str, subject: str, body: str) -> str:
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject
        return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")

    def _create_draft_sync(self, to: str, subject: str, body: str) -> str:
        service = self._get_service()
        raw = self._build_raw_message(to, subject, body)
        draft = service.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
        return draft["id"]

    async def create_draft(self, to: str, subject: str, body: str) -> str:
        if not self.is_configured():
            raise GmailUnavailableError("Gmail yapılandırılmadı.")
        try:
            return await asyncio.to_thread(self._create_draft_sync, to, subject, body)
        except Exception as exc:
            logger.exception("Gmail taslağı oluşturulamadı")
            raise GmailUnavailableError("Taslak oluşturulamadı.") from exc

    def _send_sync(self, to: str, subject: str, body: str) -> str:
        service = self._get_service()
        raw = self._build_raw_message(to, subject, body)
        sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return sent["id"]

    async def send(self, to: str, subject: str, body: str) -> str:
        if not self.is_configured():
            raise GmailUnavailableError("Gmail yapılandırılmadı.")
        try:
            return await asyncio.to_thread(self._send_sync, to, subject, body)
        except Exception as exc:
            logger.exception("E-posta gönderilemedi")
            raise GmailUnavailableError("E-posta gönderilemedi.") from exc
