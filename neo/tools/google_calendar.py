from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


class GoogleCalendarUnavailableError(Exception):
    """Raised when Google Calendar sync isn't configured or fails."""


class GoogleCalendarSync:
    """Optional Google Calendar sync for notes added via NEO.

    This needs a one-time setup that only the user can do -- OAuth requires
    their own Google login, which NEO cannot perform on their behalf:

    1. Go to https://console.cloud.google.com/ , create a project (or reuse
       one), then "APIs & Services" -> "Enabled APIs" -> enable
       "Google Calendar API".
    2. "APIs & Services" -> "Credentials" -> "Create Credentials" ->
       "OAuth client ID" -> Application type "Desktop app".
    3. Download the resulting JSON and save it as `credentials.json` in the
       NEO project root (same folder as this README). It's gitignored.
    4. The first time NEO actually syncs a note, a browser window opens for
       you to sign in and grant calendar access; after that, a token is
       cached at `data/google_token.json` and it works silently.

    Until credentials.json exists, `is_configured()` is False and notes are
    simply kept local-only -- no error, no crash.
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

        self._service = build("calendar", "v3", credentials=creds)
        return self._service

    def _add_event_sync(self, date_str: str, text: str) -> str:
        service = self._get_service()
        # For all-day events Google treats end.date as *exclusive*, so a
        # single-day event ends the following day. Sending end == start
        # describes a zero-length event; Google accepts it and it does show
        # up, but some calendar clients render that oddly.
        end = (date.fromisoformat(date_str) + timedelta(days=1)).isoformat()
        event = {"summary": text, "start": {"date": date_str}, "end": {"date": end}}
        created = service.events().insert(calendarId="primary", body=event).execute()
        return created["id"]

    async def add_event(self, date_str: str, text: str) -> str:
        if not self.is_configured():
            raise GoogleCalendarUnavailableError("Google Takvim yapılandırılmadı.")
        try:
            return await asyncio.to_thread(self._add_event_sync, date_str, text)
        except Exception as exc:
            logger.exception("Google Takvim senkronizasyonu başarısız")
            raise GoogleCalendarUnavailableError("Google Takvim'e yazılamadı.") from exc

    def _delete_event_sync(self, event_id: str) -> None:
        self._get_service().events().delete(calendarId="primary", eventId=event_id).execute()

    async def delete_event(self, event_id: str) -> None:
        if not self.is_configured():
            raise GoogleCalendarUnavailableError("Google Takvim yapılandırılmadı.")
        try:
            await asyncio.to_thread(self._delete_event_sync, event_id)
        except Exception as exc:
            logger.exception("Google Takvim'den silinemedi")
            raise GoogleCalendarUnavailableError("Google Takvim'den silinemedi.") from exc

    def _update_event_sync(self, event_id: str, text: str) -> None:
        service = self._get_service()
        service.events().patch(
            calendarId="primary", eventId=event_id, body={"summary": text}
        ).execute()

    async def update_event(self, event_id: str, text: str) -> None:
        if not self.is_configured():
            raise GoogleCalendarUnavailableError("Google Takvim yapılandırılmadı.")
        try:
            await asyncio.to_thread(self._update_event_sync, event_id, text)
        except Exception as exc:
            logger.exception("Google Takvim güncellenemedi")
            raise GoogleCalendarUnavailableError("Google Takvim güncellenemedi.") from exc
