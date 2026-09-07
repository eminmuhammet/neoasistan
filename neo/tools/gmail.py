from __future__ import annotations

from .base import RiskLevel, Tool, ToolResult
from .gmail_client import GmailClient, GmailUnavailableError


class ReadRecentEmailsTool(Tool):
    name = "read_recent_emails"
    description = (
        "Gelen kutusundaki en son e-postaların özetini (kimden, konu, "
        "tarih, kısa önizleme) döner. E-postanın tam içeriğini almaz, "
        "sadece bir liste."
    )
    # Same bar as get_calendar_notes: reading the user's own data back to
    # them, not sending anything anywhere.
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "max_results": {"type": "integer", "description": "Kaç e-posta getirileceği (varsayılan 10)."},
        },
    }

    def __init__(self, client: GmailClient) -> None:
        self._client = client

    async def run(self, max_results: int = 10, **kwargs: object) -> ToolResult:
        try:
            messages = await self._client.list_recent(max_results)
        except GmailUnavailableError as exc:
            return ToolResult(success=False, error=str(exc))
        return ToolResult(success=True, data={"messages": messages, "count": len(messages)})


class CreateEmailDraftTool(Tool):
    name = "create_email_draft"
    description = (
        "Gmail'de bir taslak e-posta oluşturur ama GÖNDERMEZ -- kullanıcı "
        "gözden geçirip kendisi göndermek istediğinde kullanılır."
    )
    # Deliberately not gated behind confirmation like send_email is: a
    # draft is invisible to everyone but the user and trivially deletable,
    # matching the plan's "taslak serbest" (drafts are unrestricted).
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Alıcı e-posta adresi."},
            "subject": {"type": "string", "description": "Konu."},
            "body": {"type": "string", "description": "E-posta metni."},
        },
        "required": ["to", "subject", "body"],
    }

    def __init__(self, client: GmailClient) -> None:
        self._client = client

    async def run(self, to: str, subject: str, body: str, **kwargs: object) -> ToolResult:
        try:
            draft_id = await self._client.create_draft(to, subject, body)
        except GmailUnavailableError as exc:
            return ToolResult(success=False, error=str(exc))
        return ToolResult(success=True, data={"draft_id": draft_id, "to": to, "subject": subject})


class SendEmailTool(Tool):
    name = "send_email"
    description = (
        "Bir e-postayı DOĞRUDAN GÖNDERİR (taslak değil). Başkasına ulaşan "
        "geri alınamaz bir eylem olduğu için sadece yardımcı modunda ve "
        "her seferinde kullanıcı onayıyla çalışır. Emin değilsen önce "
        "create_email_draft kullan."
    )
    # HIGH, deliberately: PermissionManager refuses HIGH outright in
    # assistant mode and still asks for confirmation every single time
    # even in helper mode (unlike MEDIUM, which helper mode auto-approves)
    # -- exactly the "her zaman ayrı onay" bar the plan sets for sending
    # messages on the user's behalf.
    risk = RiskLevel.HIGH
    input_schema = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Alıcı e-posta adresi."},
            "subject": {"type": "string", "description": "Konu."},
            "body": {"type": "string", "description": "E-posta metni."},
        },
        "required": ["to", "subject", "body"],
    }

    def __init__(self, client: GmailClient) -> None:
        self._client = client

    async def run(self, to: str, subject: str, body: str, **kwargs: object) -> ToolResult:
        try:
            message_id = await self._client.send(to, subject, body)
        except GmailUnavailableError as exc:
            return ToolResult(success=False, error=str(exc))
        return ToolResult(success=True, data={"message_id": message_id, "to": to, "subject": subject})
