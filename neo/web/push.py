from __future__ import annotations

import base64
import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Web Push (iOS 16.4+ for a home-screen PWA, no App Store/Xcode/Mac needed --
# see neo/web/mobile_ui.py) needs a VAPID key pair: the private half signs
# every push so relay servers (Apple's, Google's) know it came from this
# NEO install, the public half goes to the browser so pushManager.subscribe
# can hand back a subscription only this install's private key can use.
_VAPID_CLAIMS_EMAIL = "mailto:neo@localhost"


@dataclass
class PushSubscription:
    id: int
    endpoint: str
    p256dh: str
    auth: str


def load_or_create_vapid_private_key(path: Path) -> Path:
    """Generates the VAPID key pair once and keeps the private half on
    disk -- regenerating it would silently invalidate every subscription
    a phone already made (the browser's stored subscription is tied to the
    public key it subscribed with)."""
    if path.is_file():
        return path
    from py_vapid import Vapid01

    vapid = Vapid01()
    vapid.generate_keys()
    path.parent.mkdir(parents=True, exist_ok=True)
    vapid.save_key(str(path))
    return path


def vapid_public_key_b64(private_key_path: Path) -> str:
    """The base64url-encoded uncompressed EC point the browser's
    `pushManager.subscribe({applicationServerKey: ...})` expects."""
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    from py_vapid import Vapid01

    vapid = Vapid01.from_file(str(private_key_path))
    raw = vapid.public_key.public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


class PushSubscriptionStore:
    """SQLite-backed record of phones that asked to receive NEO's proactive
    push notifications (see neo/web/mobile_ui.py's subscribe flow). Same
    one-file, no-server pattern as every other *_store.py in this codebase.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS push_subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT NOT NULL UNIQUE,
                    p256dh TEXT NOT NULL,
                    auth TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, endpoint: str, p256dh: str, auth: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO push_subscriptions (endpoint, p256dh, auth)
                VALUES (?, ?, ?)
                ON CONFLICT(endpoint) DO UPDATE SET p256dh = excluded.p256dh, auth = excluded.auth
                """,
                (endpoint, p256dh, auth),
            )
            conn.commit()

    def remove(self, endpoint: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,))
            conn.commit()

    def list_all(self) -> list[PushSubscription]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, endpoint, p256dh, auth FROM push_subscriptions"
            ).fetchall()
        return [PushSubscription(r["id"], r["endpoint"], r["p256dh"], r["auth"]) for r in rows]


def send_push_to_all(store: PushSubscriptionStore, private_key_path: Path, title: str, body: str) -> int:
    """Sends one notification to every registered phone. Runs synchronously
    (pywebpush uses `requests`, not httpx) -- callers from the qasync loop
    should wrap this in asyncio.to_thread, same as every other blocking
    call in this codebase.

    A subscription the push service reports as gone (410, or 404 -- the
    phone unsubscribed, reset, or hasn't opened Safari in long enough for
    iOS to drop it) is removed rather than retried forever. Returns how
    many sends actually succeeded.
    """
    import json

    from pywebpush import WebPushException, webpush

    payload = json.dumps({"title": title, "body": body})
    sent = 0
    for sub in store.list_all():
        try:
            webpush(
                subscription_info={
                    "endpoint": sub.endpoint,
                    "keys": {"p256dh": sub.p256dh, "auth": sub.auth},
                },
                data=payload,
                vapid_private_key=str(private_key_path),
                vapid_claims={"sub": _VAPID_CLAIMS_EMAIL},
                headers={"urgency": "normal"},
            )
            sent += 1
        except WebPushException as exc:
            status = getattr(exc.response, "status_code", None)
            if status in (404, 410):
                logger.info("Push aboneliği artık geçersiz, kaldırılıyor")
                store.remove(sub.endpoint)
            else:
                logger.warning("Push gönderilemedi: %s", exc)
    return sent
