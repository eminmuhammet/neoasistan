from __future__ import annotations

import logging
import secrets
import threading
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from .push import PushSubscriptionStore

logger = logging.getLogger(__name__)


class MessageIn(BaseModel):
    text: str


class SubscriptionKeys(BaseModel):
    p256dh: str
    auth: str


class SubscriptionIn(BaseModel):
    endpoint: str
    keys: SubscriptionKeys

# 32 bytes of urlsafe base64 is plenty to make guessing infeasible over a
# LAN, while staying short enough to type once from a phone if QR/copy
# isn't convenient.
TOKEN_BYTES = 32
DEFAULT_PORT = 8765


class MessageHandler(Protocol):
    async def handle_message(self, text: str) -> str: ...


def load_or_create_token(path: Path) -> str:
    """The token is the entire access control for this panel -- it exists
    on disk exactly once, generated on first run, and is never transmitted
    anywhere except by the user manually copying it to their phone."""
    if path.is_file():
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    token = secrets.token_urlsafe(TOKEN_BYTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token, encoding="utf-8")
    return token


def build_app(
    agent: MessageHandler,
    token: str,
    push_store: PushSubscriptionStore | None = None,
    vapid_public_key: str | None = None,
):
    """Builds the FastAPI app. Deferred import: only the live web panel
    (and its tests) need fastapi, keeping it out of every other module's
    import graph.

    push_store/vapid_public_key are optional (default None) so existing
    callers/tests that only care about the chat panel don't need to know
    about Web Push at all -- when omitted, /vapid_public_key and
    /subscribe simply report the feature as unavailable rather than
    erroring."""
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import HTMLResponse, Response

    from .mobile_ui import MANIFEST_JSON, PAGE_HTML, SERVICE_WORKER_JS, build_icon_png

    app = FastAPI(title="NEO Panel", docs_url=None, redoc_url=None)
    _icon_cache: dict[str, bytes] = {}

    # No token on these: the page shell, manifest, icon and service worker
    # are not sensitive (nothing here can act on the user's behalf) --
    # gating them behind the header token would just break "open this URL
    # in Safari" and "Add to Home Screen", which can't attach headers to a
    # plain navigation. The token still guards every real action (/status,
    # /message).
    @app.get("/", response_class=HTMLResponse)
    async def mobile_page() -> str:
        return PAGE_HTML

    @app.get("/manifest.json")
    async def manifest() -> Response:
        return Response(content=MANIFEST_JSON, media_type="application/manifest+json")

    @app.get("/sw.js")
    async def service_worker() -> Response:
        return Response(content=SERVICE_WORKER_JS, media_type="application/javascript")

    @app.get("/icon.png")
    async def icon() -> Response:
        if "png" not in _icon_cache:
            _icon_cache["png"] = build_icon_png()
        return Response(content=_icon_cache["png"], media_type="image/png")

    @app.get("/vapid_public_key")
    async def get_vapid_public_key() -> dict[str, Any]:
        # Public by design -- it's the whole point of "public" key; the
        # browser needs it before the user has even entered the token, to
        # ask Notification permission and build a subscription.
        if vapid_public_key is None:
            return {"available": False, "key": None}
        return {"available": True, "key": vapid_public_key}

    def _require_token(x_neo_token: str | None) -> None:
        # Constant-time compare: this token is the only thing standing
        # between "anyone on the LAN" and "controls this computer", so it
        # gets the same treatment as a password, not a plain ==.
        if x_neo_token is None or not secrets.compare_digest(x_neo_token, token):
            raise HTTPException(status_code=401, detail="Geçersiz ya da eksik token.")

    @app.get("/status")
    async def status(x_neo_token: str | None = Header(default=None)) -> dict[str, Any]:
        _require_token(x_neo_token)
        return {"status": "ok"}

    @app.post("/message")
    async def message(
        body: MessageIn, x_neo_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        _require_token(x_neo_token)
        if not body.text.strip():
            raise HTTPException(status_code=400, detail="Mesaj boş olamaz.")
        reply = await agent.handle_message(body.text)
        return {"reply": reply}

    @app.post("/subscribe")
    async def subscribe(
        body: SubscriptionIn, x_neo_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        _require_token(x_neo_token)
        if push_store is None:
            raise HTTPException(status_code=501, detail="Push bildirimleri yapılandırılmadı.")
        push_store.add(body.endpoint, body.keys.p256dh, body.keys.auth)
        return {"success": True}

    @app.post("/unsubscribe")
    async def unsubscribe(
        body: dict[str, str], x_neo_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        _require_token(x_neo_token)
        if push_store is not None and body.get("endpoint"):
            push_store.remove(body["endpoint"])
        return {"success": True}

    return app


class WebPanelServer:
    """Runs the FastAPI panel in its own OS thread with its own event
    loop, so it never competes with (or has to be woken by) the qasync
    loop the rest of NEO runs on. Same LAN-only tradeoff every home
    device with a local web UI makes -- the token is what makes exposing
    it acceptable, not network placement."""

    def __init__(
        self,
        agent: MessageHandler,
        token: str,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        push_store: PushSubscriptionStore | None = None,
        vapid_public_key: str | None = None,
    ) -> None:
        self._agent = agent
        self._token = token
        self._host = host
        self._port = port
        self._push_store = push_store
        self._vapid_public_key = vapid_public_key
        self._thread: threading.Thread | None = None
        self._server = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="neo-web-panel")
        self._thread.start()

    def _run(self) -> None:
        import uvicorn

        app = build_app(self._agent, self._token, self._push_store, self._vapid_public_key)
        config = uvicorn.Config(app, host=self._host, port=self._port, log_level="warning")
        self._server = uvicorn.Server(config)
        try:
            self._server.run()
        except Exception:
            logger.exception("Web paneli başlatılamadı")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
