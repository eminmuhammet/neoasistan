from __future__ import annotations

import logging
import secrets
import threading
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MessageIn(BaseModel):
    text: str

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


def build_app(agent: MessageHandler, token: str):
    """Builds the FastAPI app. Deferred import: only the live web panel
    (and its tests) need fastapi, keeping it out of every other module's
    import graph."""
    from fastapi import FastAPI, Header, HTTPException

    app = FastAPI(title="NEO Panel", docs_url=None, redoc_url=None)

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
    ) -> None:
        self._agent = agent
        self._token = token
        self._host = host
        self._port = port
        self._thread: threading.Thread | None = None
        self._server = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="neo-web-panel")
        self._thread.start()

    def _run(self) -> None:
        import uvicorn

        app = build_app(self._agent, self._token)
        config = uvicorn.Config(app, host=self._host, port=self._port, log_level="warning")
        self._server = uvicorn.Server(config)
        try:
            self._server.run()
        except Exception:
            logger.exception("Web paneli başlatılamadı")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
