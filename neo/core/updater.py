from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..config.version import __version__, is_newer

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT_SECONDS = 120


class UpdateError(Exception):
    """Raised when an update can't be checked, downloaded or verified."""


@dataclass
class UpdateInfo:
    version: str
    url: str
    sha256: str
    notes: str = ""


def _require_https(url: str) -> None:
    if not url.lower().startswith("https://"):
        raise UpdateError("Güncelleme adresi https olmalı.")


async def check_for_update(manifest_url: str) -> UpdateInfo | None:
    """Fetches the release manifest and returns it only if it advertises a
    newer version than the one running.

    The manifest is a small JSON:
        {"version": "0.2.0", "url": "https://.../NEO-0.2.0.zip",
         "sha256": "…", "notes": "…"}

    The manifest URL is the trust anchor -- it must be an address the user
    controls, since whatever it points at will be unpacked over the install.
    """
    _require_https(manifest_url)
    payload = await asyncio.to_thread(_fetch_json, manifest_url)

    version = str(payload.get("version", "")).strip()
    url = str(payload.get("url", "")).strip()
    sha256 = str(payload.get("sha256", "")).strip().lower()
    if not version or not url or not sha256:
        raise UpdateError("Güncelleme bilgisi eksik (version/url/sha256).")
    _require_https(url)

    if not is_newer(version, __version__):
        return None
    return UpdateInfo(version=version, url=url, sha256=sha256, notes=str(payload.get("notes", "")))


def _fetch_json(url: str) -> dict:
    import httpx

    try:
        response = httpx.get(url, timeout=15.0, follow_redirects=True)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.exception("Güncelleme bilgisi alınamadı")
        raise UpdateError("Güncelleme sunucusuna ulaşamadım.") from exc


async def download_update(info: UpdateInfo) -> Path:
    return await asyncio.to_thread(_download_and_verify, info)


def _download_and_verify(info: UpdateInfo) -> Path:
    import httpx

    # Re-checked here and not only in check_for_update: this is the function
    # that fetches the bytes which get unpacked over the installation, so it
    # must not rely on a caller having validated the address. A live test
    # showed an http:// URL getting past this point and being rejected only
    # because that particular host happened to 404.
    _require_https(info.url)

    target = Path(tempfile.gettempdir()) / f"NEO-{info.version}.zip"
    digest = hashlib.sha256()
    try:
        with httpx.stream(
            "GET", info.url, timeout=DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=True
        ) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
                    digest.update(chunk)
    except Exception as exc:
        logger.exception("Güncelleme indirilemedi")
        raise UpdateError("Güncelleme paketi indirilemedi.") from exc

    if digest.hexdigest() != info.sha256:
        target.unlink(missing_ok=True)
        raise UpdateError("Güncelleme paketinin doğrulaması başarısız (sha256 uyuşmuyor).")

    if not zipfile.is_zipfile(target):
        target.unlink(missing_ok=True)
        raise UpdateError("Güncelleme paketi geçerli bir zip değil.")

    return target


def install_dir() -> Path:
    """Where the app lives -- the folder holding the frozen exe, or the
    project root when running from source."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent.parent


def apply_update(zip_path: Path, target_dir: Path | None = None) -> None:
    """Hands the swap to a detached helper script and returns.

    Windows locks the running executable, so the app itself can't overwrite
    its own files: the helper waits for this process to exit first, then
    replaces the folder and relaunches.
    """
    target = target_dir or install_dir()
    staging = Path(tempfile.mkdtemp(prefix="neo-update-"))
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(staging)

    # A zip that contains a single top-level folder unwraps to that folder.
    entries = list(staging.iterdir())
    source = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging

    relaunch = sys.executable if getattr(sys, "frozen", False) else str(target / "run_neo.py")
    script = Path(tempfile.gettempdir()) / "neo_update.cmd"
    script.write_text(
        "@echo off\r\n"
        "echo NEO guncelleniyor...\r\n"
        f'"{sys.executable}" -c "import time; time.sleep(3)"\r\n'
        f'robocopy "{source}" "{target}" /E /IS /IT /R:2 /W:1 >nul\r\n'
        f'start "" "{relaunch}"\r\n'
        'del "%~f0"\r\n',
        encoding="utf-8",
    )

    subprocess.Popen(
        ["cmd.exe", "/c", str(script)],
        creationflags=getattr(subprocess, "DETACHED_PROCESS", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        close_fds=True,
        cwd=os.path.dirname(str(script)),
    )
