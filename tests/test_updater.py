import asyncio
import hashlib
import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neo.config.version import __version__, is_newer, parse_version
from neo.core.updater import UpdateError, UpdateInfo, check_for_update, download_update


def test_parse_version_handles_prefixes_and_junk():
    assert parse_version("v1.2.3") == (1, 2, 3)
    assert parse_version("0.1.0") == (0, 1, 0)
    assert parse_version("2.0.0-beta") == (2, 0, 0)


def test_is_newer_compares_numerically_not_lexically():
    assert is_newer("0.10.0", "0.9.0") is True
    assert is_newer("0.9.0", "0.10.0") is False
    assert is_newer(__version__, __version__) is False


def _manifest_response(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def test_check_for_update_returns_none_when_up_to_date():
    payload = {"version": __version__, "url": "https://example.com/a.zip", "sha256": "ab"}
    with patch("httpx.get", return_value=_manifest_response(payload)):
        assert asyncio.run(check_for_update("https://example.com/latest.json")) is None


def test_check_for_update_returns_info_when_newer():
    payload = {
        "version": "99.0.0",
        "url": "https://example.com/NEO-99.zip",
        "sha256": "abc",
        "notes": "yeni",
    }
    with patch("httpx.get", return_value=_manifest_response(payload)):
        info = asyncio.run(check_for_update("https://example.com/latest.json"))

    assert info is not None
    assert info.version == "99.0.0"
    assert info.notes == "yeni"


def test_check_for_update_rejects_plain_http():
    with pytest.raises(UpdateError):
        asyncio.run(check_for_update("http://example.com/latest.json"))


def test_check_for_update_rejects_incomplete_manifest():
    with patch("httpx.get", return_value=_manifest_response({"version": "99.0.0"})):
        with pytest.raises(UpdateError):
            asyncio.run(check_for_update("https://example.com/latest.json"))


def _zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("NEO/marker.txt", "hello")
    return buffer.getvalue()


class _FakeStream:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self):
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.iter_bytes.return_value = [self._payload]
        return response

    def __exit__(self, *args) -> None:
        return None


def test_download_rejects_a_tampered_package(tmp_path):
    payload = _zip_bytes()
    info = UpdateInfo(version="99.0.0", url="https://example.com/a.zip", sha256="deadbeef")

    with patch("httpx.stream", return_value=_FakeStream(payload)):
        with pytest.raises(UpdateError):
            asyncio.run(download_update(info))


def test_download_accepts_a_matching_checksum(tmp_path):
    payload = _zip_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    info = UpdateInfo(version="99.0.0", url="https://example.com/a.zip", sha256=digest)

    with patch("httpx.stream", return_value=_FakeStream(payload)):
        path = asyncio.run(download_update(info))

    assert Path(path).exists()
    assert zipfile.is_zipfile(path)
    Path(path).unlink(missing_ok=True)
