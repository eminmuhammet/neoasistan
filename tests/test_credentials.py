"""PasswordStore: the helper-mode password never touches disk in plain
text, only a salted scrypt hash does, with a lockout against brute force."""

import json
import time

import pytest

from neo.config.credentials import (
    LOCKOUT_SECONDS,
    MAX_ATTEMPTS,
    NoPasswordSetError,
    PasswordStore,
)


def _store(tmp_path):
    return PasswordStore(tmp_path / "access.json")


def test_not_configured_before_a_password_is_set(tmp_path):
    assert _store(tmp_path).is_configured() is False


def test_set_and_verify_correct_password(tmp_path):
    store = _store(tmp_path)
    store.set_password("gizli-sifre-123")

    assert store.is_configured() is True
    assert store.verify("gizli-sifre-123") is True


def test_wrong_password_is_rejected(tmp_path):
    store = _store(tmp_path)
    store.set_password("dogru-sifre")

    assert store.verify("yanlis-sifre") is False


def test_verify_without_a_configured_password_raises(tmp_path):
    with pytest.raises(NoPasswordSetError):
        _store(tmp_path).verify("her-hangi-bir-sey")


def test_password_is_never_written_to_disk_in_plain_text(tmp_path):
    path = tmp_path / "access.json"
    store = PasswordStore(path)
    secret = "cok-gizli-bir-sifre-buraya"

    store.set_password(secret)

    raw = path.read_text(encoding="utf-8")
    assert secret not in raw
    payload = json.loads(raw)
    assert "salt" in payload and "hash" in payload
    assert payload["hash"] != secret


def test_same_password_hashes_differently_with_different_salts(tmp_path):
    """Confirms a real per-password salt is used -- without one, two users
    with the same password would have identical hashes on disk, letting an
    attacker recognize repeated passwords across installs."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    PasswordStore(a).set_password("ayni-sifre")
    PasswordStore(b).set_password("ayni-sifre")

    hash_a = json.loads(a.read_text(encoding="utf-8"))["hash"]
    hash_b = json.loads(b.read_text(encoding="utf-8"))["hash"]
    assert hash_a != hash_b


def test_setting_a_new_password_invalidates_the_old_one(tmp_path):
    store = _store(tmp_path)
    store.set_password("eski-sifre")
    store.set_password("yeni-sifre")

    assert store.verify("eski-sifre") is False
    assert store.verify("yeni-sifre") is True


def test_lockout_after_max_failed_attempts(tmp_path):
    store = _store(tmp_path)
    store.set_password("dogru-sifre")

    for _ in range(MAX_ATTEMPTS):
        assert store.verify("yanlis") is False

    # Even the correct password is refused once locked out -- otherwise the
    # lockout would only slow down guessing the wrong password, not
    # actually stop a brute-force loop that eventually hits the right one.
    assert store.verify("dogru-sifre") is False
    assert store.lock_remaining_seconds() > 0


def test_lockout_expires_after_the_window(tmp_path, monkeypatch):
    import neo.config.credentials as creds

    store = _store(tmp_path)
    store.set_password("dogru-sifre")
    for _ in range(MAX_ATTEMPTS):
        store.verify("yanlis")
    assert store.lock_remaining_seconds() > 0

    frozen_future = time.time() + LOCKOUT_SECONDS + 1
    monkeypatch.setattr(creds.time, "time", lambda: frozen_future)

    assert store.lock_remaining_seconds() == 0.0
    assert store.verify("dogru-sifre") is True


def test_a_correct_password_resets_the_failed_attempt_counter(tmp_path):
    store = _store(tmp_path)
    store.set_password("dogru-sifre")

    store.verify("yanlis")
    store.verify("yanlis")
    assert store.verify("dogru-sifre") is True

    # Attempts reset, so it takes a fresh MAX_ATTEMPTS to lock out again --
    # not just one more after the earlier near-miss.
    for _ in range(MAX_ATTEMPTS - 1):
        assert store.verify("yanlis") is False
    assert store.lock_remaining_seconds() == 0.0
