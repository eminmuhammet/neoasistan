from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from pathlib import Path

# scrypt cost parameters: N=2**14 is hashlib's documented "interactive
# login" profile (RFC 7914) -- expensive enough that brute-forcing a short
# password takes real time per guess, cheap enough that verifying the
# correct one stays instant for a once-in-a-while unlock.
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_KEY_LENGTH = 32
_SALT_LENGTH = 16

# Without a limit, a short password can be tried thousands of times a
# second against an offline copy of the hash. Five tries then a five-minute
# pause makes that impractical while staying out of the way of someone who
# just mistyped once or twice.
MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 300.0


class NoPasswordSetError(Exception):
    """Raised when helper mode is requested but no password has ever been configured."""


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_KEY_LENGTH,
    )


class PasswordStore:
    """Keeps only a salted scrypt hash of the helper-mode password on disk --
    the password itself never touches storage, and this class never holds
    it in memory past the single call that checks it.

    One JSON file, matching the rest of this codebase's "small SQLite/JSON
    file per concern" style: {salt, hash, failed_attempts, locked_until}.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def is_configured(self) -> bool:
        return self._path.exists()

    def set_password(self, password: str) -> None:
        """(Re)sets the password, clearing any lockout/attempt count from a
        previous one -- setting a new password is itself something only
        someone who already has access can do, so it's a reasonable time to
        wipe the slate."""
        salt = os.urandom(_SALT_LENGTH)
        digest = _hash_password(password, salt)
        self._save(
            {
                "salt": salt.hex(),
                "hash": digest.hex(),
                "failed_attempts": 0,
                "locked_until": 0.0,
            }
        )

    def _load(self) -> dict:
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _save(self, payload: dict) -> None:
        self._path.write_text(json.dumps(payload), encoding="utf-8")

    def lock_remaining_seconds(self) -> float:
        """0 when not configured or not currently locked out."""
        if not self.is_configured():
            return 0.0
        payload = self._load()
        return max(0.0, payload.get("locked_until", 0.0) - time.time())

    def verify(self, password: str) -> bool:
        if not self.is_configured():
            raise NoPasswordSetError("Yardımcı modu için henüz bir şifre belirlenmedi.")

        payload = self._load()
        if payload.get("locked_until", 0.0) - time.time() > 0:
            return False

        salt = bytes.fromhex(payload["salt"])
        expected = bytes.fromhex(payload["hash"])
        candidate = _hash_password(password, salt)
        # Constant-time comparison: a plain == on the digests would leak how
        # many leading bytes matched through how long the comparison takes,
        # letting a patient attacker recover the hash byte by byte.
        correct = hmac.compare_digest(candidate, expected)

        if correct:
            payload["failed_attempts"] = 0
            payload["locked_until"] = 0.0
        else:
            payload["failed_attempts"] = payload.get("failed_attempts", 0) + 1
            if payload["failed_attempts"] >= MAX_ATTEMPTS:
                payload["locked_until"] = time.time() + LOCKOUT_SECONDS
                payload["failed_attempts"] = 0
        self._save(payload)
        return correct
