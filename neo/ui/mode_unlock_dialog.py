from __future__ import annotations

import asyncio
import logging

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ..config.credentials import NoPasswordSetError, PasswordStore
from ..core.access_mode import AccessModeManager

logger = logging.getLogger(__name__)

# Kept deliberately independent of main_window.py: this is a standalone
# top-level dialog that main.py can wire directly into
# Agent.set_mode_unlock_control(), so "yardımcı moduna geç" works end to
# end without needing to touch the main window's layout at all. Where the
# mode badge actually lives on screen is a separate, purely visual
# integration step.


class _PasswordDialog(QDialog):
    """One password field, an inline error label, OK/Cancel.

    Never triggered by voice: a spoken password would pass through Whisper
    into the conversation transcript, which syncs to OneDrive -- i.e. the
    password would leave the machine in plain text. This dialog only ever
    opens in response to a request; typing into it is the only way in.
    """

    def __init__(self, title: str, prompt: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(prompt))

        self.input = QLineEdit()
        self.input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.input)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #ff4d4f;")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.input.setFocus()

    def show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.show()
        self.input.clear()
        self.input.setFocus()


class _SetupPasswordDialog(QDialog):
    """First-time setup: no password exists yet, so two fields (enter,
    confirm) instead of one -- a single typo in a never-seen-again password
    would otherwise lock the user out of their own helper mode immediately."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Yardımcı Modu Şifresi Belirle")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Yardımcı modu için henüz bir şifre yok. Bu şifre, NEO'ya "
                "tam yetki verirken kullanılacak -- sadece sen bileceksin, "
                "diskte düz metin olarak tutulmaz."
            )
        )

        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setPlaceholderText("Yeni şifre")
        layout.addWidget(self.password_input)

        self.confirm_input = QLineEdit()
        self.confirm_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.confirm_input.setPlaceholderText("Şifreyi tekrar yaz")
        layout.addWidget(self.confirm_input)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #ff4d4f;")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.password_input.setFocus()

    def show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.show()
        self.password_input.clear()
        self.confirm_input.clear()
        self.password_input.setFocus()


async def _run_retry_loop(dialog: QDialog, on_ok_clicked) -> bool:
    """Keeps `dialog` open across repeated OK-button presses.

    `on_ok_clicked` is an async, no-argument callback: on each OK click it
    either calls `dialog.accept()` (success) or `dialog.show_error(...)`
    (stay open, try again) -- never both, never neither. The OK button is
    wired exactly once, for the dialog's whole lifetime, which is what
    makes retries safe: an earlier version reconnected a new handler on
    every loop iteration without disconnecting the last one, so a second
    attempt fired every previous handler too, each trying to resolve an
    already-finished asyncio.Future and raising InvalidStateError.
    """
    result_future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()

    def _on_finished(code: int) -> None:
        if not result_future.done():
            result_future.set_result(code == QDialog.DialogCode.Accepted)

    dialog.finished.connect(_on_finished)

    ok_button = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
    ok_button.clicked.connect(lambda: asyncio.ensure_future(on_ok_clicked()))

    # open(), not exec(): exec() spins a nested Qt event loop that qasync
    # refuses to run other tasks inside, the same hazard confirm_action
    # once hit (see its own history for the live crash that caused).
    dialog.open()

    try:
        return await result_future
    except asyncio.CancelledError:
        dialog.close()
        raise


async def _setup_new_password(store: PasswordStore, parent: QWidget | None) -> bool:
    dialog = _SetupPasswordDialog(parent)

    async def on_ok() -> None:
        password = dialog.password_input.text()
        confirm = dialog.confirm_input.text()
        if not password:
            dialog.show_error("Şifre boş olamaz.")
            return
        if password != confirm:
            dialog.show_error("Şifreler eşleşmiyor, tekrar dener misin?")
            return
        store.set_password(password)
        dialog.accept()

    return await _run_retry_loop(dialog, on_ok)


async def _verify_existing_password(store: PasswordStore, parent: QWidget | None) -> bool:
    dialog = _PasswordDialog(
        "Yardımcı Moduna Geç",
        "Tam yetkili yardımcı moduna geçmek için şifreni gir:",
        parent,
    )

    remaining = store.lock_remaining_seconds()
    if remaining > 0:
        dialog.show_error(_lockout_message(remaining))

    async def on_ok() -> None:
        remaining = store.lock_remaining_seconds()
        if remaining > 0:
            dialog.show_error(_lockout_message(remaining))
            return
        if store.verify(dialog.input.text()):
            dialog.accept()
        else:
            dialog.show_error("Şifre yanlış, tekrar dener misin?")

    return await _run_retry_loop(dialog, on_ok)


def _lockout_message(remaining_seconds: float) -> str:
    minutes = max(1, round(remaining_seconds / 60))
    return f"Çok fazla yanlış deneme oldu, yaklaşık {minutes} dakika sonra tekrar dene."


async def request_helper_mode_unlock(
    password_store: PasswordStore,
    mode_manager: AccessModeManager,
    parent: QWidget | None = None,
) -> bool:
    """The full flow behind "yardımcı moduna geç": first-time setup if no
    password exists yet, otherwise a verify prompt with retry-on-mismatch
    and the store's own lockout after too many wrong attempts.

    Returns whether helper mode ended up unlocked. Agent never sees the
    password or how it is checked -- only this boolean.
    """
    try:
        if not password_store.is_configured():
            unlocked = await _setup_new_password(password_store, parent)
        else:
            unlocked = await _verify_existing_password(password_store, parent)
    except NoPasswordSetError:
        # Can only happen from a race (deleted between the is_configured()
        # check and verify()); treat it as "not unlocked" rather than
        # crashing the whole request.
        logger.exception("Yardımcı modu şifresi kontrol edilirken tutarsızlık")
        return False

    if unlocked:
        mode_manager.unlock_helper_mode()
    return unlocked
