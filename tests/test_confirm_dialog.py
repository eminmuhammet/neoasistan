"""The confirmation dialog must not block the asyncio loop.

`box.exec()` spins a nested Qt event loop while the awaiting coroutine is
suspended. qasync refuses to run other tasks inside that loop, so a routine
background timer firing mid-dialog raised "Cannot enter into task ... while
another task is being executed" and destroyed the waiting task -- observed
live killing an update part-way through the install.
"""

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "neo" / "ui" / "main_window.py"


def _function(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} bulunamadı")


def _calls(func):
    return {
        node.func.attr
        for node in ast.walk(func)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def test_confirmation_does_not_use_exec():
    calls = _calls(_function("confirm_action"))

    assert "exec" not in calls, "exec() nested bir Qt döngüsü açıyor, qasync bunu reddediyor"
    assert "open" in calls


def test_confirmation_awaits_a_future():
    """The answer has to arrive through the loop, not by blocking it."""
    func = _function("confirm_action")

    assert any(isinstance(node, ast.Await) for node in ast.walk(func))
    assert "create_future" in _calls(func)


def test_confirmation_is_still_a_coroutine():
    """PermissionManager awaits this; it denies by default if it can't."""
    assert isinstance(_function("confirm_action"), ast.AsyncFunctionDef)


def test_cancelling_closes_the_dialog():
    """A cancelled command must not leave an orphaned modal on screen."""
    func = _function("confirm_action")
    handlers = [n for n in ast.walk(func) if isinstance(n, ast.ExceptHandler)]

    assert handlers, "CancelledError ele alınmıyor"
    assert "close" in _calls(func)


def test_about_dialog_does_not_use_the_blocking_static_method():
    """QMessageBox.about() calls exec() internally -- the same nested-loop
    hazard confirm_action had. Live evidence: a qasync 'Cannot enter into
    task' crash at the exact moment a reply was being handled, on a build
    where the About dialog still used QMessageBox.about(). The task that
    crashed was the one carrying the reply to be spoken, which is why the
    voice reply went missing that session."""
    calls = _calls(_function("_on_info_clicked"))

    assert "about" not in calls, "QMessageBox.about() ic ice Qt dongusu aciyor"
    assert "open" in calls
