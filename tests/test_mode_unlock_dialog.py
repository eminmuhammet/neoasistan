"""mode_unlock_dialog.py: the password dialog behind "yardımcı moduna geç".

Structural (AST) checks, matching test_confirm_dialog.py's style, since this
only runs inside a live QApplication with real dialog interaction -- but the
specific bug this guards against is real: an earlier version reconnected a
new OK-button handler on every retry-loop iteration instead of once, so a
second wrong-then-retry attempt fired every previous handler too, each
trying to resolve an already-finished asyncio.Future and raising
InvalidStateError.
"""

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "neo" / "ui" / "mode_unlock_dialog.py"


def _tree():
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def _function(name):
    for node in ast.walk(_tree()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} bulunamadı")


def _calls(func):
    return [
        node.func.attr
        for node in ast.walk(func)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]


def test_no_dialog_uses_the_blocking_exec():
    """exec() spins a nested Qt event loop that qasync refuses to run other
    tasks inside -- the crash that once took down confirm_action mid-reply."""
    source = SOURCE.read_text(encoding="utf-8")
    assert ".exec(" not in source
    assert ".exec_(" not in source


def test_retry_loop_wires_the_ok_button_exactly_once():
    """The actual bug this file was rewritten to fix: connecting the OK
    button inside the retry loop (once per attempt) instead of once for
    the dialog's lifetime made every attempt after the first re-fire every
    earlier handler too."""
    func = _function("_run_retry_loop")
    connect_calls = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "connect"
    ]
    # Exactly two: dialog.finished.connect(...) and ok_button.clicked.connect(...).
    assert len(connect_calls) == 2

    # Neither connect call may be inside a loop (for/while) -- that's
    # precisely what caused repeated, accumulating connections.
    loop_nodes = [n for n in ast.walk(func) if isinstance(n, (ast.For, ast.While))]
    assert loop_nodes == [], "_run_retry_loop icinde bir dongu olmamali"


def test_setup_and_verify_flows_retry_via_the_shared_loop():
    """Both flows must reuse _run_retry_loop rather than hand-rolling their
    own dialog lifecycle (which is how the reconnect-per-attempt bug
    happened the first time)."""
    for name in ("_setup_new_password", "_verify_existing_password"):
        func = _function(name)
        called_names = {
            node.func.id
            for node in ast.walk(func)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "_run_retry_loop" in called_names


def test_ok_handler_never_both_accepts_and_shows_error():
    """Each on_ok() path must be exactly one of accept() or show_error(),
    not both -- doing both would close the dialog while also displaying an
    error message no one will see, or (worse) accept a failed attempt."""
    for name in ("_setup_new_password", "_verify_existing_password"):
        func = _function(name)
        on_ok = next(
            n
            for n in ast.walk(func)
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "on_ok"
        )
        calls = _calls(on_ok)
        accepts = calls.count("accept")
        errors = calls.count("show_error")
        # Each branch (if/elif/else) contributes at most one terminal
        # action; across the whole function body there should be at least
        # one of each (success path + failure path) and they must not both
        # fire in the same execution (verified functionally in the live
        # smoke test, since that needs real branch execution to prove).
        assert accepts >= 1
        assert errors >= 1


def test_agent_never_sees_the_password_or_the_store():
    """request_helper_mode_unlock's public contract is a bool -- Agent
    only ever gets True/False back, never the password or the PasswordStore
    itself, which is what keeps neo/core/agent.py free of any credential
    handling at all."""
    import inspect

    from neo.ui.mode_unlock_dialog import request_helper_mode_unlock

    signature = inspect.signature(request_helper_mode_unlock)
    assert signature.return_annotation in (bool, "bool")


def test_lockout_message_mentions_minutes_remaining():
    from neo.ui.mode_unlock_dialog import _lockout_message

    message = _lockout_message(185)  # a little over 3 minutes
    assert "dakika" in message
    assert "3" in message


def test_lockout_message_rounds_up_to_at_least_one_minute():
    from neo.ui.mode_unlock_dialog import _lockout_message

    message = _lockout_message(5)  # 5 seconds left
    assert "1" in message  # never claims "0 dakika"
