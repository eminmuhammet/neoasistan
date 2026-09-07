"""The wake-up cue is spoken audio, so the mic must be muted while it plays --
otherwise the always-on wake loop can hear NEO's own "Dinliyorum efendim"
and hand it back as if the user had said it.

Structural (AST) rather than a live Qt test, matching test_shutdown_names.py:
this path only runs inside a running QApplication with a real audio device.
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


def test_wake_loop_is_muted_by_the_speaking_flag():
    """`is_muted` must be wired to the same flag _on_wake_detected sets."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    source = SOURCE.read_text(encoding="utf-8")
    assert "is_muted=lambda: self._speaking" in source


def test_on_wake_detected_sets_speaking_around_the_cue():
    func = _function("_on_wake_detected")
    dump = ast.dump(func)

    assert "play_activation_chime" in dump
    # Must set the flag before awaiting the cue and clear it afterward via
    # try/finally, not a plain assignment that a raised exception could skip.
    has_try = any(isinstance(n, ast.Try) for n in ast.walk(func))
    assert has_try, "cue oynatimi try/finally icinde olmali"

    assigns_true = [
        n
        for n in ast.walk(func)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "_speaking" for t in n.targets)
        and isinstance(n.value, ast.Constant)
        and n.value.value is True
    ]
    assert assigns_true, "_speaking = True set edilmiyor"


def test_speaking_flag_is_cleared_in_a_finally_block():
    func = _function("_on_wake_detected")
    try_nodes = [n for n in ast.walk(func) if isinstance(n, ast.Try)]
    assert try_nodes

    cleared_in_finally = any(
        isinstance(stmt, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "_speaking" for t in stmt.targets)
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is False
        for try_node in try_nodes
        for stmt in try_node.finalbody
    )
    assert cleared_in_finally, "_speaking, cue basarisiz olsa bile finally'de False'a donmeli"
