"""Guards against the failure that broke the update twice: a name used in
the shutdown path not actually being imported.

`_install_update` runs as a fire-and-forget task, so a NameError there is
swallowed into an unhandled-task warning -- NEO closed its window, the
process stayed alive, and the update silently never applied. Nothing in the
suite touched that line because exercising it needs a running Qt app.
"""

import ast
import builtins
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "neo" / "ui" / "main_window.py"


def _module_names(tree):
    """Everything the module defines or imports at top level."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} bulunamadı")


def _free_names(func):
    """Global names a function references, minus its own locals/params."""
    bound = {a.arg for a in func.args.args} | {a.arg for a in func.args.kwonlyargs}
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Lambda):
            bound.update(a.arg for a in node.args.args)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)  # `except X as exc`
        elif isinstance(node, ast.comprehension):
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name):
                    bound.add(target.id)
    used = {
        node.id
        for node in ast.walk(func)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    return used - bound - set(dir(builtins))


def test_shutdown_path_has_no_undefined_names():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    available = _module_names(tree)

    for function_name in ("_shutdown", "_install_update", "check_for_update"):
        missing = _free_names(_function(tree, function_name)) - available
        assert not missing, f"{function_name} tanımsız isim kullanıyor: {missing}"


def test_qapplication_is_actually_imported():
    """The specific name that broke it -- twice."""
    from neo.ui import main_window

    assert hasattr(main_window, "QApplication")
    assert hasattr(main_window, "QTimer")
    assert hasattr(main_window, "os")


def test_hard_exit_is_armed_before_any_fallible_step():
    """Ordering matters: the earlier version scheduled the forced exit last,
    so the first line raising meant it was never scheduled at all."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    body = _function(tree, "_shutdown").body

    exit_index = next(
        i
        for i, node in enumerate(body)
        if "singleShot" in ast.dump(node) and "_exit" in ast.dump(node)
    )
    cleanup_index = next(i for i, node in enumerate(body) if isinstance(node, ast.For))

    assert exit_index < cleanup_index
