"""In a windowed (console=False) PyInstaller build, Windows gives the
process no console, so sys.stdout and sys.stderr are None -- not closed,
not redirected, actually None. Any library that assumes a console exists
(tqdm, used by both openWakeWord's model download and huggingface_hub's)
crashes calling `None.write(...)`.

Live reproduction: openWakeWord's download progress bar hit exactly this
inside a background asyncio task, and the corrupted state left the whole
window "Yanıt Vermiyor" (not responding) with nothing further in the log --
the exception was real and even caught correctly, but something it left
behind (tqdm's global thread lock, most likely) silently wedged an
unrelated download running at the same time. run_neo.py installs a no-op
stream in place of None before anything else imports, so this class of
crash can't happen at all, for tqdm or any future dependency doing the
same thing.
"""

import runpy
import sys


def _install_null_streams_like_run_neo_does():
    """Executes run_neo.py's module-level setup (the None-swap) without
    running its __main__ block or importing neo.main, which needs a real
    settings environment this test shouldn't have to provide."""
    runpy.run_path("run_neo.py", run_name="not_main")


def test_none_stdout_is_replaced(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    _install_null_streams_like_run_neo_does()

    assert sys.stdout is not None
    assert sys.stderr is not None


def test_a_real_console_is_left_alone(monkeypatch, capsys):
    """Must not swallow real console output when one genuinely exists --
    e.g. `python -m neo.main` from a terminal during development."""
    import io

    real_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", real_stdout)

    _install_null_streams_like_run_neo_does()

    assert sys.stdout is real_stdout


def test_the_null_stream_never_raises():
    monkeypatch_module = runpy.run_path("run_neo.py", run_name="not_main")
    stream = monkeypatch_module["_NullStream"]()

    stream.write("anything")
    stream.write(b"bytes too")
    stream.flush()
    assert stream.isatty() is False


def test_the_null_stream_satisfies_what_tqdm_actually_calls():
    """Reproduces the real crash directly: a tqdm progress bar constructed
    against a None-like stream used to raise AttributeError from inside
    tqdm's own __init__. Constructing one against _NullStream must not."""
    import tqdm

    module = runpy.run_path("run_neo.py", run_name="not_main")
    stream = module["_NullStream"]()

    bar = tqdm.tqdm(total=100, file=stream, disable=False)
    bar.update(10)
    bar.close()
