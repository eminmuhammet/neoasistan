"""Entry point for launching NEO outside the project directory.

Windows starts autostart entries with an arbitrary working directory, so
`python -m neo.main` wouldn't find the package. Running this file works from
anywhere because Python puts the script's own folder on sys.path.
"""

import sys


class _NullStream:
    """Stands in for sys.stdout/stderr, which PyInstaller leaves as None in
    a windowed (console=False) build -- there is no console to write to.

    Discovered when openWakeWord's model download crashed the continuous-
    listening background task: its progress bar (tqdm) calls
    `sys.stdout.write()` unconditionally, and `None.write()` raised
    AttributeError deep inside a thread the rest of the app was waiting on,
    which surfaced as the whole window going "Yanıt Vermiyor" with nothing
    further in the log -- the exception was real, just never printed
    anywhere for the same reason it happened. Any other dependency that
    assumes a console exists (tqdm is used by several: huggingface_hub's
    own model downloads, openwakeword's) would hit the same crash on its
    first run in a packaged build, so this is installed unconditionally
    and early, before anything else imports.
    """

    def write(self, *args, **kwargs) -> int:
        return 0

    def flush(self, *args, **kwargs) -> None:
        pass

    def isatty(self) -> bool:
        return False


if sys.stdout is None:
    sys.stdout = _NullStream()
if sys.stderr is None:
    sys.stderr = _NullStream()

from neo.main import main

if __name__ == "__main__":
    raise SystemExit(main())
