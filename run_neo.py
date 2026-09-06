"""Entry point for launching NEO outside the project directory.

Windows starts autostart entries with an arbitrary working directory, so
`python -m neo.main` wouldn't find the package. Running this file works from
anywhere because Python puts the script's own folder on sys.path.
"""

from neo.main import main

if __name__ == "__main__":
    raise SystemExit(main())
