"""``afv-gui`` entry point: check the Python runtime, then start the desktop app.

Kept free of Qt imports, so an unsupported runtime gets a clear message instead of an
import error or a later crash.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

# PySide6 6.12's Linux build drops a reference to True on every Signal.emit() from
# Python. Python 3.12 made True immortal; on 3.11 the app would eventually abort.
LINUX_MIN_PYTHON = (3, 12)


def unsupported_runtime_message(
    platform: str | None = None,
    version: tuple[int, int] | None = None,
) -> str | None:
    """Return why the desktop app can't run on this runtime, or ``None`` if it can."""
    plat = sys.platform if platform is None else platform
    major, minor = sys.version_info[:2] if version is None else version
    if plat.startswith("linux") and (major, minor) < LINUX_MIN_PYTHON:
        need = ".".join(map(str, LINUX_MIN_PYTHON))
        return (
            f"The Audio from Video desktop app needs Python {need} or newer on Linux "
            f"(this is Python {major}.{minor}). The Qt library it uses (PySide6 6.12) has a "
            "bug on Linux that crashes Python 3.11 after a while.\n"
            f"Install the app with Python {need} or newer (for example: "
            f'uv tool install --python {need} "audio-from-video[gui]"), or use the afv '
            "command-line tool, which works on this Python."
        )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the desktop app, or explain why it can't run on this Python."""
    message = unsupported_runtime_message()
    if message is not None:
        print(message, file=sys.stderr)
        return 1
    from audio_from_video.gui.app import main as app_main

    return app_main(argv)
