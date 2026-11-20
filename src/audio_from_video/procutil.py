"""Subprocess helpers shared by every FFmpeg and ffprobe call."""

from __future__ import annotations

import subprocess
import sys

# Windows CREATE_NO_WINDOW; subprocess only exposes the constant on Windows.
_CREATE_NO_WINDOW = 0x08000000


def no_window_creationflags() -> int:
    """creationflags that stop a console window flashing up for each call on Windows.

    A console program started from a windowed app (such as a GUI front end) gets its own
    console window unless CREATE_NO_WINDOW is set. Elsewhere the value is 0.
    """
    if sys.platform == "win32":
        return int(getattr(subprocess, "CREATE_NO_WINDOW", _CREATE_NO_WINDOW))
    return 0
