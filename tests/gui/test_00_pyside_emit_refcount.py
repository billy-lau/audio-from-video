"""Guard against the PySide6 ``Signal.emit()`` refcount bug on Python 3.11.

PySide6 6.12.0's Linux build returns ``True`` from ``Signal.emit()`` without taking a
reference, so every emit from Python drops one reference to ``True``. Python 3.12 and
later made ``True`` immortal, but on Python 3.11 the count eventually reaches zero and
the interpreter aborts ("bool_dealloc: deallocating True or False"). The app emits
signals from Python constantly, so this would crash real sessions. The desktop app
therefore requires Python 3.12+ on Linux (see ``gui/launcher.py``); this test fails if
the GUI is ever run on Python 3.11 with a PySide6 build that has the bug. The file name
sorts first so it runs before a leak could abort the session.
"""

from __future__ import annotations

import sys

import pytest
from PySide6.QtCore import QObject, Signal


class _Emitter(QObject):
    changed = Signal(int)


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="True is immortal on Python 3.12+")
def test_signal_emit_keeps_a_reference_to_its_return_value() -> None:
    emitter = _Emitter()
    held: list[object] = []
    before = sys.getrefcount(True)
    for _ in range(100):
        held.append(emitter.changed.emit(1))
    gained = sys.getrefcount(True) - before
    assert gained == len(held), (
        f"Signal.emit() leaked {len(held) - gained} references to True over {len(held)} "
        "emits. This PySide6 build has the 6.12.0 refcount bug and will abort Python "
        "3.11; use Python 3.12 or newer for the desktop app."
    )
