"""Guard against the PySide6 refcount bug that crashes the desktop app on Python 3.11.

PySide6 6.12.0's Linux build drops references it doesn't own: one to ``True`` on every
``Signal.emit()``, and one to ``None`` on every call that returns ``None`` (for
example, any setter). Python 3.12 and later made ``True`` and ``None`` immortal, but
on Python 3.11 a count eventually reaches zero and the interpreter aborts
("bool_dealloc: deallocating True or False"). The app makes such calls constantly, so
real sessions would crash. The desktop app therefore requires Python 3.12+ on Linux
(see ``gui/launcher.py``). CI runs these checks on macOS and Windows with Python 3.11,
where the app is still supported, so a leaky build on those platforms fails here. The
file name sorts first so the checks run before a leak could abort the session.
"""

from __future__ import annotations

import sys

import pytest
from PySide6.QtCore import QObject, Signal

pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 12), reason="True and None are immortal on Python 3.12+"
)

_CALLS = 100
_FIX = "This PySide6 build has the 6.12.0 refcount bug and will abort Python 3.11."


class _Emitter(QObject):
    changed = Signal(int)


def test_signal_emit_keeps_a_reference_to_its_return_value() -> None:
    emitter = _Emitter()
    held: list[object] = []
    before = sys.getrefcount(True)
    for _ in range(_CALLS):
        held.append(emitter.changed.emit(1))
    gained = sys.getrefcount(True) - before
    assert gained == _CALLS, (
        f"Signal.emit() leaked {_CALLS - gained} references to True over {_CALLS} calls. {_FIX}"
    )
    assert all(result is True for result in held)


def test_none_returning_calls_keep_a_reference_to_none() -> None:
    obj = QObject()
    held: list[object] = []
    before = sys.getrefcount(None)
    for index in range(_CALLS):
        held.append(obj.setObjectName(f"probe-{index}"))
    gained = sys.getrefcount(None) - before
    assert gained == _CALLS, (
        f"QObject.setObjectName() leaked {_CALLS - gained} references to None over "
        f"{_CALLS} calls. {_FIX}"
    )
