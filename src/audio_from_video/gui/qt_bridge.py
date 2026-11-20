"""Qt adapter marshalling worker-thread ``PromptBridge`` requests onto the UI thread."""

from __future__ import annotations

from typing import TYPE_CHECKING

import shiboken6
from PySide6.QtCore import QCoreApplication, QObject, Qt, Signal, Slot

if TYPE_CHECKING:
    from audio_from_video.gui.bridge import PromptRequest


class _PromptSignalRelay(QObject):
    """Child ``QObject`` emitting internal cross-thread prompt and notice signals."""

    dispatch_requested = Signal(object)  # PromptRequest
    notice_posted = Signal(str)


class QtPromptAdapter(QObject):
    """UI-thread ``QObject`` exposing ``dispatch`` and ``notify`` for ``PromptBridge``.

    Internal signals are connected to bound methods on the adapter itself so
    worker emissions are always queued onto the UI thread.
    """

    prompt_requested = Signal(object)  # PromptRequest
    notice_received = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._active = True
        self._relay = _PromptSignalRelay(self)
        self._relay.dispatch_requested.connect(
            self._deliver_prompt,
            Qt.ConnectionType.QueuedConnection,
        )
        self._relay.notice_posted.connect(
            self._deliver_notice,
            Qt.ConnectionType.QueuedConnection,
        )

    def _deactivate(self) -> None:
        """Stop forwarding queued prompt or notice signals during teardown."""
        self._active = False

    def dispatch(self, request: PromptRequest) -> None:
        """Emit an internal signal carrying ``request`` from the worker thread."""
        if not self._active or QCoreApplication.instance() is None:
            return
        if not shiboken6.isValid(self) or not shiboken6.isValid(self._relay):
            return
        try:
            self._relay.dispatch_requested.emit(request)
        except RuntimeError:
            self._active = False

    def notify(self, message: str) -> None:
        """Emit an internal signal carrying ``message`` from the worker thread."""
        if not self._active or QCoreApplication.instance() is None:
            return
        if not shiboken6.isValid(self) or not shiboken6.isValid(self._relay):
            return
        try:
            self._relay.notice_posted.emit(message)
        except RuntimeError:
            self._active = False

    @Slot(object)
    def _deliver_prompt(self, request: object) -> None:
        """Forward a queued ``PromptRequest`` on the UI thread."""
        if not self._active:
            return
        self.prompt_requested.emit(request)

    @Slot(str)
    def _deliver_notice(self, message: str) -> None:
        """Forward a queued user-facing notice on the UI thread."""
        if not self._active:
            return
        self.notice_received.emit(message)
