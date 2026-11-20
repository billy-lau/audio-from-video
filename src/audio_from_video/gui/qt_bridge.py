"""Qt adapter marshalling worker-thread ``PromptBridge`` requests onto the UI thread."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:
    from audio_from_video.gui.bridge import PromptRequest


class QtPromptAdapter(QObject):
    """UI-thread ``QObject`` exposing ``dispatch`` and ``notify`` for ``PromptBridge``.

    Internal signals are connected to bound methods on the adapter itself so
    worker emissions are always queued onto the UI thread.
    """

    prompt_requested = Signal(object)  # PromptRequest
    notice_received = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

    def dispatch(self, request: PromptRequest) -> None:
        """Emit an internal signal carrying ``request`` from the worker thread."""
        _ = request
        raise NotImplementedError("QtPromptAdapter is implemented in Stage 2 lane G.")

    def notify(self, message: str) -> None:
        """Emit an internal signal carrying ``message`` from the worker thread."""
        _ = message
        raise NotImplementedError("QtPromptAdapter is implemented in Stage 2 lane G.")
