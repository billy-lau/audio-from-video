"""Window-modal dialog for choosing a release edition for a matched recording."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QWidget

if TYPE_CHECKING:
    from audio_from_video.gui.bridge import PromptRequest


class EditionDialog(QDialog):
    """Prompt dialog for picking a ``ReleaseRef`` edition (opened with ``open()``)."""

    answered = Signal(int, object)  # request_id, ReleaseRef

    def __init__(self, request: PromptRequest, parent: QWidget) -> None:
        super().__init__(parent)
        self._request = request
