"""Window-modal dialog for supplying an English alias for non-Latin metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QWidget

if TYPE_CHECKING:
    from audio_from_video.gui.bridge import PromptRequest


class AliasDialog(QDialog):
    """Prompt dialog returning an ``AliasPromptResponse`` (opened with ``open()``)."""

    answered = Signal(int, object)  # request_id, AliasPromptResponse

    def __init__(self, request: PromptRequest, parent: QWidget) -> None:
        super().__init__(parent)
        self._request = request
