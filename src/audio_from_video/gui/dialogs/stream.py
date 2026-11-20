"""Window-modal dialog for selecting an audio stream on multi-track input files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QWidget

if TYPE_CHECKING:
    from audio_from_video.gui.bridge import PromptRequest


class StreamDialog(QDialog):
    """Prompt dialog for choosing an ``AudioStreamInfo`` (opened with ``open()``)."""

    answered = Signal(int, object)  # request_id, AudioStreamInfo

    def __init__(self, request: PromptRequest, parent: QWidget) -> None:
        super().__init__(parent)
        self._request = request
