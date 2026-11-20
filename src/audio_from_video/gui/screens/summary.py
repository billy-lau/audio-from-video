"""Summary screen displaying outcome counts, failures, unmatched tracks, and actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from audio_from_video.gui.controller import RunController


class SummaryScreen(QWidget):
    """Summary screen widget (listens to ``summary_ready`` and ``error``)."""

    def __init__(
        self,
        controller: RunController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
