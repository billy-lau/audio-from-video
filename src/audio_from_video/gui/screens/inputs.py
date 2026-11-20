"""Inputs screen for choosing media files, folders, output directory, and run options."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from audio_from_video.gui.controller import RunController


class InputsScreen(QWidget):
    """Inputs screen widget (listens to ``input_error`` and ``settings_saved``)."""

    def __init__(
        self,
        controller: RunController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
