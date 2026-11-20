"""First-run Setup screen for MusicBrainz contact, FFmpeg check, and offline opt-in."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from audio_from_video.gui.controller import RunController


class SetupScreen(QWidget):
    """Setup screen widget (listens to ``binaries_status`` and ``setup_error``)."""

    def __init__(
        self,
        controller: RunController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
