"""Desktop GUI entry point (``afv-gui``) and top-level ``MainWindow`` container."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QMainWindow, QWidget

if TYPE_CHECKING:
    from audio_from_video.gui.controller import RunController


class MainWindow(QMainWindow):
    """Top-level application window hosting the screens, model, Settings dialog, and dialogs."""

    def __init__(
        self,
        controller: RunController | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller


def main() -> int:
    """Launch the ``afv-gui`` desktop application."""
    raise NotImplementedError("audio_from_video.gui.app.main is implemented in Stage 3.")
