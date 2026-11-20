"""Settings screen for editing config keys, cache maintenance, and GUI defaults."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from audio_from_video.gui.controller import RunController


class SettingsScreen(QWidget):
    """Settings screen widget.

    Listens to ``cache_stats_ready``, ``cache_cleared``, and ``settings_saved``.
    """

    def __init__(
        self,
        controller: RunController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
