"""Planning progress screen showing probing, lookup stats, notices, and cancel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from audio_from_video.gui.controller import RunController


class PlanningScreen(QWidget):
    """Planning progress screen widget.

    Listens to ``event_received``, ``stats_changed``, and ``notice``.
    """

    def __init__(
        self,
        controller: RunController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
