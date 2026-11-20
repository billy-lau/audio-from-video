"""Plan review screen displaying the planned tracks, banners, filters, and details pane."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from audio_from_video.gui.controller import RunController
    from audio_from_video.gui.models.plan_table import PlanTableModel


class ReviewScreen(QWidget):
    """Plan review screen widget (listens to ``banner`` and ``notice``; reads ``model``)."""

    def __init__(
        self,
        controller: RunController,
        model: PlanTableModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._model = model
