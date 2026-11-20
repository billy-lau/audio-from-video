"""Run progress screen displaying extraction progress, row states, and cancel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from audio_from_video.gui.controller import RunController
    from audio_from_video.gui.models.plan_table import PlanTableModel


class RunScreen(QWidget):
    """Run progress screen widget (reads the shared ``PlanTableModel``)."""

    def __init__(
        self,
        controller: RunController,
        model: PlanTableModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._model = model
