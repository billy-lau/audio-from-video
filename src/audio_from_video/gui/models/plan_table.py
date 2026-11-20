"""Shared ``QAbstractTableModel`` displaying planned and executing tracks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    Qt,
    Slot,
)

if TYPE_CHECKING:
    from audio_from_video.events import TrackEvent
    from audio_from_video.models import PlanItem


class PlanTableModel(QAbstractTableModel):
    """Single table model owned by ``MainWindow`` and shared by Review and Run screens.

    Columns: Status, Source, Destination (relative to the output root), Action,
    Match, and Warnings.

    Roles: ``DisplayRole``, ``ToolTipRole`` (full text), ``DecorationRole``
    (status icon), ``AccessibleTextRole``, ``UserRole`` (``PlanItem``), and
    ``UserRole + 1`` (latest ``TrackStage``).
    """

    COLUMNS: tuple[str, ...] = (
        "Status",
        "Source",
        "Destination",
        "Action",
        "Match",
        "Warnings",
    )

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._items: list[PlanItem] = []

    def rowCount(
        self,
        parent: QModelIndex | QPersistentModelIndex = QModelIndex(),
    ) -> int:
        """Return the number of rows in the model."""
        if parent.isValid():
            return 0
        return len(self._items)

    def columnCount(
        self,
        parent: QModelIndex | QPersistentModelIndex = QModelIndex(),
    ) -> int:
        """Return the number of table columns."""
        if parent.isValid():
            return 0
        return len(self.COLUMNS)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = int(Qt.ItemDataRole.DisplayRole),
    ) -> object:
        """Return column header labels for horizontal display orientation."""
        if (
            orientation == Qt.Orientation.Horizontal
            and role == int(Qt.ItemDataRole.DisplayRole)
            and 0 <= section < len(self.COLUMNS)
        ):
            return self.COLUMNS[section]
        return None

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = int(Qt.ItemDataRole.DisplayRole),
    ) -> object:
        """Return display, tooltip, decoration, accessible, or user data for ``index``."""
        _ = index, role
        return None

    @Slot(object)
    def set_items(self, items: list[PlanItem]) -> None:
        """Replace the current list of planned items."""
        _ = items
        raise NotImplementedError("PlanTableModel is implemented in Stage 2 lane H.")

    @Slot(object)
    def apply_event(self, event: TrackEvent) -> None:
        """Update row state from a Stage 1 or Stage 2 ``TrackEvent``."""
        _ = event
        raise NotImplementedError("PlanTableModel is implemented in Stage 2 lane H.")

    @Slot(int)
    def refresh_row(self, row: int) -> None:
        """Notify attached views that ``row`` has changed."""
        _ = row
        raise NotImplementedError("PlanTableModel is implemented in Stage 2 lane H.")
