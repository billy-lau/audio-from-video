"""Window-modal dialog for choosing a release edition for a matched recording."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from audio_from_video.gui.bridge import EditionPrompt, PromptKind
from audio_from_video.gui.dialogs._common import (
    EM_DASH,
    MiddleElideLabel,
    configure_accessible,
    configure_readonly_table,
    make_heading_font,
    make_table_item,
)
from audio_from_video.presentation import format_edition_details, group_editions

if TYPE_CHECKING:
    from audio_from_video.gui.bridge import PromptRequest
    from audio_from_video.models import ReleaseRef

_EDITION_COLUMNS = (
    "#",
    "Date",
    "Country",
    "Format",
    "Discs",
    "Tracks",
    "Edition",
    "Pressings",
    "Default",
)


class EditionDialog(QDialog):
    """Prompt dialog for picking a ``ReleaseRef`` edition (opened with ``open()``)."""

    answered = Signal(int, object)  # request_id, ReleaseRef

    def __init__(self, request: PromptRequest, parent: QWidget) -> None:
        super().__init__(parent)
        if request.kind != PromptKind.EDITION:
            raise ValueError(f"EditionDialog requires PromptKind.EDITION, got {request.kind!r}")
        if not isinstance(request.payload, EditionPrompt):
            raise TypeError(
                f"EditionDialog requires an EditionPrompt payload, got {request.payload!r}"
            )

        self._request = request
        self._payload: EditionPrompt = request.payload
        self._grouped: list[tuple[ReleaseRef, int]] = group_editions(
            self._payload.releases,
            self._payload.default,
        )
        self._editions: list[ReleaseRef] = [rep for rep, _ in self._grouped]
        self._answered = False
        self._withdrawn = False

        self.setObjectName("edition_dialog")
        self.setWindowTitle("Select Release Edition")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumWidth(600)
        self.resize(780, 380)
        configure_accessible(
            self,
            "Select release edition",
            (
                f"Choose which release edition to use for {self._payload.candidate.artist} - "
                f"{self._payload.candidate.title}."
            ),
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        self._heading_label = QLabel("Select release edition", self)
        self._heading_label.setObjectName("edition_heading_label")
        self._heading_label.setFont(make_heading_font(self._heading_label))
        self._heading_label.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self._heading_label)

        candidate = self._payload.candidate
        candidate_summary = f"{candidate.artist} {EM_DASH} {candidate.title}"
        self._candidate_label = MiddleElideLabel(candidate_summary, self)
        self._candidate_label.setObjectName("edition_candidate_label")
        configure_accessible(
            self._candidate_label,
            "Matched recording",
            f"Recording: {candidate.artist} - {candidate.title}.",
        )
        root.addWidget(self._candidate_label)

        self._file_label = MiddleElideLabel(f"File: {request.media_input.path.name}", self)
        self._file_label.setObjectName("edition_file_label")
        self._file_label.set_full_tooltip(str(request.media_input.path))
        configure_accessible(
            self._file_label,
            "Source file",
            f"Source file name: {request.media_input.path.name}.",
        )
        root.addWidget(self._file_label)

        self._table = QTableWidget(self)
        self._table.setObjectName("edition_table")
        configure_readonly_table(
            self._table,
            _EDITION_COLUMNS,
            accessible_name="Release editions",
            accessible_description=(
                "Grouped release editions showing date, country, format, disc count, "
                "track count, edition details, pressings count, and default flag."
            ),
        )
        self._populate_table()
        self._table.cellDoubleClicked.connect(self._on_row_double_clicked)
        root.addWidget(self._table, stretch=1)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch(1)

        self._use_default_button = QPushButton("Use &default", self)
        self._use_default_button.setObjectName("edition_use_default_button")
        self._use_default_button.setAutoDefault(False)
        self._use_default_button.setToolTip(
            f"Keep the default release edition ({format_edition_details(self._payload.default)})."
        )
        configure_accessible(
            self._use_default_button,
            "Use default edition",
            "Select the recommended default release edition.",
        )
        self._use_default_button.clicked.connect(self._on_use_default)
        buttons.addWidget(self._use_default_button)

        self._use_selected_button = QPushButton("&Use selected edition", self)
        self._use_selected_button.setObjectName("edition_use_selected_button")
        self._use_selected_button.setDefault(True)
        self._use_selected_button.setEnabled(bool(self._editions))
        self._use_selected_button.setToolTip("Apply the highlighted release edition.")
        configure_accessible(
            self._use_selected_button,
            "Use selected edition",
            "Apply the currently highlighted release edition.",
        )
        self._use_selected_button.clicked.connect(self._on_use_selected)
        buttons.addWidget(self._use_selected_button)

        root.addLayout(buttons)

        self.setTabOrder(self._table, self._use_selected_button)
        self.setTabOrder(self._use_selected_button, self._use_default_button)
        if self._editions:
            self._table.setFocus()

    def _populate_table(self) -> None:
        self._table.setRowCount(len(self._grouped))
        default_row = 0
        align_right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        for row, (ref, count) in enumerate(self._grouped):
            is_default = ref == self._payload.default
            if is_default:
                default_row = row

            details = format_edition_details(ref)
            formats_text = ", ".join(ref.formats) if ref.formats else EM_DASH
            discs_text = str(ref.disc_count) if ref.disc_count is not None else EM_DASH
            tracks_text = str(ref.track_count) if ref.track_count is not None else EM_DASH
            details_tip = details
            if ref.status:
                details_tip = f"{details} [{ref.status}]"

            self._table.setItem(row, 0, make_table_item(str(row + 1), alignment=align_right))
            self._table.setItem(row, 1, make_table_item(ref.date or EM_DASH))
            self._table.setItem(row, 2, make_table_item(ref.country or EM_DASH))
            self._table.setItem(row, 3, make_table_item(formats_text))
            self._table.setItem(row, 4, make_table_item(discs_text, alignment=align_right))
            self._table.setItem(row, 5, make_table_item(tracks_text, alignment=align_right))
            self._table.setItem(row, 6, make_table_item(details, tooltip=details_tip))
            self._table.setItem(row, 7, make_table_item(str(count), alignment=align_right))
            self._table.setItem(row, 8, make_table_item("Default" if is_default else ""))

        header = self._table.horizontalHeader()
        for col in (0, 1, 2, 3, 4, 5, 7, 8):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)

        if self._editions:
            self._table.selectRow(default_row)
            self._table.setCurrentCell(default_row, 0)

    def _emit_answer(self, release: ReleaseRef, *, accepted: bool) -> None:
        if self._answered or self._withdrawn:
            return
        self._answered = True
        if accepted:
            super().accept()
        else:
            super().reject()
        self.answered.emit(self._request.request_id, release)

    @Slot()
    def _on_use_selected(self) -> None:
        row = self._table.currentRow()
        if 0 <= row < len(self._editions):
            self._emit_answer(self._editions[row], accepted=True)
        else:
            self._emit_answer(self._payload.default, accepted=True)

    @Slot()
    def _on_use_default(self) -> None:
        self._emit_answer(self._payload.default, accepted=True)

    @Slot(int, int)
    def _on_row_double_clicked(self, row: int, column: int) -> None:
        _ = column
        if 0 <= row < len(self._editions):
            self._emit_answer(self._editions[row], accepted=True)

    def withdraw(self) -> None:
        """Close the dialog programmatically without emitting ``answered``."""
        if self._withdrawn:
            return
        self._withdrawn = True
        if self.isVisible():
            super().reject()

    def reject(self) -> None:
        """Handle Escape or window dismissal by returning the default release edition."""
        if self._withdrawn or self._answered:
            super().reject()
            return
        self._emit_answer(self._payload.default, accepted=False)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Emit the safe default edition once if closed directly while open."""
        if self._withdrawn or self._answered:
            event.accept()
            return
        if self.isVisible():
            self._emit_answer(self._payload.default, accepted=False)
        event.accept()
