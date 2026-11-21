"""Window-modal dialog for choosing a release edition for a matched recording."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QObject, QSize, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QFont, QKeyEvent, QResizeEvent, QShowEvent
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QStyle,
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
    dialog_button_icon,
    make_context_card,
    make_dialog_header,
    make_divider_line,
    make_heading_font,
    make_notice_banner,
    make_table_item,
    set_button_role,
)
from audio_from_video.gui.resources import load_icon
from audio_from_video.gui.theme import refresh_themed_widgets, repolish
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
_COMPACT_COLUMNS = (0, 1, 2, 3, 4, 5, 7, 8)
_EDITION_COL_INDEX = 6
_MIN_EDITION_COL_WIDTH = 220
_MAX_FORMAT_COL_WIDTH = 156
_BUTTON_ICON_SIZE = QSize(14, 14)
_TABLE_ICON_SIZE = QSize(15, 15)
_KEY_MODIFIER_MASK = (
    Qt.KeyboardModifier.ShiftModifier
    | Qt.KeyboardModifier.ControlModifier
    | Qt.KeyboardModifier.AltModifier
    | Qt.KeyboardModifier.MetaModifier
)


def _format_count_noun(count: int, singular: str, plural: str) -> str:
    """Return a singular or plural count phrase."""
    return f"{count} {singular if count == 1 else plural}"


def _format_edition_count_summary(grouped: list[tuple[ReleaseRef, int]]) -> str:
    """Return a concise summary of grouped edition and pressing counts."""
    edition_count = len(grouped)
    pressing_count = sum(count for _, count in grouped)
    if edition_count == 0:
        return "No editions"
    editions_text = _format_count_noun(edition_count, "edition", "editions")
    pressings_text = _format_count_noun(pressing_count, "pressing", "pressings")
    return f"{editions_text} · {pressings_text}"


def _format_edition_summary(ref: ReleaseRef, count: int, *, is_default: bool) -> str:
    """Build a full unelided summary string for the selected release edition."""
    details = format_edition_details(ref)
    parts: list[str] = []
    if ref.status:
        parts.append(ref.status)
    if ref.date:
        parts.append(ref.date)
    if ref.country:
        parts.append(ref.country)
    if ref.formats:
        parts.append(", ".join(ref.formats))
    if ref.disc_count is not None:
        parts.append(_format_count_noun(ref.disc_count, "disc", "discs"))
    if ref.track_count is not None:
        parts.append(_format_count_noun(ref.track_count, "track", "tracks"))
    parts.append(_format_count_noun(count, "pressing", "pressings"))
    if is_default:
        parts.append("Default")
    return f"{details}  {EM_DASH}  {' · '.join(parts)}"


def _make_caption_font(widget: QWidget) -> QFont:
    """Return a slightly smaller font relative to ``widget`` for subtle labels."""
    return make_heading_font(widget, scale=0.92, bold=False)


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
        self.setMinimumHeight(395)
        self.resize(920, 460)
        configure_accessible(
            self,
            "Select release edition",
            (
                f"Choose which release edition to use for {self._payload.candidate.artist} - "
                f"{self._payload.candidate.title}."
            ),
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(12)

        # -- Header (icon badge, heading, and instructional subtitle) ------------
        self._heading_label = QLabel("Select release edition", self)
        self._heading_label.setObjectName("edition_heading_label")
        self._heading_label.setFont(make_heading_font(self._heading_label, scale=1.28))
        self._heading_label.setProperty("afvText", "title")
        self._heading_label.setTextFormat(Qt.TextFormat.PlainText)
        self._heading_label.setWordWrap(True)
        configure_accessible(
            self._heading_label,
            "Dialog heading",
            "Select a release edition for the matched recording.",
        )

        self._subtitle_label = QLabel(
            "Pressings sharing format, disc count, track count, and edition name are grouped.",
            self,
        )
        self._subtitle_label.setObjectName("edition_subtitle_label")
        self._subtitle_label.setFont(_make_caption_font(self._subtitle_label))
        self._subtitle_label.setWordWrap(True)
        self._subtitle_label.setTextFormat(Qt.TextFormat.PlainText)
        configure_accessible(
            self._subtitle_label,
            "Dialog instructions",
            "Pressings sharing format, disc count, track count, and edition name are grouped.",
        )

        header_row, self._header_icon = make_dialog_header(
            self,
            icon_name="status_info",
            heading_label=self._heading_label,
            subtitle_label=self._subtitle_label,
            icon_size=26,
        )
        configure_accessible(
            self._header_icon,
            "Edition dialog icon",
            "Informational badge indicating release edition selection.",
        )
        root.addLayout(header_row)

        # -- Context card (matched recording and source file) --------------------
        context_card = make_context_card(self, object_name="edition_context_card")
        context_row = QHBoxLayout(context_card)
        context_row.setContentsMargins(12, 8, 12, 8)
        context_row.setSpacing(12)

        context_text_col = QVBoxLayout()
        context_text_col.setContentsMargins(0, 0, 0, 0)
        context_text_col.setSpacing(3)

        candidate = self._payload.candidate
        candidate_summary = f"{candidate.artist} {EM_DASH} {candidate.title}"
        self._candidate_label = MiddleElideLabel(candidate_summary, context_card)
        self._candidate_label.setObjectName("edition_candidate_label")
        cand_font = QFont(self._candidate_label.font())
        cand_font.setWeight(QFont.Weight.DemiBold)
        self._candidate_label.setFont(cand_font)
        configure_accessible(
            self._candidate_label,
            "Matched recording",
            f"Recording: {candidate.artist} - {candidate.title}.",
        )
        context_text_col.addWidget(self._candidate_label)

        self._file_label = MiddleElideLabel(f"File: {request.media_input.path.name}", context_card)
        self._file_label.setObjectName("edition_file_label")
        self._file_label.setFont(_make_caption_font(self._file_label))
        self._file_label.setProperty("afvText", "muted")
        self._file_label.set_full_tooltip(str(request.media_input.path))
        configure_accessible(
            self._file_label,
            "Source file",
            f"Source file name: {request.media_input.path.name}.",
        )
        context_text_col.addWidget(self._file_label)
        context_row.addLayout(context_text_col, stretch=1)

        count_summary = _format_edition_count_summary(self._grouped)
        self._count_label = QLabel(count_summary, context_card)
        self._count_label.setObjectName("edition_count_label")
        self._count_label.setFont(_make_caption_font(self._count_label))
        self._count_label.setProperty("afvText", "caption")
        self._count_label.setTextFormat(Qt.TextFormat.PlainText)
        configure_accessible(
            self._count_label,
            "Edition count",
            f"Grouped summary: {count_summary}.",
        )
        context_row.addWidget(self._count_label, alignment=Qt.AlignmentFlag.AlignTop)
        repolish(context_card, include_children=True)
        root.addWidget(context_card)

        # -- Optional empty-state notice banner ----------------------------------
        self._empty_banner, self._empty_banner_label = make_notice_banner(
            self,
            object_name="edition_empty_banner",
            pixmap_kind=QStyle.StandardPixmap.SP_MessageBoxInformation,
        )
        empty_msg = "No release editions were returned. Choose Use default to continue."
        self._empty_banner_label.setText(empty_msg)
        configure_accessible(self._empty_banner, "No editions notice", empty_msg)
        configure_accessible(self._empty_banner_label, "No editions message", empty_msg)
        self._empty_banner.setVisible(not self._editions)
        root.addWidget(self._empty_banner)

        # -- Grouped editions table ----------------------------------------------
        self._table = QTableWidget(self)
        self._table.setObjectName("edition_table")
        self._table.setProperty("afvDensity", "compact")
        repolish(self._table, include_children=True)
        self._table.setMinimumHeight(140)
        self._table.setIconSize(_TABLE_ICON_SIZE)
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
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.cellDoubleClicked.connect(self._on_row_double_clicked)
        self._table.installEventFilter(self)
        root.addWidget(self._table, stretch=1)

        # -- Selected-edition detail readout strip -------------------------------
        self._detail_card = QFrame(self)
        self._detail_card.setObjectName("edition_detail_card")
        self._detail_card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._detail_card.setFrameShape(QFrame.Shape.StyledPanel)
        self._detail_card.setFrameShadow(QFrame.Shadow.Plain)
        self._detail_card.setProperty("afvSurface", "card-inset")
        detail_layout = QHBoxLayout(self._detail_card)
        detail_layout.setContentsMargins(12, 7, 12, 7)
        detail_layout.setSpacing(8)

        detail_prefix = QLabel("Selected:", self._detail_card)
        detail_prefix.setFont(make_heading_font(detail_prefix, scale=0.92, bold=True))
        detail_prefix.setProperty("afvText", "muted")
        detail_prefix.setTextFormat(Qt.TextFormat.PlainText)
        detail_layout.addWidget(detail_prefix, alignment=Qt.AlignmentFlag.AlignTop)

        self._detail_label = QLabel(self._detail_card)
        self._detail_label.setObjectName("edition_detail_label")
        self._detail_label.setFont(_make_caption_font(self._detail_label))
        self._detail_label.setProperty("afvText", "muted")
        self._detail_label.setTextFormat(Qt.TextFormat.PlainText)
        self._detail_label.setWordWrap(True)
        self._detail_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._detail_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        configure_accessible(
            self._detail_label,
            "Selected edition details",
            "Full details, status, date, country, format, and counts for the highlighted edition.",
        )
        detail_layout.addWidget(self._detail_label, stretch=1)
        repolish(self._detail_card, include_children=True)
        self._detail_card.setVisible(bool(self._editions))
        root.addWidget(self._detail_card)

        root.addWidget(make_divider_line(self))

        # -- Footer action bar ---------------------------------------------------
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(8)

        self._hint_label = QLabel("Double-click or press Enter to confirm.", self)
        self._hint_label.setObjectName("edition_hint_label")
        self._hint_label.setFont(_make_caption_font(self._hint_label))
        self._hint_label.setProperty("afvText", "caption")
        self._hint_label.setWordWrap(True)
        self._hint_label.setTextFormat(Qt.TextFormat.PlainText)
        configure_accessible(
            self._hint_label,
            "Shortcut hint",
            "Double-click a row or press Enter to apply the highlighted release edition.",
        )
        self._hint_label.setVisible(bool(self._editions))
        if self._editions:
            buttons.addWidget(self._hint_label, stretch=1)
        else:
            buttons.addWidget(self._hint_label)
            buttons.addStretch(1)

        has_editions = bool(self._editions)
        self._use_default_button = QPushButton("Use &default", self)
        self._use_default_button.setObjectName("edition_use_default_button")
        self._use_default_button.setAutoDefault(False)
        self._use_default_button.setDefault(not has_editions)
        self._use_default_button.setToolTip(
            f"Keep the default release edition ({format_edition_details(self._payload.default)})."
        )
        configure_accessible(
            self._use_default_button,
            "Use default edition",
            "Select the recommended default release edition.",
        )
        set_button_role(self._use_default_button, "secondary" if has_editions else "primary")
        self._use_default_button.clicked.connect(self._on_use_default)
        buttons.addWidget(self._use_default_button)

        self._use_selected_button = QPushButton("&Use selected edition", self)
        self._use_selected_button.setObjectName("edition_use_selected_button")
        self._use_selected_button.setAutoDefault(False)
        self._use_selected_button.setDefault(has_editions)
        self._use_selected_button.setEnabled(has_editions)
        self._use_selected_button.setIconSize(_BUTTON_ICON_SIZE)
        self._use_selected_button.setToolTip("Apply the highlighted release edition.")
        configure_accessible(
            self._use_selected_button,
            "Use selected edition",
            "Apply the currently highlighted release edition.",
        )
        set_button_role(self._use_selected_button, "primary" if has_editions else "secondary")
        self._use_selected_button.setIcon(dialog_button_icon("check_mark", on_primary=has_editions))
        self._use_selected_button.clicked.connect(self._on_use_selected)
        buttons.addWidget(self._use_selected_button)

        root.addLayout(buttons)

        self._update_detail_readout()

        self.setTabOrder(self._table, self._use_selected_button)
        self.setTabOrder(self._use_selected_button, self._use_default_button)
        if self._editions:
            self._table.setFocus()
        else:
            self._use_default_button.setFocus()

    def _populate_table(self) -> None:
        self._table.setRowCount(len(self._grouped))
        default_row = 0
        align_right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        bold_font = QFont(self._table.font())
        bold_font.setBold(True)
        default_icon = load_icon("status_ok")

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

            row_items = (
                make_table_item(str(row + 1), alignment=align_right),
                make_table_item(ref.date or EM_DASH),
                make_table_item(ref.country or EM_DASH),
                make_table_item(formats_text),
                make_table_item(discs_text, alignment=align_right),
                make_table_item(tracks_text, alignment=align_right),
                make_table_item(details, tooltip=details_tip),
                make_table_item(str(count), alignment=align_right),
                make_table_item(
                    "Default" if is_default else "",
                    tooltip="Recommended default edition" if is_default else "",
                ),
            )
            if is_default:
                row_items[8].setIcon(default_icon)
                for item in row_items:
                    item.setFont(bold_font)
            for col, item in enumerate(row_items):
                self._table.setItem(row, col, item)

        header = self._table.horizontalHeader()
        header.setMinimumSectionSize(32)
        for col in range(len(_EDITION_COLUMNS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
        self._table.resizeColumnsToContents()
        for col in _COMPACT_COLUMNS:
            natural = max(header.sectionSize(col), 32)
            if col == 3:
                natural = min(natural, _MAX_FORMAT_COL_WIDTH)
            elif col == 8:
                natural += 12
            header.resizeSection(col, natural)
        self._sync_edition_column_width()

        if self._editions:
            self._table.selectRow(default_row)
            self._table.setCurrentCell(default_row, 0)

    def _sync_edition_column_width(self) -> None:
        """Keep the ``Edition`` column wide enough to stay readable across dialog widths."""
        viewport_w = self._table.viewport().width()
        if viewport_w <= 0:
            viewport_w = max(0, self.width() - 42)
        other_width = sum(self._table.columnWidth(col) for col in _COMPACT_COLUMNS)
        edition_w = max(_MIN_EDITION_COL_WIDTH, viewport_w - other_width)
        self._table.setColumnWidth(_EDITION_COL_INDEX, edition_w)

    def _update_detail_readout(self) -> None:
        """Show the full unelided metadata of the currently selected edition."""
        if not self._grouped:
            self._detail_label.setText("")
            return
        row = self._table.currentRow()
        if not (0 <= row < len(self._grouped)):
            row = 0
        ref, count = self._grouped[row]
        is_default = ref == self._payload.default
        summary = _format_edition_summary(ref, count, is_default=is_default)
        self._detail_label.setText(summary)
        self._detail_label.setToolTip(summary)
        configure_accessible(
            self._detail_label,
            "Selected edition details",
            summary,
        )

    @Slot()
    def _on_selection_changed(self) -> None:
        self._update_detail_readout()

    def showEvent(self, event: QShowEvent) -> None:
        """Size the Edition column to match the live viewport once shown."""
        super().showEvent(event)
        self._sync_edition_column_width()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Reflow the Edition column cleanly while protecting its minimum width."""
        super().resizeEvent(event)
        self._sync_edition_column_width()

    def changeEvent(self, event: QEvent) -> None:
        """Refresh button icon contrast on live theme or palette changes."""
        super().changeEvent(event)
        if event.type() in (
            QEvent.Type.PaletteChange,
            QEvent.Type.ApplicationPaletteChange,
            QEvent.Type.StyleChange,
        ) and hasattr(self, "_use_selected_button"):
            refresh_themed_widgets(self)
            self._use_selected_button.setIcon(
                dialog_button_icon("check_mark", on_primary=bool(self._editions))
            )

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Confirm the highlighted edition on Return/Enter while the table has focus."""
        if (
            watched is self._table
            and event.type() == QEvent.Type.KeyPress
            and isinstance(event, QKeyEvent)
            and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and not (event.modifiers() & _KEY_MODIFIER_MASK)
            and self._editions
        ):
            self._on_use_selected()
            event.accept()
            return True
        return super().eventFilter(watched, event)

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
