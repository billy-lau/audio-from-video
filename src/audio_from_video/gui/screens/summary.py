"""Summary screen displaying outcome counts, failures, unmatched tracks, and actions."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QSignalBlocker, QSize, Qt, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from audio_from_video.gui.controller import ControllerState
from audio_from_video.gui.models.plan_table import format_destination
from audio_from_video.gui.resources import load_icon
from audio_from_video.gui.screens._common import (
    ActionBarFooter,
    IconStatusBanner,
    MiddleElideLabel,
    SectionCard,
    StatusLine,
    ThemedSvgLabel,
    apply_primary_button_role,
    apply_secondary_button_role,
    join_oxford,
    log_directory,
    make_bold_font,
    make_heading_label,
    make_mono_font,
    make_muted_label,
    open_local_path,
    open_local_path_with_fallback,
    set_accessible,
)
from audio_from_video.gui.theme import repolish, set_widget_role
from audio_from_video.models import RunOutcome, RunSummary

if TYPE_CHECKING:
    from audio_from_video.gui.controller import RunController

_MAX_CONTENT_WIDTH = 920
_EMPTY_LIST_HEIGHT = 38
_MAX_LIST_HEIGHT = 176
_MIN_POPULATED_LIST_HEIGHT = 44
_LIST_ROW_ESTIMATE_HEIGHT = 30


class _SummaryListCard(SectionCard):
    """Elevated rounded card wrapping a summary ``QListWidget`` with ``setTitle`` parity."""

    def __init__(
        self,
        title: str,
        *,
        icon_name: str,
        hint: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            title,
            icon_name=icon_name,
            margins=(16, 12, 16, 12),
            spacing=8,
            parent=parent,
        )
        self._hint_label: QLabel | None = None
        if hint:
            self._hint_label = make_muted_label(hint, scale=0.86, word_wrap=False, parent=self)
            self._hint_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._hint_label.setVisible(False)
            self.add_header_accessory(self._hint_label)

    def title(self) -> str:
        """Return the card's current heading text."""
        if self.title_label is None:
            return ""
        return self.title_label.text()

    def setTitle(self, title: str) -> None:
        """Update the card's heading text."""
        if self.title_label is not None:
            self.title_label.setText(title)

    def set_hint_visible(self, visible: bool) -> None:
        """Show or hide the header accessory hint (e.g., double-click instruction)."""
        if self._hint_label is not None and self._hint_label.isHidden() == visible:
            self._hint_label.setVisible(visible)


def _configure_summary_list(widget: QListWidget) -> None:
    """Apply consistent padding, elision, smooth scrolling, and compact idle height."""
    widget.setAlternatingRowColors(True)
    widget.setTextElideMode(Qt.TextElideMode.ElideMiddle)
    widget.setIconSize(QSize(16, 16))
    widget.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    widget.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    _apply_list_height(widget, count=0)


def _apply_list_height(widget: QListWidget, count: int) -> None:
    """Size a summary list widget compactly when empty and generously when populated."""
    if count <= 0:
        height = _EMPTY_LIST_HEIGHT
    else:
        height = min(
            _MAX_LIST_HEIGHT,
            max(_MIN_POPULATED_LIST_HEIGHT, count * _LIST_ROW_ESTIMATE_HEIGHT + 12),
        )
    widget.setMinimumHeight(height)
    widget.setMaximumHeight(height)


def _build_kpi_tile(
    status_line: StatusLine,
    parent: QWidget,
    *,
    object_name: str,
) -> QFrame:
    """Wrap a ``StatusLine`` inside a sleek ``afvSurface='kpi-tile'`` rounded frame."""
    tile = QFrame(parent)
    tile.setObjectName(object_name)
    tile.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    tile.setFrameShape(QFrame.Shape.NoFrame)
    tile.setFrameShadow(QFrame.Shadow.Plain)
    tile.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    tile.setProperty("afvSurface", "kpi-tile")
    tile.setProperty("afvTone", "none")
    tile.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    status_line.text_label.setWordWrap(False)
    status_line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    tile_layout = QHBoxLayout(tile)
    tile_layout.setContentsMargins(10, 9, 10, 9)
    tile_layout.setSpacing(0)
    status_line.setParent(tile)
    tile_layout.addWidget(status_line, stretch=1)
    return tile


def _set_kpi_tile_style(tile: QFrame, line: StatusLine, *, count: int, active_tone: str) -> None:
    """Apply semantic surface tint and crisp weight to a KPI tile when its count is nonzero."""
    target_tone = active_tone if count > 0 else "none"
    set_widget_role(tile, "afvTone", target_tone, include_children=True)
    font = QFont(line.text_label.font())
    font.setWeight(QFont.Weight.DemiBold if count > 0 else QFont.Weight.Normal)
    line.text_label.setFont(font)


class SummaryScreen(QWidget):
    """Summary screen widget (listens to ``summary_ready`` and ``error``)."""

    def __init__(
        self,
        controller: RunController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("summary_screen")
        self._controller = controller
        self._summary: RunSummary | None = None
        self._cancelled_count = 0
        self._not_run_count = 0
        self._selected_folder: Path | None = None

        set_accessible(
            self,
            "Run summary screen",
            "Displays outcome counts, interrupted notice, failures, unmatched tracks, "
            "and actions to open folders, save a JSON report, or start a new batch.",
        )

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setObjectName("summary_scroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        set_accessible(scroll, "Summary content", "Scrollable container for the run summary.")

        scroll_host = QWidget(scroll)
        scroll_host.setObjectName("summary_scroll_host")
        host_layout = QHBoxLayout(scroll_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(0)

        body = QWidget(scroll_host)
        body.setObjectName("summary_body")
        body.setMaximumWidth(_MAX_CONTENT_WIDTH)
        body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(14)

        # Header band with title, outcome badge chip, dynamic subtitle, and output root strip.
        header_col = QVBoxLayout()
        header_col.setContentsMargins(0, 0, 0, 0)
        header_col.setSpacing(6)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(12)

        title_block = QVBoxLayout()
        title_block.setContentsMargins(0, 0, 0, 0)
        title_block.setSpacing(2)
        self.title_label = make_heading_label("Run Summary", parent=body)
        self.title_label.setObjectName("summary_title_label")
        title_block.addWidget(self.title_label)

        self._outcome_subtitle = make_muted_label(
            "Batch outcome totals, output location, and track results.",
            scale=0.92,
            parent=body,
        )
        self._outcome_subtitle.setObjectName("summary_outcome_subtitle")
        title_block.addWidget(self._outcome_subtitle)
        title_row.addLayout(title_block, stretch=1)

        self._outcome_badge = QLabel("Pending", body)
        self._outcome_badge.setObjectName("summary_outcome_badge")
        self._outcome_badge.setTextFormat(Qt.TextFormat.PlainText)
        self._outcome_badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._outcome_badge.setProperty("afvSurface", "chip")
        self._outcome_badge.setProperty("afvTone", "neutral")
        set_accessible(
            self._outcome_badge,
            "Run outcome badge",
            "High-level status summary for the finished batch.",
        )
        title_row.addWidget(self._outcome_badge, alignment=Qt.AlignmentFlag.AlignVCenter)
        header_col.addLayout(title_row)

        output_strip = QFrame(body)
        output_strip.setObjectName("summary_output_strip")
        output_strip.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        output_strip.setFrameShape(QFrame.Shape.NoFrame)
        output_strip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        output_strip.setProperty("afvSurface", "card-inset")
        output_strip_layout = QHBoxLayout(output_strip)
        output_strip_layout.setContentsMargins(12, 7, 12, 7)
        output_strip_layout.setSpacing(8)
        output_icon = ThemedSvgLabel("folder", size=15, parent=output_strip)
        output_strip_layout.addWidget(output_icon, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.output_root_label = MiddleElideLabel("Output folder: —", output_strip)
        self.output_root_label.setObjectName("summary_output_root_label")
        self.output_root_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.output_root_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.output_root_label.setProperty("afvText", "muted")
        set_accessible(
            self.output_root_label,
            "Output folder path",
            "Destination folder where extracted MP3 files were saved.",
        )
        output_strip_layout.addWidget(self.output_root_label, stretch=1)
        header_col.addWidget(output_strip)
        layout.addLayout(header_col)

        # Interrupted banner (shown when summary.interrupted is True).
        self.interrupted_banner = IconStatusBanner(body)
        layout.addWidget(self.interrupted_banner)

        # Error panel (shown when controller emits error(title, details)).
        self.error_panel = QFrame(body)
        self.error_panel.setObjectName("summary_error_panel")
        self.error_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.error_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.error_panel.setFrameShadow(QFrame.Shadow.Plain)
        self.error_panel.setProperty("afvBanner", True)
        self.error_panel.setProperty("afvTone", "error")
        error_layout = QVBoxLayout(self.error_panel)
        error_layout.setContentsMargins(14, 12, 14, 12)
        error_layout.setSpacing(8)

        self.error_title_line = StatusLine(
            "status_error",
            "Error",
            icon_size=18,
            selectable=True,
            parent=self.error_panel,
        )
        self.error_title_line.text_label.setFont(make_bold_font(self.error_title_line.text_label))
        error_layout.addWidget(self.error_title_line)

        self.error_details = QPlainTextEdit(self.error_panel)
        self.error_details.setObjectName("summary_error_details")
        self.error_details.setReadOnly(True)
        self.error_details.setProperty("afvReadonly", True)
        self.error_details.setFont(make_mono_font(self.error_details))
        self.error_details.setMaximumHeight(130)
        set_accessible(
            self.error_details,
            "Error details",
            "Diagnostic details for the error reported during the run.",
        )
        error_layout.addWidget(self.error_details)

        err_actions = QHBoxLayout()
        err_actions.setContentsMargins(0, 0, 0, 0)
        err_actions.setSpacing(8)
        self.copy_details_button = QPushButton("Copy &details", self.error_panel)
        self.copy_details_button.setObjectName("summary_copy_details_button")
        apply_secondary_button_role(self.copy_details_button, icon_name="copy")
        self.copy_details_button.setToolTip(
            "Copy the error title and diagnostic details to the clipboard."
        )
        set_accessible(
            self.copy_details_button,
            "Copy error details",
            "Copy the error title and diagnostic details to the clipboard.",
        )
        self.open_log_button = QPushButton("Open &log folder", self.error_panel)
        self.open_log_button.setObjectName("summary_open_log_button")
        apply_secondary_button_role(self.open_log_button, icon_name="folder")
        self.open_log_button.setToolTip(
            "Open the application log folder in the system file manager."
        )
        set_accessible(
            self.open_log_button,
            "Open log folder",
            "Open the application log directory in the system file manager.",
        )
        self._copy_feedback_label = make_muted_label(
            "",
            scale=0.88,
            word_wrap=False,
            parent=self.error_panel,
        )
        self._copy_feedback_label.setVisible(False)
        err_actions.addWidget(self.copy_details_button)
        err_actions.addWidget(self.open_log_button)
        err_actions.addWidget(self._copy_feedback_label, alignment=Qt.AlignmentFlag.AlignVCenter)
        err_actions.addStretch(1)
        error_layout.addLayout(err_actions)
        repolish(self.error_panel, include_children=True)
        self.error_panel.setVisible(False)
        layout.addWidget(self.error_panel)

        # Outcome counts grid (every RunOutcome plus Cancelled and Not run) inside KPI tiles.
        counts_card = SectionCard(
            "Outcome Counts",
            subtitle="Totals across succeeded, unmatched, skipped, failed, and cancelled tracks.",
            margins=(14, 12, 14, 14),
            spacing=10,
            parent=body,
        )
        counts_card.setObjectName("summary_counts_card")

        self.succeeded_line = StatusLine("status_ok", "Succeeded: 0", parent=counts_card)
        self.unmatched_line = StatusLine("status_unmatched", "Unmatched: 0", parent=counts_card)
        self.skipped_line = StatusLine("status_skip", "Skipped: 0", parent=counts_card)
        self.failed_line = StatusLine("status_error", "Failed: 0", parent=counts_card)
        self.skipped_no_audio_line = StatusLine(
            "status_skip",
            "Skipped (no audio): 0",
            parent=counts_card,
        )
        self.cancelled_line = StatusLine("status_cancelled", "Cancelled: 0", parent=counts_card)
        self.not_run_line = StatusLine("status_cancelled", "Not run: 0", parent=counts_card)

        self.count_lines: dict[str, StatusLine] = {
            RunOutcome.SUCCEEDED.value: self.succeeded_line,
            RunOutcome.SKIPPED.value: self.skipped_line,
            RunOutcome.SKIPPED_NO_AUDIO.value: self.skipped_no_audio_line,
            RunOutcome.UNMATCHED.value: self.unmatched_line,
            RunOutcome.FAILED.value: self.failed_line,
            "cancelled": self.cancelled_line,
            "not_run": self.not_run_line,
        }

        self._kpi_tiles: dict[str, QFrame] = {
            "succeeded": _build_kpi_tile(
                self.succeeded_line, counts_card, object_name="summary_tile_succeeded"
            ),
            "unmatched": _build_kpi_tile(
                self.unmatched_line, counts_card, object_name="summary_tile_unmatched"
            ),
            "skipped": _build_kpi_tile(
                self.skipped_line, counts_card, object_name="summary_tile_skipped"
            ),
            "failed": _build_kpi_tile(
                self.failed_line, counts_card, object_name="summary_tile_failed"
            ),
            "skipped_no_audio": _build_kpi_tile(
                self.skipped_no_audio_line,
                counts_card,
                object_name="summary_tile_skipped_no_audio",
            ),
            "cancelled": _build_kpi_tile(
                self.cancelled_line, counts_card, object_name="summary_tile_cancelled"
            ),
            "not_run": _build_kpi_tile(
                self.not_run_line, counts_card, object_name="summary_tile_not_run"
            ),
        }

        top_tiles_row = QHBoxLayout()
        top_tiles_row.setContentsMargins(0, 0, 0, 0)
        top_tiles_row.setSpacing(8)
        for key in ("succeeded", "unmatched", "skipped", "failed"):
            top_tiles_row.addWidget(self._kpi_tiles[key], stretch=1)
        counts_card.body_layout.addLayout(top_tiles_row)

        bottom_tiles_row = QHBoxLayout()
        bottom_tiles_row.setContentsMargins(0, 0, 0, 0)
        bottom_tiles_row.setSpacing(8)
        for key in ("skipped_no_audio", "cancelled", "not_run"):
            bottom_tiles_row.addWidget(self._kpi_tiles[key], stretch=1)
        counts_card.body_layout.addLayout(bottom_tiles_row)

        layout.addWidget(counts_card)

        # Failures section (compact single-row card when empty, expanding when populated).
        self.failures_group = _SummaryListCard(
            "Failures (0)",
            icon_name="status_error",
            hint="Double-click to open source folder",
            parent=body,
        )
        self.failures_group.setObjectName("summary_failures_card")
        self.failures_list = QListWidget(self.failures_group)
        self.failures_list.setObjectName("summary_failures_list")
        _configure_summary_list(self.failures_list)
        set_accessible(
            self.failures_list,
            "Failed tracks list",
            "Tracks that failed during planning or extraction, with error messages. "
            "Double-click an item to open its source folder.",
        )
        self.failures_group.body_layout.addWidget(self.failures_list)
        layout.addWidget(self.failures_group)

        # Unmatched tracks section.
        self.unmatched_group = _SummaryListCard(
            "Unmatched Tracks (0)",
            icon_name="status_unmatched",
            hint="Double-click to open output folder",
            parent=body,
        )
        self.unmatched_group.setObjectName("summary_unmatched_card")
        self.unmatched_list = QListWidget(self.unmatched_group)
        self.unmatched_list.setObjectName("summary_unmatched_list")
        _configure_summary_list(self.unmatched_list)
        set_accessible(
            self.unmatched_list,
            "Unmatched tracks list",
            "Extracted tracks saved without a confirmed online match. "
            "Double-click an item to open its output folder.",
        )
        self.unmatched_group.body_layout.addWidget(self.unmatched_list)
        layout.addWidget(self.unmatched_group)

        # Optional warnings section (shown only when warnings exist).
        self.warnings_group = _SummaryListCard(
            "Warnings (0)",
            icon_name="status_warn",
            parent=body,
        )
        self.warnings_group.setObjectName("summary_warnings_card")
        self.warnings_list = QListWidget(self.warnings_group)
        self.warnings_list.setObjectName("summary_warnings_list")
        _configure_summary_list(self.warnings_list)
        set_accessible(
            self.warnings_list,
            "Track warnings list",
            "Warnings recorded while processing tracks in this batch.",
        )
        self.warnings_group.body_layout.addWidget(self.warnings_list)
        self.warnings_group.setVisible(False)
        layout.addWidget(self.warnings_group)

        layout.addStretch(1)

        host_layout.addWidget(body)

        scroll.setWidget(scroll_host)
        outer_layout.addWidget(scroll, stretch=1)

        # Inline notice for missing-folder fallback or launcher failures docked right above footer.
        self._notice_slot = QWidget(self)
        self._notice_slot.setObjectName("summary_folder_notice_slot")
        self._notice_slot.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._notice_layout = QVBoxLayout(self._notice_slot)
        self._notice_layout.setContentsMargins(0, 0, 0, 0)
        self._notice_layout.setSpacing(0)
        self.folder_notice_banner = IconStatusBanner(self._notice_slot)
        self._notice_layout.addWidget(self.folder_notice_banner)
        self._notice_slot.setVisible(False)
        outer_layout.addWidget(self._notice_slot)

        # Sticky footer action bar pinned outside QScrollArea so all CTAs remain visible.
        footer = ActionBarFooter(self, margins=(20, 10, 20, 12), spacing=8)
        footer.setObjectName("summary_footer")

        self.open_output_button = QPushButton("Open &output", footer)
        self.open_output_button.setObjectName("summary_open_output_button")
        apply_secondary_button_role(self.open_output_button, icon_name="folder")
        self.open_output_button.setEnabled(False)
        self.open_output_button.setToolTip(
            "Open the batch output folder in the system file manager."
        )
        set_accessible(
            self.open_output_button,
            "Open output folder",
            "Open the batch output folder in the system file manager.",
        )

        self.open_containing_button = QPushButton("Track &folder", footer)
        self.open_containing_button.setObjectName("summary_open_containing_button")
        apply_secondary_button_role(self.open_containing_button, icon_name="folder")
        self.open_containing_button.setEnabled(False)
        self.open_containing_button.setToolTip(
            "Open the folder containing the selected track in the system file manager."
        )
        set_accessible(
            self.open_containing_button,
            "Open containing folder",
            "Open the folder containing the selected track in the system file manager.",
        )

        self.save_report_button = QPushButton("&Save report…", footer)
        self.save_report_button.setObjectName("summary_save_report_button")
        apply_secondary_button_role(self.save_report_button, icon_name="save")
        self.save_report_button.setEnabled(False)
        self.save_report_button.setToolTip(
            "Export the summary, track results, and warnings as a JSON report file."
        )
        set_accessible(
            self.save_report_button,
            "Save JSON report",
            "Export the summary, track results, and warnings as a JSON report file.",
        )

        self.new_batch_button = QPushButton("&New batch", footer)
        self.new_batch_button.setObjectName("summary_new_batch_button")
        apply_primary_button_role(self.new_batch_button, icon_name="refresh")
        self.new_batch_button.setDefault(True)
        self.new_batch_button.setToolTip(
            "Close the finished session and return to the Inputs screen for a new batch."
        )
        set_accessible(
            self.new_batch_button,
            "New batch",
            "Close the finished session and return to the Inputs screen for a new batch.",
        )

        footer.content_layout.addWidget(self.open_output_button)
        footer.content_layout.addWidget(self.open_containing_button)
        footer.content_layout.addWidget(self.save_report_button)
        footer.content_layout.addStretch(1)
        footer.content_layout.addWidget(self.new_batch_button)
        outer_layout.addWidget(footer)

        # Wire UI and controller signals via bound methods only.
        self.open_output_button.clicked.connect(self._on_open_output_folder)
        self.open_containing_button.clicked.connect(self._on_open_containing_folder)
        self.save_report_button.clicked.connect(self._on_save_report)
        self.new_batch_button.clicked.connect(self._on_new_batch)
        self.copy_details_button.clicked.connect(self._on_copy_error_details)
        self.open_log_button.clicked.connect(self._on_open_log_folder)
        self.failures_list.currentItemChanged.connect(self._on_failure_selected)
        self.unmatched_list.currentItemChanged.connect(self._on_unmatched_selected)
        self.failures_list.itemDoubleClicked.connect(self._on_failure_activated)
        self.unmatched_list.itemDoubleClicked.connect(self._on_unmatched_activated)

        self._controller.summary_ready.connect(self._on_summary_ready)
        self._controller.error.connect(self._on_error)
        self._controller.state_changed.connect(self._on_state_changed)

        tab_chain: tuple[QWidget, ...] = (
            self.error_details,
            self.copy_details_button,
            self.open_log_button,
            self.failures_list,
            self.unmatched_list,
            self.warnings_list,
            self.open_output_button,
            self.open_containing_button,
            self.save_report_button,
            self.new_batch_button,
        )
        for prev_widget, next_widget in itertools.pairwise(tab_chain):
            self.setTabOrder(prev_widget, next_widget)

    @property
    def summary(self) -> RunSummary | None:
        """Return the latest ``RunSummary`` shown on this screen, if any."""
        return self._summary

    def clear_error(self) -> None:
        """Hide and clear the error details panel."""
        self.error_details.clear()
        self._copy_feedback_label.setText("")
        self._copy_feedback_label.setVisible(False)
        self.error_panel.setVisible(False)

    def containing_folder_for_selection(self) -> Path | None:
        """Return the folder to open for ``Open containing folder``."""
        if self._selected_folder is not None:
            return self._selected_folder
        if self._summary is None:
            return None
        for res in self._summary.results:
            if res.outcome == RunOutcome.UNMATCHED and res.output_path is not None:
                return res.output_path.parent
        for res in self._summary.results:
            if res.output_path is not None:
                return res.output_path.parent
        return self._summary.output_root

    def _set_folder_notice(self, icon_name: str, text: str) -> None:
        self._notice_layout.setContentsMargins(20, 6, 20, 8)
        self.folder_notice_banner.set_message(icon_name, text, role_label="Folder notice")
        if self._notice_slot.isHidden():
            self._notice_slot.setVisible(True)

    def _clear_folder_notice(self) -> None:
        self.folder_notice_banner.clear_message()
        self._notice_layout.setContentsMargins(0, 0, 0, 0)
        if not self._notice_slot.isHidden():
            self._notice_slot.setVisible(False)

    def _update_containing_button_tooltip(self, *, kind: str = "default") -> None:
        folder = self.containing_folder_for_selection()
        if folder is None:
            self.open_containing_button.setToolTip(
                "Open the folder containing the selected track in the system file manager."
            )
            return
        if kind == "source":
            self.open_containing_button.setToolTip(
                f"Open source video folder for the selected failed track:\n{folder}"
            )
        elif kind == "output":
            self.open_containing_button.setToolTip(
                f"Open output MP3 folder for the selected unmatched track:\n{folder}"
            )
        else:
            self.open_containing_button.setToolTip(
                f"Open the containing folder in the system file manager:\n{folder}"
            )

    def _update_outcome_header(
        self,
        *,
        interrupted: bool,
        succeeded: int,
        unmatched: int,
        skipped_total: int,
        failed: int,
        cancelled: int,
        not_run: int,
        warning_count: int,
    ) -> None:
        if interrupted:
            badge_text = "Interrupted"
            badge_tone = "warn"
            parts: list[str] = []
            if succeeded:
                parts.append(f"{succeeded} succeeded")
            if unmatched:
                parts.append(f"{unmatched} unmatched")
            if failed:
                parts.append(f"{failed} failed")
            if cancelled:
                parts.append(f"{cancelled} cancelled")
            if not_run:
                parts.append(f"{not_run} not run")
            detail = join_oxford(parts)
            subtitle = (
                f"Batch stopped early ({detail})."
                if detail
                else "Batch stopped before any track finished."
            )
        elif failed > 0:
            noun = "failure" if failed == 1 else "failures"
            badge_text = f"{failed} {noun}"
            badge_tone = "error"
            notes: list[str] = []
            if succeeded:
                notes.append(f"{succeeded} succeeded")
            if unmatched:
                notes.append(f"{unmatched} unmatched")
            if skipped_total:
                notes.append(f"{skipped_total} skipped")
            notes.append(f"{failed} failed")
            subtitle = f"Completed with issues: {join_oxford(notes)}."
        elif unmatched > 0:
            badge_text = "Completed with unmatched"
            badge_tone = "unmatched"
            u_noun = "track" if unmatched == 1 else "tracks"
            subtitle = (
                f"Saved {succeeded + unmatched} tracks ({unmatched} unmatched {u_noun} "
                "without an online match)."
            )
        elif warning_count > 0:
            badge_text = "Completed with warnings"
            badge_tone = "warn"
            w_noun = "warning" if warning_count == 1 else "warnings"
            subtitle = f"Finished with {warning_count} {w_noun} recorded."
        elif succeeded > 0:
            badge_text = "Completed"
            badge_tone = "ok"
            t_noun = "track" if succeeded == 1 else "tracks"
            subtitle = f"Extracted {succeeded} {t_noun} cleanly."
        elif skipped_total > 0:
            badge_text = "Nothing extracted"
            badge_tone = "neutral"
            subtitle = f"All {skipped_total} tracks were skipped."
        else:
            badge_text = "Complete"
            badge_tone = "neutral"
            subtitle = "No tracks were processed in this batch."

        self._outcome_badge.setText(badge_text)
        set_widget_role(self._outcome_badge, "afvTone", badge_tone)
        self._outcome_subtitle.setText(subtitle)

    @Slot(object, int, int)
    def _on_summary_ready(
        self,
        summary: RunSummary,
        cancelled_count: int,
        not_run_count: int,
    ) -> None:
        self._summary = summary
        self._cancelled_count = max(0, int(cancelled_count))
        self._not_run_count = max(0, int(not_run_count))
        self._selected_folder = None
        self._clear_folder_notice()

        self.output_root_label.setText(f"Output folder: {summary.output_root}")
        self.output_root_label.set_full_tooltip(str(summary.output_root))
        self.open_output_button.setEnabled(True)
        self.open_output_button.setToolTip(
            f"Open the batch output folder in the system file manager:\n{summary.output_root}"
        )
        self.open_containing_button.setEnabled(True)
        self.save_report_button.setEnabled(True)

        if summary.interrupted:
            self.interrupted_banner.set_message(
                "status_warn",
                "Interrupted — active FFmpeg processes were stopped, and temporary partial files "
                "were removed.",
                role_label="Interrupted notice",
            )
        else:
            self.interrupted_banner.clear_message()

        succeeded = summary.count(RunOutcome.SUCCEEDED)
        skipped = summary.count(RunOutcome.SKIPPED)
        skipped_no_audio = summary.count(RunOutcome.SKIPPED_NO_AUDIO)
        unmatched = summary.count(RunOutcome.UNMATCHED)
        failed = summary.count(RunOutcome.FAILED)

        self.succeeded_line.set_status("status_ok", f"Succeeded: {succeeded}")
        self.skipped_line.set_status("status_skip", f"Skipped: {skipped}")
        self.skipped_no_audio_line.set_status(
            "status_skip",
            f"Skipped (no audio): {skipped_no_audio}",
        )
        self.unmatched_line.set_status("status_unmatched", f"Unmatched: {unmatched}")
        self.failed_line.set_status("status_error", f"Failed: {failed}")
        self.cancelled_line.set_status("status_cancelled", f"Cancelled: {self._cancelled_count}")
        self.not_run_line.set_status("status_cancelled", f"Not run: {self._not_run_count}")

        _set_kpi_tile_style(
            self._kpi_tiles["succeeded"],
            self.succeeded_line,
            count=succeeded,
            active_tone="ok",
        )
        _set_kpi_tile_style(
            self._kpi_tiles["unmatched"],
            self.unmatched_line,
            count=unmatched,
            active_tone="unmatched",
        )
        _set_kpi_tile_style(
            self._kpi_tiles["skipped"],
            self.skipped_line,
            count=skipped,
            active_tone="neutral",
        )
        _set_kpi_tile_style(
            self._kpi_tiles["failed"],
            self.failed_line,
            count=failed,
            active_tone="error",
        )
        _set_kpi_tile_style(
            self._kpi_tiles["skipped_no_audio"],
            self.skipped_no_audio_line,
            count=skipped_no_audio,
            active_tone="neutral",
        )
        _set_kpi_tile_style(
            self._kpi_tiles["cancelled"],
            self.cancelled_line,
            count=self._cancelled_count,
            active_tone="warn",
        )
        _set_kpi_tile_style(
            self._kpi_tiles["not_run"],
            self.not_run_line,
            count=self._not_run_count,
            active_tone="warn",
        )

        # Populate failures list.
        self.failures_list.clear()
        failed_results = [r for r in summary.results if r.outcome == RunOutcome.FAILED]
        self.failures_group.setTitle(f"Failures ({len(failed_results)})")
        self.failures_group.set_hint_visible(bool(failed_results))
        if failed_results:
            error_icon = load_icon("status_error")
            for res in failed_results:
                name = res.item.media_input.path.name
                err = res.error or res.item.error or "Unknown error"
                entry = QListWidgetItem(f"{name}: {err}", self.failures_list)
                entry.setIcon(error_icon)
                entry.setToolTip(
                    f"Source: {res.item.media_input.path}\nError: {err}\n"
                    "Double-click to open source folder."
                )
                entry.setData(int(Qt.ItemDataRole.UserRole), res.item.media_input.path.parent)
            _apply_list_height(self.failures_list, len(failed_results))
        else:
            empty = QListWidgetItem("No failures.", self.failures_list)
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            _apply_list_height(self.failures_list, 0)

        # Populate unmatched tracks list.
        self.unmatched_list.clear()
        unmatched_results = [r for r in summary.results if r.outcome == RunOutcome.UNMATCHED]
        self.unmatched_group.setTitle(f"Unmatched Tracks ({len(unmatched_results)})")
        self.unmatched_group.set_hint_visible(bool(unmatched_results))
        if unmatched_results:
            unmatched_icon = load_icon("status_unmatched")
            for res in unmatched_results:
                name = res.item.media_input.path.name
                rel = format_destination(res.output_path, summary.output_root)
                entry = QListWidgetItem(f"{name} → {rel}", self.unmatched_list)
                entry.setIcon(unmatched_icon)
                folder = res.output_path.parent if res.output_path else summary.output_root
                tooltip = str(res.output_path) if res.output_path else str(summary.output_root)
                entry.setToolTip(
                    f"Source: {res.item.media_input.path}\nDestination: {tooltip}\n"
                    "Double-click to open output folder."
                )
                entry.setData(int(Qt.ItemDataRole.UserRole), folder)
            _apply_list_height(self.unmatched_list, len(unmatched_results))
        else:
            empty = QListWidgetItem("No unmatched tracks.", self.unmatched_list)
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            _apply_list_height(self.unmatched_list, 0)

        # Populate warnings list (shown only when warnings exist).
        self.warnings_list.clear()
        warning_rows = [
            (res.item.media_input.path.name, warn)
            for res in summary.results
            for warn in res.warnings
        ]
        self.warnings_group.setTitle(f"Warnings ({len(warning_rows)})")
        if warning_rows:
            warn_icon = load_icon("status_warn")
            for name, warn in warning_rows:
                entry = QListWidgetItem(f"{name}: {warn}", self.warnings_list)
                entry.setIcon(warn_icon)
                entry.setToolTip(f"{name}: {warn}")
            _apply_list_height(self.warnings_list, len(warning_rows))
        self.warnings_group.setVisible(bool(warning_rows))

        self._update_outcome_header(
            interrupted=summary.interrupted,
            succeeded=succeeded,
            unmatched=unmatched,
            skipped_total=skipped + skipped_no_audio,
            failed=failed,
            cancelled=self._cancelled_count,
            not_run=self._not_run_count,
            warning_count=len(warning_rows),
        )
        self._update_containing_button_tooltip()

    @Slot(object)
    def _on_state_changed(self, state: object) -> None:
        if state == ControllerState.RUNNING:
            self.clear_error()
            self._clear_folder_notice()

    @Slot(str, str)
    def _on_error(self, title: str, details: str) -> None:
        heading = title.strip() or "Extraction error"
        self.error_title_line.set_status("status_error", heading)
        self.error_details.setPlainText(details)
        self._copy_feedback_label.setText("")
        self._copy_feedback_label.setVisible(False)
        self.error_panel.setVisible(True)

    @Slot(QListWidgetItem, QListWidgetItem)
    def _on_failure_selected(
        self,
        current: QListWidgetItem | None,
        previous: QListWidgetItem | None,
    ) -> None:
        _ = previous
        if current is None:
            return
        folder = current.data(int(Qt.ItemDataRole.UserRole))
        if isinstance(folder, Path):
            self._selected_folder = folder
            with QSignalBlocker(self.unmatched_list):
                self.unmatched_list.clearSelection()
            self._update_containing_button_tooltip(kind="source")

    @Slot(QListWidgetItem, QListWidgetItem)
    def _on_unmatched_selected(
        self,
        current: QListWidgetItem | None,
        previous: QListWidgetItem | None,
    ) -> None:
        _ = previous
        if current is None:
            return
        folder = current.data(int(Qt.ItemDataRole.UserRole))
        if isinstance(folder, Path):
            self._selected_folder = folder
            with QSignalBlocker(self.failures_list):
                self.failures_list.clearSelection()
            self._update_containing_button_tooltip(kind="output")

    @Slot(QListWidgetItem)
    def _on_failure_activated(self, item: QListWidgetItem) -> None:
        folder = item.data(int(Qt.ItemDataRole.UserRole))
        if isinstance(folder, Path):
            self._selected_folder = folder
            self._update_containing_button_tooltip(kind="source")
            self._open_folder_with_feedback(folder, label="source folder")

    @Slot(QListWidgetItem)
    def _on_unmatched_activated(self, item: QListWidgetItem) -> None:
        folder = item.data(int(Qt.ItemDataRole.UserRole))
        if isinstance(folder, Path):
            self._selected_folder = folder
            self._update_containing_button_tooltip(kind="output")
            self._open_folder_with_feedback(folder, label="output folder")

    def _open_folder_with_feedback(self, folder: Path, *, label: str) -> None:
        opened, target, used_fallback = open_local_path_with_fallback(folder)
        if target is None:
            self._set_folder_notice(
                "status_warn",
                f"Could not open the {label} ({folder}): the folder no longer exists, "
                "and no existing parent folder was found.",
            )
            return
        if not opened:
            self._set_folder_notice(
                "status_warn",
                f"Could not open the {label} in the system file manager: {target}.",
            )
            return
        if used_fallback:
            self._set_folder_notice(
                "status_info",
                f"The {label} ({folder}) no longer exists; opened nearest existing parent "
                f"folder ({target}).",
            )
            return
        self._clear_folder_notice()

    @Slot()
    def _on_open_output_folder(self) -> None:
        if self._summary is None:
            return
        self._open_folder_with_feedback(self._summary.output_root, label="output folder")

    @Slot()
    def _on_open_containing_folder(self) -> None:
        folder = self.containing_folder_for_selection()
        if folder is None:
            return
        self._open_folder_with_feedback(folder, label="containing folder")

    @Slot()
    def _on_save_report(self) -> None:
        if self._summary is None:
            return
        default_path = str(self._summary.output_root / "afv-report.json")
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save JSON report",
            default_path,
            "JSON files (*.json);;All files (*)",
        )
        if not selected:
            return
        self._controller.save_report(Path(selected))

    @Slot()
    def _on_new_batch(self) -> None:
        self._controller.new_batch()

    @Slot()
    def _on_copy_error_details(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return
        title = self.error_title_line.text
        details = self.error_details.toPlainText()
        payload = f"{title}\n\n{details}".strip() if details else title
        clipboard.setText(payload)
        self._copy_feedback_label.setText("Copied to clipboard.")
        self._copy_feedback_label.setVisible(True)

    @Slot()
    def _on_open_log_folder(self) -> None:
        log_dir = log_directory()
        if not open_local_path(log_dir):
            try:
                exists = log_dir.expanduser().exists()
            except OSError:
                exists = False
            text = (
                f"Could not open the log folder in the system file manager: {log_dir}."
                if exists
                else f"The log folder ({log_dir}) does not exist yet."
            )
            self._set_folder_notice("status_info", text)
            return
        self._clear_folder_notice()
