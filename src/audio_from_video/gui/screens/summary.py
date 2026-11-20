"""Summary screen displaying outcome counts, failures, unmatched tracks, and actions."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from audio_from_video.gui.controller import ControllerState
from audio_from_video.gui.models.plan_table import format_destination
from audio_from_video.gui.resources import load_icon
from audio_from_video.gui.screens._common import (
    IconStatusBanner,
    StatusLine,
    log_directory,
    make_bold_font,
    make_heading_label,
    open_local_path,
    open_local_path_with_fallback,
    set_accessible,
)
from audio_from_video.models import RunOutcome, RunSummary

if TYPE_CHECKING:
    from audio_from_video.gui.controller import RunController


class SummaryScreen(QWidget):
    """Summary screen widget (listens to ``summary_ready`` and ``error``)."""

    def __init__(
        self,
        controller: RunController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
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

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        set_accessible(scroll, "Summary content", "Scrollable container for the run summary.")

        body = QWidget(scroll)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        self.title_label = make_heading_label("Run Summary", parent=body)
        layout.addWidget(self.title_label)

        self.output_root_label = QLabel("Output folder: —", body)
        self.output_root_label.setWordWrap(True)
        self.output_root_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.output_root_label)

        # Interrupted banner (shown when summary.interrupted is True).
        self.interrupted_banner = IconStatusBanner(body)
        layout.addWidget(self.interrupted_banner)

        # Error panel (shown when controller emits error(title, details)).
        self.error_panel = QFrame(body)
        self.error_panel.setFrameShape(QFrame.Shape.StyledPanel)
        error_layout = QVBoxLayout(self.error_panel)
        error_layout.setContentsMargins(12, 10, 12, 10)
        error_layout.setSpacing(8)
        self.error_title_line = StatusLine("status_error", "Error", parent=self.error_panel)
        error_layout.addWidget(self.error_title_line)

        self.error_details = QPlainTextEdit(self.error_panel)
        self.error_details.setReadOnly(True)
        self.error_details.setMaximumHeight(130)
        set_accessible(
            self.error_details,
            "Error details",
            "Diagnostic details for the error reported during the run.",
        )
        error_layout.addWidget(self.error_details)

        err_actions = QHBoxLayout()
        self.copy_details_button = QPushButton("&Copy details", self.error_panel)
        set_accessible(
            self.copy_details_button,
            "Copy error details",
            "Copy the error title and diagnostic details to the clipboard.",
        )
        self.open_log_button = QPushButton("Open &log folder", self.error_panel)
        self.open_log_button.setIcon(load_icon("folder"))
        set_accessible(
            self.open_log_button,
            "Open log folder",
            "Open the application log directory in the system file manager.",
        )
        err_actions.addWidget(self.copy_details_button)
        err_actions.addWidget(self.open_log_button)
        err_actions.addStretch(1)
        error_layout.addLayout(err_actions)
        self.error_panel.setVisible(False)
        layout.addWidget(self.error_panel)

        # Outcome counts (every RunOutcome plus Cancelled and Not run).
        counts_group = QGroupBox("Outcome Counts", body)
        counts_grid = QGridLayout(counts_group)
        counts_grid.setHorizontalSpacing(20)
        counts_grid.setVerticalSpacing(6)

        self.succeeded_line = StatusLine("status_ok", "Succeeded: 0", parent=counts_group)
        self.skipped_line = StatusLine("status_skip", "Skipped: 0", parent=counts_group)
        self.skipped_no_audio_line = StatusLine(
            "status_skip",
            "Skipped (no audio): 0",
            parent=counts_group,
        )
        self.unmatched_line = StatusLine(
            "status_unmatched",
            "Unmatched: 0",
            parent=counts_group,
        )
        self.failed_line = StatusLine("status_error", "Failed: 0", parent=counts_group)
        self.cancelled_line = StatusLine("status_cancelled", "Cancelled: 0", parent=counts_group)
        self.not_run_line = StatusLine("status_cancelled", "Not run: 0", parent=counts_group)

        self.count_lines: dict[str, StatusLine] = {
            RunOutcome.SUCCEEDED.value: self.succeeded_line,
            RunOutcome.SKIPPED.value: self.skipped_line,
            RunOutcome.SKIPPED_NO_AUDIO.value: self.skipped_no_audio_line,
            RunOutcome.UNMATCHED.value: self.unmatched_line,
            RunOutcome.FAILED.value: self.failed_line,
            "cancelled": self.cancelled_line,
            "not_run": self.not_run_line,
        }

        counts_grid.addWidget(self.succeeded_line, 0, 0)
        counts_grid.addWidget(self.skipped_line, 0, 1)
        counts_grid.addWidget(self.skipped_no_audio_line, 1, 0)
        counts_grid.addWidget(self.unmatched_line, 1, 1)
        counts_grid.addWidget(self.failed_line, 2, 0)
        counts_grid.addWidget(self.cancelled_line, 2, 1)
        counts_grid.addWidget(self.not_run_line, 3, 0)
        layout.addWidget(counts_group)

        # Failures section.
        self.failures_group = QGroupBox("Failures (0)", body)
        failures_layout = QVBoxLayout(self.failures_group)
        self.failures_list = QListWidget(self.failures_group)
        self.failures_list.setAlternatingRowColors(True)
        self.failures_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.failures_list.setMinimumHeight(80)
        set_accessible(
            self.failures_list,
            "Failed tracks list",
            "Tracks that failed during planning or extraction, with error messages.",
        )
        failures_layout.addWidget(self.failures_list)
        layout.addWidget(self.failures_group)

        # Unmatched section.
        self.unmatched_group = QGroupBox("Unmatched Tracks (0)", body)
        unmatched_layout = QVBoxLayout(self.unmatched_group)
        self.unmatched_list = QListWidget(self.unmatched_group)
        self.unmatched_list.setAlternatingRowColors(True)
        self.unmatched_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.unmatched_list.setMinimumHeight(80)
        set_accessible(
            self.unmatched_list,
            "Unmatched tracks list",
            "Extracted tracks saved without a confirmed online match.",
        )
        unmatched_layout.addWidget(self.unmatched_list)
        layout.addWidget(self.unmatched_group)

        # Optional warnings section.
        self.warnings_group = QGroupBox("Warnings (0)", body)
        warnings_layout = QVBoxLayout(self.warnings_group)
        self.warnings_list = QListWidget(self.warnings_group)
        self.warnings_list.setAlternatingRowColors(True)
        self.warnings_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.warnings_list.setMaximumHeight(110)
        set_accessible(
            self.warnings_list,
            "Track warnings list",
            "Warnings recorded while processing tracks in this batch.",
        )
        warnings_layout.addWidget(self.warnings_list)
        self.warnings_group.setVisible(False)
        layout.addWidget(self.warnings_group)

        # Action bar.
        actions_row = QHBoxLayout()
        actions_row.setSpacing(8)
        self.open_output_button = QPushButton("Open &output folder", body)
        self.open_output_button.setIcon(load_icon("folder"))
        self.open_output_button.setEnabled(False)
        set_accessible(
            self.open_output_button,
            "Open output folder",
            "Open the batch output folder in the system file manager.",
        )

        self.open_containing_button = QPushButton("Open containing &folder", body)
        self.open_containing_button.setIcon(load_icon("folder"))
        self.open_containing_button.setEnabled(False)
        set_accessible(
            self.open_containing_button,
            "Open containing folder",
            "Open the folder containing the selected track in the system file manager.",
        )

        self.save_report_button = QPushButton("&Save report…", body)
        self.save_report_button.setEnabled(False)
        set_accessible(
            self.save_report_button,
            "Save JSON report",
            "Export the summary, track results, and warnings as a JSON report file.",
        )

        self.new_batch_button = QPushButton("&New batch", body)
        self.new_batch_button.setFont(make_bold_font(self.new_batch_button))
        self.new_batch_button.setDefault(True)
        set_accessible(
            self.new_batch_button,
            "New batch",
            "Close the finished session and return to the Inputs screen for a new batch.",
        )

        actions_row.addWidget(self.open_output_button)
        actions_row.addWidget(self.open_containing_button)
        actions_row.addWidget(self.save_report_button)
        actions_row.addStretch(1)
        actions_row.addWidget(self.new_batch_button)
        layout.addLayout(actions_row)

        # Inline notice for missing-folder parent fallback or launcher failures.
        self.folder_notice_banner = IconStatusBanner(body)
        layout.addWidget(self.folder_notice_banner)
        layout.addStretch(1)

        scroll.setWidget(body)
        outer_layout.addWidget(scroll)

        # Wire UI and controller signals via bound methods only.
        self.open_output_button.clicked.connect(self._on_open_output_folder)
        self.open_containing_button.clicked.connect(self._on_open_containing_folder)
        self.save_report_button.clicked.connect(self._on_save_report)
        self.new_batch_button.clicked.connect(self._on_new_batch)
        self.copy_details_button.clicked.connect(self._on_copy_error_details)
        self.open_log_button.clicked.connect(self._on_open_log_folder)
        self.failures_list.currentItemChanged.connect(self._on_failure_selected)
        self.unmatched_list.currentItemChanged.connect(self._on_unmatched_selected)

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
        self.folder_notice_banner.clear_message()

        self.output_root_label.setText(f"Output folder: {summary.output_root}")
        self.open_output_button.setEnabled(True)
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

        self.succeeded_line.set_status(
            "status_ok",
            f"Succeeded: {summary.count(RunOutcome.SUCCEEDED)}",
        )
        self.skipped_line.set_status(
            "status_skip",
            f"Skipped: {summary.count(RunOutcome.SKIPPED)}",
        )
        self.skipped_no_audio_line.set_status(
            "status_skip",
            f"Skipped (no audio): {summary.count(RunOutcome.SKIPPED_NO_AUDIO)}",
        )
        self.unmatched_line.set_status(
            "status_unmatched",
            f"Unmatched: {summary.count(RunOutcome.UNMATCHED)}",
        )
        self.failed_line.set_status(
            "status_error",
            f"Failed: {summary.count(RunOutcome.FAILED)}",
        )
        self.cancelled_line.set_status(
            "status_cancelled",
            f"Cancelled: {self._cancelled_count}",
        )
        self.not_run_line.set_status(
            "status_cancelled",
            f"Not run: {self._not_run_count}",
        )

        # Populate failures list.
        self.failures_list.clear()
        failed_results = [r for r in summary.results if r.outcome == RunOutcome.FAILED]
        self.failures_group.setTitle(f"Failures ({len(failed_results)})")
        if failed_results:
            error_icon = load_icon("status_error")
            for res in failed_results:
                name = res.item.media_input.path.name
                err = res.error or res.item.error or "Unknown error"
                entry = QListWidgetItem(f"{name}: {err}", self.failures_list)
                entry.setIcon(error_icon)
                entry.setToolTip(f"{res.item.media_input.path}\n{err}")
                entry.setData(int(Qt.ItemDataRole.UserRole), res.item.media_input.path.parent)
        else:
            empty = QListWidgetItem("No failures.", self.failures_list)
            empty.setFlags(Qt.ItemFlag.NoItemFlags)

        # Populate unmatched tracks list.
        self.unmatched_list.clear()
        unmatched_results = [r for r in summary.results if r.outcome == RunOutcome.UNMATCHED]
        self.unmatched_group.setTitle(f"Unmatched Tracks ({len(unmatched_results)})")
        if unmatched_results:
            unmatched_icon = load_icon("status_unmatched")
            for res in unmatched_results:
                name = res.item.media_input.path.name
                rel = format_destination(res.output_path, summary.output_root)
                entry = QListWidgetItem(f"{name} → {rel}", self.unmatched_list)
                entry.setIcon(unmatched_icon)
                folder = res.output_path.parent if res.output_path else summary.output_root
                tooltip = str(res.output_path) if res.output_path else str(summary.output_root)
                entry.setToolTip(f"Source: {res.item.media_input.path}\nDestination: {tooltip}")
                entry.setData(int(Qt.ItemDataRole.UserRole), folder)
        else:
            empty = QListWidgetItem("No unmatched tracks.", self.unmatched_list)
            empty.setFlags(Qt.ItemFlag.NoItemFlags)

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
        self.warnings_group.setVisible(bool(warning_rows))

    @Slot(object)
    def _on_state_changed(self, state: object) -> None:
        if state == ControllerState.RUNNING:
            self.clear_error()
            self.folder_notice_banner.clear_message()

    @Slot(str, str)
    def _on_error(self, title: str, details: str) -> None:
        heading = title.strip() or "Extraction error"
        self.error_title_line.set_status("status_error", heading)
        self.error_details.setPlainText(details)
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

    def _open_folder_with_feedback(self, folder: Path, *, label: str) -> None:
        opened, target, used_fallback = open_local_path_with_fallback(folder)
        if target is None:
            self.folder_notice_banner.set_message(
                "status_warn",
                f"Could not open the {label} ({folder}): the folder no longer exists, "
                "and no existing parent folder was found.",
                role_label="Folder notice",
            )
            return
        if not opened:
            self.folder_notice_banner.set_message(
                "status_warn",
                f"Could not open the {label} in the system file manager: {target}.",
                role_label="Folder notice",
            )
            return
        if used_fallback:
            self.folder_notice_banner.set_message(
                "status_info",
                f"The {label} ({folder}) no longer exists; opened nearest existing parent "
                f"folder ({target}).",
                role_label="Folder notice",
            )
            return
        self.folder_notice_banner.clear_message()

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
            self.folder_notice_banner.set_message(
                "status_info",
                text,
                role_label="Folder notice",
            )
            return
        self.folder_notice_banner.clear_message()
