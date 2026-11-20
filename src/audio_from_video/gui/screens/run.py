"""Run progress screen displaying extraction progress, row states, and cancel."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from PySide6.QtCore import QModelIndex, QPersistentModelIndex, Qt, QTimer, Slot
from PySide6.QtGui import QHideEvent, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from audio_from_video.events import NOT_RUN_MESSAGE, TrackStage
from audio_from_video.gui.controller import ControllerState
from audio_from_video.gui.models.plan_table import (
    COL_ACTION,
    COL_DESTINATION,
    COL_MATCH,
    COL_SOURCE,
    COL_STATUS,
    COL_WARNINGS,
    PlanTableModel,
)
from audio_from_video.gui.screens._common import (
    IconStatusBanner,
    StatusLine,
    make_heading_label,
    set_accessible,
)
from audio_from_video.models import RunOutcome

if TYPE_CHECKING:
    from audio_from_video.gui.controller import RunController

CANCELLING_MESSAGE = "Cancelling… stopping FFmpeg and removing partial files"

_REFRESH_COALESCE_MS = 20

_BUCKET_QUEUED = 0
_BUCKET_EXTRACTING = 1
_BUCKET_DONE = 2
_BUCKET_SKIPPED = 3
_BUCKET_FAILED = 4
_BUCKET_CANCELLED = 5
_BUCKET_NOT_RUN = 6
_BUCKET_COUNT = 7


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
        self._cancelling = False
        self._last_announced_bucket = -1
        self._last_completed = -1
        self._row_buckets: list[int] = []
        self._counts: list[int] = [0] * _BUCKET_COUNT
        self._failed_messages: dict[int, str] = {}
        self._stale_while_hidden = False
        self._dirty = False
        self._last_progress_max = -1
        self._last_progress_value = -1
        self._last_counts_text = ""

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(_REFRESH_COALESCE_MS)
        self._refresh_timer.timeout.connect(self._on_refresh_timeout)

        set_accessible(
            self,
            "Run progress screen",
            "Displays live extraction progress and status for each track in the batch.",
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        self.title_label = make_heading_label("Extracting Audio", parent=self)
        layout.addWidget(self.title_label)

        self.status_line = StatusLine("status_working", "Preparing extraction…", parent=self)
        layout.addWidget(self.status_line)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        set_accessible(
            self.progress_bar,
            "Overall extraction progress",
            "0 of 0 tracks complete.",
        )
        layout.addWidget(self.progress_bar)

        self.counts_label = QLabel(
            "Queued: 0 · Extracting: 0 · Done: 0 · Skipped: 0 · Failed: 0 · Cancelled: 0",
            self,
        )
        self.counts_label.setWordWrap(True)
        self.counts_label.setTextFormat(Qt.TextFormat.PlainText)
        set_accessible(
            self.counts_label,
            "Track status counts",
            "Live counts of queued, extracting, completed, skipped, failed, and cancelled tracks.",
        )
        layout.addWidget(self.counts_label)

        self.error_banner = IconStatusBanner(self)
        layout.addWidget(self.error_banner)

        self.table_view = QTableView(self)
        self.table_view.setModel(self._model)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setWordWrap(False)
        self.table_view.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.table_view.setShowGrid(False)
        self.table_view.verticalHeader().setVisible(False)
        header = self.table_view.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(COL_STATUS, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_SOURCE, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_DESTINATION, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_ACTION, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_MATCH, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_WARNINGS, QHeaderView.ResizeMode.Stretch)
        header.resizeSection(COL_STATUS, 140)
        header.resizeSection(COL_SOURCE, 180)
        header.resizeSection(COL_DESTINATION, 220)
        header.resizeSection(COL_ACTION, 165)
        header.resizeSection(COL_MATCH, 190)
        set_accessible(
            self.table_view,
            "Extraction progress table",
            "Live per-track status during Stage 2 extraction, including failures and errors.",
        )
        layout.addWidget(self.table_view, stretch=1)

        footer = QHBoxLayout()
        footer.addStretch(1)
        self.cancel_button = QPushButton("&Cancel", self)
        set_accessible(
            self.cancel_button,
            "Cancel extraction",
            "Stop active FFmpeg processes, remove partial files, and move to Summary.",
        )
        footer.addWidget(self.cancel_button)
        layout.addLayout(footer)

        # Wire model, button, and controller signals using bound methods only.
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self._model.modelReset.connect(self._on_model_changed)
        self._model.rowsInserted.connect(self._on_model_rows_changed)
        self._model.rowsRemoved.connect(self._on_model_rows_changed)
        self._model.dataChanged.connect(self._on_model_data_changed)
        self._controller.state_changed.connect(self._on_state_changed)

        self.setTabOrder(self.table_view, self.cancel_button)

        self._refresh_progress()

    @property
    def is_cancelling(self) -> bool:
        """Return True while cancellation has been confirmed and is in progress."""
        return self._cancelling

    def set_cancelling(self, cancelling: bool) -> None:
        """Update the screen's cancelling indicator and button enablement."""
        self._cancelling = cancelling
        self.cancel_button.setEnabled(not cancelling)
        if cancelling:
            self.status_line.set_status("status_warn", CANCELLING_MESSAGE)
        else:
            self._refresh_progress()

    def confirm_cancel(self) -> bool:
        """Ask the user to confirm stopping the active extraction run."""
        answer = QMessageBox.question(
            self,
            "Cancel extraction?",
            "Stop the active extraction? Running FFmpeg processes will be stopped, "
            "and partial files and newly created empty folders will be removed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _sync_output_root(self) -> None:
        ctrl_root = self._controller.output_root
        if ctrl_root is not None and self._model.output_root != ctrl_root:
            self._model.set_output_root(ctrl_root)

    def _classify_row(self, row: int) -> tuple[int, str | None]:
        """Classify ``row`` into a status bucket and optional failure snippet."""
        stage = self._model.stage_at(row)
        if stage == TrackStage.EXTRACTING:
            return (_BUCKET_EXTRACTING, None)
        if stage == TrackStage.FINISHED:
            outcome = self._model.outcome_at(row)
            if outcome == RunOutcome.FAILED:
                message = self._model.message_at(row)
                item = self._model.item_at(row)
                err = message or (item.error if item is not None else None)
                if err:
                    name = item.media_input.path.name if item is not None else f"row {row + 1}"
                    return (_BUCKET_FAILED, f"{name}: {err}")
                return (_BUCKET_FAILED, None)
            if outcome in (RunOutcome.SKIPPED, RunOutcome.SKIPPED_NO_AUDIO):
                return (_BUCKET_SKIPPED, None)
            return (_BUCKET_DONE, None)
        if stage == TrackStage.CANCELLED:
            if self._model.message_at(row) == NOT_RUN_MESSAGE:
                return (_BUCKET_NOT_RUN, None)
            return (_BUCKET_CANCELLED, None)
        return (_BUCKET_QUEUED, None)

    def _rebuild_counts(self) -> None:
        total = self._model.rowCount()
        buckets: list[int] = [0] * total
        counts: list[int] = [0] * _BUCKET_COUNT
        failures: dict[int, str] = {}
        for row in range(total):
            bucket, err = self._classify_row(row)
            buckets[row] = bucket
            counts[bucket] += 1
            if err is not None:
                failures[row] = err
        self._row_buckets = buckets
        self._counts = counts
        self._failed_messages = failures

    def _apply_progress_widgets(self) -> None:
        counts_arr = self._counts
        total = len(self._row_buckets)
        queued = counts_arr[_BUCKET_QUEUED]
        extracting = counts_arr[_BUCKET_EXTRACTING]
        done = counts_arr[_BUCKET_DONE]
        skipped = counts_arr[_BUCKET_SKIPPED]
        failed = counts_arr[_BUCKET_FAILED]
        cancelled = counts_arr[_BUCKET_CANCELLED]
        not_run = counts_arr[_BUCKET_NOT_RUN]

        completed = done + skipped + failed + cancelled + not_run
        if total > 0:
            target_value = min(completed, total)
            if self._last_progress_max != total:
                self._last_progress_max = total
                self.progress_bar.setRange(0, total)
            if self._last_progress_value != target_value:
                self._last_progress_value = target_value
                self.progress_bar.setValue(target_value)
            pct = round((completed / total) * 100)
            bucket = pct // 10
            if bucket != self._last_announced_bucket or completed != self._last_completed:
                self._last_announced_bucket = bucket
                self._last_completed = completed
                self.progress_bar.setAccessibleDescription(
                    f"Extraction progress: {completed} of {total} tracks finished ({pct}%)."
                )
        else:
            if self._last_progress_max != 1:
                self._last_progress_max = 1
                self.progress_bar.setRange(0, 1)
            if self._last_progress_value != 0:
                self._last_progress_value = 0
                self.progress_bar.setValue(0)
            self.progress_bar.setAccessibleDescription("0 of 0 tracks complete.")

        parts = [
            f"Queued: {queued}",
            f"Extracting: {extracting}",
            f"Done: {done}",
            f"Skipped: {skipped}",
            f"Failed: {failed}",
            f"Cancelled: {cancelled}",
        ]
        if not_run > 0:
            parts.append(f"Not run: {not_run}")
        counts_text = " · ".join(parts)
        if counts_text != self._last_counts_text:
            self._last_counts_text = counts_text
            self.counts_label.setText(counts_text)

        if self._failed_messages:
            sample = list(itertools.islice(self._failed_messages.values(), 3))
            self.error_banner.set_message(
                "status_error",
                f"Failed ({failed}): " + "; ".join(sample),
                role_label="Extraction error",
            )
        else:
            self.error_banner.clear_message()

        if self._cancelling:
            self.status_line.set_status("status_warn", CANCELLING_MESSAGE)
            return

        if total == 0:
            self.status_line.set_status("status_pending", "Waiting for tracks…")
        elif completed >= total:
            if failed > 0:
                self.status_line.set_status(
                    "status_warn",
                    f"Finished {completed} of {total} tracks ({failed} failed).",
                )
            elif cancelled > 0:
                self.status_line.set_status(
                    "status_cancelled",
                    f"Stopped after {completed - cancelled} of {total} tracks "
                    f"({cancelled} cancelled).",
                )
            else:
                self.status_line.set_status(
                    "status_ok",
                    f"Finished {completed} of {total} tracks.",
                )
        elif extracting > 0:
            self.status_line.set_status(
                "status_working",
                f"Extracting ({completed} of {total} finished, {extracting} active)…",
            )
        else:
            self.status_line.set_status(
                "status_pending",
                f"Queued ({completed} of {total} finished)…",
            )

    def _schedule_refresh(self) -> None:
        if not self.isVisible():
            return
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def flush_updates(self) -> None:
        """Apply any pending deferred progress counts and banner updates immediately."""
        if self._refresh_timer.isActive():
            self._refresh_timer.stop()
        self._sync_output_root()
        if self._stale_while_hidden:
            self._rebuild_counts()
            self._stale_while_hidden = False
            self._dirty = True
        if self._dirty:
            self._apply_progress_widgets()
            self._dirty = False

    def _refresh_progress(self) -> None:
        if self._refresh_timer.isActive():
            self._refresh_timer.stop()
        self._sync_output_root()
        self._rebuild_counts()
        self._stale_while_hidden = False
        self._dirty = False
        self._apply_progress_widgets()

    def showEvent(self, event: QShowEvent) -> None:
        """Flush any progress updates deferred while the screen was hidden."""
        super().showEvent(event)
        if self._stale_while_hidden or self._dirty:
            self.flush_updates()

    def hideEvent(self, event: QHideEvent) -> None:
        """Pause deferred refresh timers while the screen is hidden."""
        super().hideEvent(event)
        if self._refresh_timer.isActive():
            self._refresh_timer.stop()

    @Slot()
    def _on_cancel_clicked(self) -> None:
        if self._cancelling:
            return
        if not self.confirm_cancel():
            return
        self.set_cancelling(True)
        self._controller.cancel()

    @Slot(object)
    def _on_state_changed(self, state: object) -> None:
        if state == ControllerState.CANCELLING:
            self.set_cancelling(True)
        elif state == ControllerState.RUNNING:
            self._cancelling = False
            self.cancel_button.setEnabled(True)
            self._refresh_progress()

    @Slot()
    def _on_model_changed(self) -> None:
        self._last_announced_bucket = -1
        self._last_completed = -1
        self._refresh_progress()

    @Slot(QModelIndex, int, int)
    def _on_model_rows_changed(
        self,
        parent: QModelIndex | QPersistentModelIndex,
        first: int,
        last: int,
    ) -> None:
        _ = parent, first, last
        self._refresh_progress()

    @Slot()
    def _on_refresh_timeout(self) -> None:
        if not self.isVisible():
            return
        self.flush_updates()

    @Slot(QModelIndex, QModelIndex, list)
    def _on_model_data_changed(
        self,
        top_left: QModelIndex | QPersistentModelIndex,
        bottom_right: QModelIndex | QPersistentModelIndex,
        roles: list[int] | None = None,
    ) -> None:
        _ = roles
        if not self.isVisible():
            self._stale_while_hidden = True
            return
        row_count = self._model.rowCount()
        if row_count == 0:
            self._refresh_progress()
            return
        if (
            not top_left.isValid()
            or not bottom_right.isValid()
            or self._stale_while_hidden
            or len(self._row_buckets) != row_count
        ):
            self._stale_while_hidden = True
            self._schedule_refresh()
            return
        if top_left.column() == COL_DESTINATION and bottom_right.column() == COL_DESTINATION:
            return

        first = max(0, top_left.row())
        last = min(row_count - 1, bottom_right.row())
        for row in range(first, last + 1):
            old_bucket = self._row_buckets[row]
            new_bucket, err = self._classify_row(row)
            if new_bucket != old_bucket:
                self._counts[old_bucket] -= 1
                self._counts[new_bucket] += 1
                self._row_buckets[row] = new_bucket
                self._dirty = True
            if err is not None:
                if self._failed_messages.get(row) != err:
                    self._failed_messages[row] = err
                    self._dirty = True
            elif row in self._failed_messages:
                del self._failed_messages[row]
                self._dirty = True

        if self._dirty:
            self._schedule_refresh()
