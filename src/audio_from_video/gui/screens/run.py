"""Run progress screen displaying extraction progress, row states, and cancel."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QModelIndex, QObject, QPersistentModelIndex, Qt, QTimer, Slot
from PySide6.QtGui import QFont, QHideEvent, QResizeEvent, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from audio_from_video.events import NOT_RUN_MESSAGE, TrackStage
from audio_from_video.gui.controller import ControllerState
from audio_from_video.gui.delegates import PaddedTextDelegate, StatusPillDelegate
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
    ActionBarFooter,
    IconStatusBanner,
    MiddleElideLabel,
    SectionCard,
    StatusLine,
    apply_danger_button_role,
    configure_table_view,
    make_divider,
    make_heading_label,
    make_muted_label,
    set_accessible,
)
from audio_from_video.gui.theme import set_widget_role
from audio_from_video.models import RunOutcome

if TYPE_CHECKING:
    from audio_from_video.gui.controller import RunController

CANCELLING_MESSAGE = "Cancelling… stopping FFmpeg and removing partial files"

_REFRESH_COALESCE_MS = 20

_BASE_COLUMN_WIDTHS: tuple[tuple[int, int], ...] = (
    (COL_STATUS, 174),
    (COL_SOURCE, 136),
    (COL_DESTINATION, 160),
    (COL_ACTION, 112),
    (COL_MATCH, 156),
)
_BASE_TABLE_VIEWPORT_WIDTH = 920
_EXTRA_WIDTH_SHARES: tuple[tuple[int, float], ...] = (
    (COL_SOURCE, 0.26),
    (COL_DESTINATION, 0.32),
    (COL_ACTION, 0.04),
    (COL_MATCH, 0.20),
)

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
        self.setObjectName("run_screen")
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
        self._last_chip_text = ""
        self._updating_columns = False
        self._user_resized_columns = False

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(_REFRESH_COALESCE_MS)
        self._refresh_timer.timeout.connect(self._on_refresh_timeout)

        set_accessible(
            self,
            "Run progress screen",
            "Displays live extraction progress and status for each track in the batch.",
        )

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        content = QWidget(self)
        content.setObjectName("run_content")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 14, 24, 12)
        layout.setSpacing(10)

        # Elevated progress hero card grouping title, live status, progress bar, and counts.
        progress_card = SectionCard(
            margins=(16, 12, 16, 12),
            spacing=8,
            parent=content,
        )
        progress_card.setObjectName("run_progress_card")

        hero_header = QHBoxLayout()
        hero_header.setContentsMargins(0, 0, 0, 0)
        hero_header.setSpacing(12)

        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(2)
        self.title_label = make_heading_label("Extracting Audio", parent=progress_card)
        self.title_label.setObjectName("run_title_label")
        title_col.addWidget(self.title_label)

        self._subtitle_label = make_muted_label(
            "Converting media files, embedding tags, and saving MP3 tracks.",
            scale=0.9,
            parent=progress_card,
        )
        self._subtitle_label.setObjectName("run_subtitle_label")
        title_col.addWidget(self._subtitle_label)
        hero_header.addLayout(title_col, stretch=1)

        self._progress_chip = QLabel("0 / 0 tracks", progress_card)
        self._progress_chip.setObjectName("run_progress_chip")
        self._progress_chip.setTextFormat(Qt.TextFormat.PlainText)
        self._progress_chip.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._progress_chip.setProperty("afvSurface", "chip")
        self._progress_chip.setProperty("afvTone", "neutral")
        set_accessible(
            self._progress_chip,
            "Tracks finished badge",
            "Compact readout showing how many tracks have finished in this batch.",
        )
        hero_header.addWidget(self._progress_chip, alignment=Qt.AlignmentFlag.AlignVCenter)
        progress_card.body_layout.addLayout(hero_header)
        progress_card.body_layout.addWidget(make_divider(progress_card))

        self.status_line = StatusLine(
            "status_working",
            "Preparing extraction…",
            icon_size=18,
            parent=progress_card,
        )
        self.status_line.setObjectName("run_status_line")
        status_font = QFont(self.status_line.text_label.font())
        status_font.setWeight(QFont.Weight.DemiBold)
        self.status_line.text_label.setFont(status_font)
        progress_card.body_layout.addWidget(self.status_line)

        self.progress_bar = QProgressBar(progress_card)
        self.progress_bar.setObjectName("run_progress_bar")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setProperty("afvDensity", "labeled")
        set_accessible(
            self.progress_bar,
            "Overall extraction progress",
            "0 of 0 tracks complete.",
        )
        progress_card.body_layout.addWidget(self.progress_bar)

        counts_strip = QFrame(progress_card)
        counts_strip.setObjectName("run_counts_strip")
        counts_strip.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        counts_strip.setFrameShape(QFrame.Shape.NoFrame)
        counts_strip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        counts_strip.setProperty("afvSurface", "card-inset")
        counts_layout = QHBoxLayout(counts_strip)
        counts_layout.setContentsMargins(12, 7, 12, 7)
        counts_layout.setSpacing(0)

        self.counts_label = QLabel(
            "Queued: 0 · Extracting: 0 · Done: 0 · Skipped: 0 · Failed: 0 · Cancelled: 0",
            counts_strip,
        )
        self.counts_label.setObjectName("run_counts_label")
        self.counts_label.setWordWrap(True)
        self.counts_label.setTextFormat(Qt.TextFormat.PlainText)
        self.counts_label.setProperty("afvText", "muted")
        set_accessible(
            self.counts_label,
            "Track status counts",
            "Live counts of queued, extracting, completed, skipped, failed, and cancelled tracks.",
        )
        counts_layout.addWidget(self.counts_label, stretch=1)
        progress_card.body_layout.addWidget(counts_strip)

        layout.addWidget(progress_card)

        self.error_banner = IconStatusBanner(content)
        layout.addWidget(self.error_banner)

        self.table_view = QTableView(content)
        self.table_view.setObjectName("run_table_view")
        self.table_view.setModel(self._model)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table_view.setMouseTracking(True)
        configure_table_view(self.table_view)

        self._status_delegate = StatusPillDelegate(self.table_view)
        self._text_delegate = PaddedTextDelegate(self.table_view)
        self.table_view.setItemDelegateForColumn(COL_STATUS, self._status_delegate)
        for col in (COL_SOURCE, COL_DESTINATION, COL_ACTION, COL_MATCH, COL_WARNINGS):
            self.table_view.setItemDelegateForColumn(col, self._text_delegate)

        header = self.table_view.horizontalHeader()
        header.setStretchLastSection(True)
        header.setMinimumSectionSize(64)
        header.setSectionResizeMode(COL_STATUS, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_SOURCE, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_DESTINATION, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_ACTION, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_MATCH, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_WARNINGS, QHeaderView.ResizeMode.Stretch)
        # Balanced initial widths fitting inside 980x720 with zero horizontal scrollbar,
        # leaving enough room for multi-word pills ("Done (unmatched)", "Skipped (no audio)").
        self._updating_columns = True
        try:
            for col_idx, width in _BASE_COLUMN_WIDTHS:
                header.resizeSection(col_idx, width)
        finally:
            self._updating_columns = False
        header.sectionResized.connect(self._on_header_section_resized)
        table_vp = self.table_view.viewport()
        if table_vp is not None:
            table_vp.installEventFilter(self)
        set_accessible(
            self.table_view,
            "Extraction progress table",
            "Live per-track status during Stage 2 extraction, including failures and errors.",
        )
        layout.addWidget(self.table_view, stretch=1)

        outer_layout.addWidget(content, stretch=1)

        # Sticky footer pinning Cancel reliably at the bottom right across all sizes.
        footer = ActionBarFooter(self, margins=(24, 10, 24, 12), spacing=10)
        footer.setObjectName("run_footer")
        self._footer_hint = MiddleElideLabel(
            "Cancelling stops active FFmpeg processes and removes partial files.",
            footer,
        )
        self._footer_hint.setObjectName("run_footer_hint")
        self._footer_hint.setProperty("afvText", "muted")
        self._footer_hint.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        footer.content_layout.addWidget(
            self._footer_hint,
            stretch=1,
            alignment=Qt.AlignmentFlag.AlignVCenter,
        )

        self.cancel_button = QPushButton("&Cancel", footer)
        self.cancel_button.setObjectName("run_cancel_button")
        apply_danger_button_role(self.cancel_button)
        self.cancel_button.setToolTip(
            "Stop active FFmpeg processes, remove partial files, and move to the Summary screen."
        )
        set_accessible(
            self.cancel_button,
            "Cancel extraction",
            "Stop active FFmpeg processes, remove partial files, and move to Summary.",
        )
        footer.content_layout.addWidget(self.cancel_button)
        outer_layout.addWidget(footer)

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
            set_widget_role(self._progress_chip, "afvTone", "warn")
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
        noun = "track" if total == 1 else "tracks"
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
                    f"Extraction progress: {completed} of {total} {noun} finished ({pct}%)."
                )
        else:
            if self._last_progress_max != 1:
                self._last_progress_max = 1
                self.progress_bar.setRange(0, 1)
            if self._last_progress_value != 0:
                self._last_progress_value = 0
                self.progress_bar.setValue(0)
            self.progress_bar.setAccessibleDescription("0 of 0 tracks complete.")

        chip_text = f"{completed} / {total} {noun}"
        if chip_text != self._last_chip_text:
            self._last_chip_text = chip_text
            self._progress_chip.setText(chip_text)

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
            overflow = len(self._failed_messages) - len(sample)
            suffix = f" (+{overflow} more)" if overflow > 0 else ""
            self.error_banner.set_message(
                "status_error",
                f"Failed ({failed}): " + "; ".join(sample) + suffix,
                role_label="Extraction error",
            )
        else:
            self.error_banner.clear_message()

        if self._cancelling:
            set_widget_role(self._progress_chip, "afvTone", "warn")
            self.status_line.set_status("status_warn", CANCELLING_MESSAGE)
            return

        if total == 0:
            set_widget_role(self._progress_chip, "afvTone", "neutral")
            self.status_line.set_status("status_pending", "Waiting for tracks…")
        elif completed >= total:
            if failed > 0:
                set_widget_role(self._progress_chip, "afvTone", "error")
                self.status_line.set_status(
                    "status_warn",
                    f"Finished {completed} of {total} {noun} ({failed} failed).",
                )
            elif cancelled > 0:
                set_widget_role(self._progress_chip, "afvTone", "warn")
                self.status_line.set_status(
                    "status_cancelled",
                    f"Stopped after {completed - cancelled} of {total} {noun} "
                    f"({cancelled} cancelled).",
                )
            else:
                set_widget_role(self._progress_chip, "afvTone", "ok")
                self.status_line.set_status(
                    "status_ok",
                    f"Finished {completed} of {total} {noun}.",
                )
        elif extracting > 0:
            set_widget_role(
                self._progress_chip,
                "afvTone",
                "error" if failed > 0 else "working",
            )
            self.status_line.set_status(
                "status_working",
                f"Extracting ({completed} of {total} finished, {extracting} active)…",
            )
        else:
            set_widget_role(
                self._progress_chip,
                "afvTone",
                "error" if failed > 0 else "neutral",
            )
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

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Rebalance wide-screen table column proportions when the viewport resizes."""
        if watched is self.table_view.viewport() and event.type() == QEvent.Type.Resize:
            self._distribute_table_columns()
        return super().eventFilter(watched, event)

    def _distribute_table_columns(self) -> None:
        """Expand text columns proportionally on wide displays without overwriting user drags."""
        if self._user_resized_columns or self._updating_columns:
            return
        viewport = self.table_view.viewport()
        if viewport is None:
            return
        viewport_w = viewport.width()
        if viewport_w <= 0:
            return
        header = self.table_view.horizontalHeader()
        if header is None:
            return
        extra = max(0, viewport_w - _BASE_TABLE_VIEWPORT_WIDTH)
        shares = dict(_EXTRA_WIDTH_SHARES)
        self._updating_columns = True
        try:
            for col_idx, base_w in _BASE_COLUMN_WIDTHS:
                target_w = base_w + int(extra * shares.get(col_idx, 0.0))
                if header.sectionSize(col_idx) != target_w:
                    header.resizeSection(col_idx, target_w)
        finally:
            self._updating_columns = False

    @Slot(int, int, int)
    def _on_header_section_resized(
        self,
        logical_index: int,
        old_size: int,
        new_size: int,
    ) -> None:
        if (
            not self._updating_columns
            and logical_index != COL_WARNINGS
            and old_size > 0
            and old_size != new_size
            and self.isVisible()
        ):
            self._user_resized_columns = True

    def showEvent(self, event: QShowEvent) -> None:
        """Flush any progress updates deferred while the screen was hidden."""
        super().showEvent(event)
        if self._stale_while_hidden or self._dirty:
            self.flush_updates()
        self._distribute_table_columns()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Keep wide-window table column proportions balanced without stealing user drags."""
        super().resizeEvent(event)
        self._distribute_table_columns()

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
