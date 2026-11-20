"""Planning progress screen showing probing, lookup stats, notices, and cancel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from audio_from_video.events import LookupStats, TrackEvent, TrackStage
from audio_from_video.gui.controller import ControllerState
from audio_from_video.gui.resources import load_icon
from audio_from_video.gui.screens._common import (
    IconStatusBanner,
    StatusLine,
    log_directory,
    make_heading_label,
    open_local_path,
    set_accessible,
)

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
        self._expected_total: int | None = None
        self._max_index: int = -1
        self._probed: set[int] = set()
        self._looked_up: set[int] = set()
        self._planned: set[int] = set()
        self._notices: list[str] = []
        self._last_announced_bucket: int = -1
        self._last_announced_text: str = ""
        self._cancelling = False

        set_accessible(
            self,
            "Planning progress screen",
            "Displays media probing, metadata lookup progress, lookup statistics, and notices.",
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        self.title_label = make_heading_label("Scanning and Planning", parent=self)
        layout.addWidget(self.title_label)

        # Notice banner area (displays outage warnings and provider notices).
        self.notice_banner = IconStatusBanner(self)
        layout.addWidget(self.notice_banner)

        # Progress group.
        progress_group = QGroupBox("Progress", self)
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setSpacing(10)

        self.status_line = StatusLine("status_working", "Scanning folders…", parent=progress_group)
        progress_layout.addWidget(self.status_line)

        self.progress_bar = QProgressBar(progress_group)
        self.progress_bar.setRange(0, 0)  # Indeterminate while folder discovery runs.
        self.progress_bar.setTextVisible(True)
        set_accessible(
            self.progress_bar,
            "Planning progress",
            "Scanning folders.",
        )
        progress_layout.addWidget(self.progress_bar)

        self.detail_label = QLabel("", progress_group)
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextFormat(Qt.TextFormat.PlainText)
        progress_layout.addWidget(self.detail_label)

        self.stats_label = QLabel("Cache hits: 0 · MusicBrainz requests: 0", progress_group)
        self.stats_label.setTextFormat(Qt.TextFormat.PlainText)
        set_accessible(
            self.stats_label,
            "Lookup statistics",
            "Cache hits and outbound MusicBrainz request count for the current planning run.",
        )
        progress_layout.addWidget(self.stats_label)
        layout.addWidget(progress_group)

        # Optional error panel (for unexpected planning exceptions).
        self.error_frame = QFrame(self)
        self.error_frame.setFrameShape(QFrame.Shape.StyledPanel)
        error_layout = QVBoxLayout(self.error_frame)
        error_layout.setContentsMargins(12, 10, 12, 10)
        error_layout.setSpacing(8)
        self.error_header = StatusLine("status_error", "Planning failed", parent=self.error_frame)
        error_layout.addWidget(self.error_header)

        self.error_details = QPlainTextEdit(self.error_frame)
        self.error_details.setReadOnly(True)
        self.error_details.setMaximumHeight(120)
        set_accessible(
            self.error_details,
            "Planning error details",
            "Diagnostic details for the error encountered during planning.",
        )
        error_layout.addWidget(self.error_details)

        err_buttons = QHBoxLayout()
        self.copy_details_button = QPushButton("Copy &details", self.error_frame)
        set_accessible(
            self.copy_details_button,
            "Copy error details",
            "Copy the error message and details to the clipboard.",
        )
        self.open_log_button = QPushButton("Open &log folder", self.error_frame)
        self.open_log_button.setIcon(load_icon("folder"))
        set_accessible(
            self.open_log_button,
            "Open log folder",
            "Open the application log directory in the file manager.",
        )
        err_buttons.addWidget(self.copy_details_button)
        err_buttons.addWidget(self.open_log_button)
        err_buttons.addStretch(1)
        error_layout.addLayout(err_buttons)
        self.error_frame.setVisible(False)
        layout.addWidget(self.error_frame)

        # Cancel action bar.
        footer = QHBoxLayout()
        footer.addStretch(1)
        self.cancel_button = QPushButton("&Cancel", self)
        set_accessible(
            self.cancel_button,
            "Cancel planning",
            "Stop probing and metadata lookup and return to the Inputs screen.",
        )
        footer.addWidget(self.cancel_button)
        layout.addLayout(footer)
        layout.addStretch(1)

        # Wire widget events and controller signals via bound methods only.
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self.copy_details_button.clicked.connect(self._on_copy_details)
        self.open_log_button.clicked.connect(self._on_open_log)

        self._controller.event_received.connect(self._on_event_received)
        self._controller.stats_changed.connect(self._on_stats_changed)
        self._controller.notice.connect(self._on_notice)
        self._controller.state_changed.connect(self._on_state_changed)

        self.setTabOrder(self.copy_details_button, self.open_log_button)
        self.setTabOrder(self.open_log_button, self.cancel_button)

    @property
    def notices(self) -> tuple[str, ...]:
        """Return unique notices received during the current planning attempt."""
        return tuple(self._notices)

    @property
    def last_announced_progress(self) -> str:
        """Return the most recent throttled accessibility progress description."""
        return self._last_announced_text

    def set_total_tracks(self, total: int | None) -> None:
        """Hint the total number of tracks being planned and switch to a determinate bar."""
        if total is not None and total > 0:
            self._expected_total = total
            if self._planned:
                done = len(self._planned)
                self.progress_bar.setRange(0, max(total, done))
                self.progress_bar.setValue(min(done, max(total, done)))
            elif self._looked_up:
                done = len(self._looked_up)
                self.progress_bar.setRange(0, max(total, done))
                self.progress_bar.setValue(min(done, max(total, done)))
            elif self._probed:
                done = len(self._probed)
                self.progress_bar.setRange(0, max(total, done))
                self.progress_bar.setValue(min(done, max(total, done)))
            else:
                self.progress_bar.setRange(0, total)
                self.progress_bar.setValue(0)
        else:
            self._expected_total = None

    def reset(self) -> None:
        """Reset indicators, notices, and error panel for a fresh planning run."""
        self._expected_total = None
        self._max_index = -1
        self._probed.clear()
        self._looked_up.clear()
        self._planned.clear()
        self._notices.clear()
        self._last_announced_bucket = -1
        self._last_announced_text = ""
        self._cancelling = False
        self.status_line.set_status("status_working", "Scanning folders…")
        self.detail_label.setText("")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setAccessibleDescription("Scanning folders.")
        self.stats_label.setText("Cache hits: 0 · MusicBrainz requests: 0")
        self.notice_banner.clear_message()
        self.error_details.clear()
        self.error_frame.setVisible(False)
        self.cancel_button.setEnabled(True)

    def show_error(self, title: str, details: str) -> None:
        """Display an inline planning error panel with copy and log actions."""
        heading = title.strip() or "Planning failed"
        self.error_header.set_status("status_error", heading)
        self.error_details.setPlainText(details)
        self.error_frame.setVisible(True)

    def _effective_total(self) -> int:
        observed = max(self._max_index + 1, len(self._probed), len(self._looked_up))
        if self._expected_total is not None:
            return max(self._expected_total, observed, 1)
        return max(observed, 1)

    def _announce_progress(
        self, current: int, total: int, phase: str, *, force: bool = False
    ) -> None:
        if total <= 0:
            return
        pct = min(100, max(0, round((current / total) * 100)))
        bucket = pct // 10
        if force or bucket != self._last_announced_bucket or current == total:
            self._last_announced_bucket = bucket
            description = f"{phase}: {current} of {total} ({pct}%)."
            self._last_announced_text = description
            self.progress_bar.setAccessibleDescription(description)

    @Slot(object)
    def _on_event_received(self, event: TrackEvent) -> None:
        if event.index < 0:
            if (
                event.index == -1
                and event.stage == TrackStage.PROBING
                and event.message
                and event.message.startswith("total_tracks=")
            ):
                raw_total = event.message.split("=", 1)[1].strip()
                try:
                    total_tracks = int(raw_total)
                except ValueError:
                    total_tracks = 0
                if total_tracks > 0:
                    self.set_total_tracks(total_tracks)
                    self.progress_bar.setRange(0, total_tracks)
                    self.progress_bar.setValue(0)
                    self.status_line.set_status(
                        "status_working",
                        f"Probing media (0 of {total_tracks})…",
                    )
                    self._announce_progress(0, total_tracks, "Probing", force=True)
                return
            if event.index == -1 and event.message:
                self._on_notice(event.message)
            return

        if event.index >= 0:
            self._max_index = max(self._max_index, event.index)

        stage = event.stage
        filename = event.source.name if event.source is not None else f"track {event.index + 1}"

        if stage == TrackStage.PROBING:
            self._probed.add(event.index)
            total = self._effective_total()
            done = len(self._probed)
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(min(done, total))
            self.status_line.set_status(
                "status_working",
                f"Probing media ({done} of {total})…",
            )
            self.detail_label.setText(filename)
            self._announce_progress(done, total, "Probing", force=True)
            return

        if stage == TrackStage.LOOKUP:
            self._looked_up.add(event.index)
            total = self._effective_total()
            done = len(self._looked_up)
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(min(done, total))
            self.status_line.set_status(
                "status_working",
                f"Looking up metadata ({done} of {total})…",
            )
            self.detail_label.setText(filename)
            self._announce_progress(done, total, "Looking up metadata", force=True)
            return

        if stage == TrackStage.AWAITING_INPUT:
            self.status_line.set_status("status_info", "Waiting for your answer…")
            self.detail_label.setText(filename)
            return

        if stage == TrackStage.PLANNED:
            self._planned.add(event.index)
            total = self._effective_total()
            done = len(self._planned)
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(min(done, total))
            self.status_line.set_status(
                "status_ok",
                f"Planned {done} of {total} tracks.",
            )
            self.detail_label.setText(filename)
            self._announce_progress(done, total, "Planning complete", force=True)

    @Slot(object)
    def _on_stats_changed(self, stats: LookupStats) -> None:
        self.stats_label.setText(
            f"Cache hits: {stats.cache_hits} · MusicBrainz requests: {stats.mb_requests}"
        )

    @Slot(str)
    def _on_notice(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        if cleaned not in self._notices:
            self._notices.append(cleaned)
        lowered = cleaned.casefold()
        is_warning = any(
            word in lowered
            for word in ("unreachable", "unavailable", "offline", "warning", "error", "failed")
        )
        icon_name = "status_warn" if is_warning else "status_info"
        joined = "\n".join(self._notices)
        self.notice_banner.set_message(icon_name, joined, role_label="Planning notice")

    @Slot(object)
    def _on_state_changed(self, state: object) -> None:
        if state == ControllerState.PLANNING:
            self.reset()
        elif state == ControllerState.CANCELLING:
            self._cancelling = True
            self.cancel_button.setEnabled(False)
            self.status_line.set_status(
                "status_warn",
                "Cancelling… waiting for active step to finish.",
            )

    @Slot()
    def _on_cancel_clicked(self) -> None:
        if self._cancelling:
            return
        self._cancelling = True
        self.cancel_button.setEnabled(False)
        self.status_line.set_status(
            "status_warn",
            "Cancelling… waiting for active step to finish.",
        )
        self._controller.cancel()

    @Slot()
    def _on_copy_details(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return
        heading = self.error_header.text
        details = self.error_details.toPlainText()
        payload = f"{heading}\n\n{details}".strip() if details else heading
        clipboard.setText(payload)

    @Slot()
    def _on_open_log(self) -> None:
        log_dir = log_directory()
        if not open_local_path(log_dir):
            try:
                exists = log_dir.expanduser().exists()
            except OSError:
                exists = False
            text = (
                f"Could not open log folder: {log_dir}."
                if exists
                else f"Log folder does not exist yet: {log_dir}."
            )
            self.notice_banner.set_message(
                "status_info",
                text,
                role_label="Log folder notice",
            )
