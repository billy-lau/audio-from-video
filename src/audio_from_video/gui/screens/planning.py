"""Planning progress screen showing probing, lookup stats, notices, and cancel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from audio_from_video.events import LookupStats, TrackEvent, TrackStage
from audio_from_video.gui.controller import ControllerState
from audio_from_video.gui.screens._common import (
    IconStatusBanner,
    MiddleElideLabel,
    SectionCard,
    StatusLine,
    ThemedSvgLabel,
    apply_secondary_button_role,
    log_directory,
    make_divider,
    make_heading_label,
    make_mono_font,
    make_muted_label,
    open_local_path,
    set_accessible,
)
from audio_from_video.gui.theme import set_widget_role

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

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setObjectName("planning_scroll_area")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        set_accessible(scroll, "Planning content", "Scrollable container for planning progress.")

        body = QWidget(scroll)
        body.setObjectName("planning_scroll_body")
        body_v = QVBoxLayout(body)
        body_v.setContentsMargins(24, 20, 24, 20)
        body_v.setSpacing(0)
        body_v.addStretch(1)

        center_row = QHBoxLayout()
        center_row.setContentsMargins(0, 0, 0, 0)
        center_row.setSpacing(0)

        card = SectionCard(
            margins=(24, 22, 24, 20),
            spacing=14,
            parent=body,
        )
        card.setObjectName("planning_card")
        card.setMinimumWidth(420)
        card.setMaximumWidth(640)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        center_row.addStretch(0)
        center_row.addWidget(card, stretch=1)
        center_row.addStretch(0)
        body_v.addLayout(center_row)
        body_v.addStretch(1)

        # Card header with heading and calm subtitle.
        header_col = QVBoxLayout()
        header_col.setContentsMargins(0, 0, 0, 0)
        header_col.setSpacing(3)
        self.title_label = make_heading_label("Scanning and Planning", parent=card)
        self.title_label.setObjectName("planning_title_label")
        subtitle = make_muted_label(
            "Probing media files and looking up track metadata.",
            parent=card,
        )
        subtitle.setObjectName("planning_subtitle_label")
        header_col.addWidget(self.title_label)
        header_col.addWidget(subtitle)
        card.body_layout.addLayout(header_col)

        # Compact, read-only 3-phase step strip so bar resets between phases make sense.
        self._phase_labels: dict[str, QLabel] = {}
        phase_row = QHBoxLayout()
        phase_row.setContentsMargins(0, 0, 0, 0)
        phase_row.setSpacing(6)
        phase_specs: tuple[tuple[str, str], ...] = (
            ("scan", "1. Scan"),
            ("probe", "2. Probe"),
            ("match", "3. Match"),
        )
        for idx, (key, caption) in enumerate(phase_specs):
            if idx > 0:
                sep = ThemedSvgLabel("chevron_right", size=10, parent=card)
                sep.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                phase_row.addWidget(sep, alignment=Qt.AlignmentFlag.AlignVCenter)
            pill = QLabel(caption, card)
            pill.setObjectName(f"planning_phase_{key}")
            pill.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            pill.setTextFormat(Qt.TextFormat.PlainText)
            pill.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            pill.setProperty("afvStepState", "active" if idx == 0 else "pending")
            self._phase_labels[key] = pill
            phase_row.addWidget(pill, alignment=Qt.AlignmentFlag.AlignVCenter)
        phase_row.addStretch(1)
        card.body_layout.addLayout(phase_row)

        # Notice banner area (displays outage warnings and provider notices).
        self.notice_banner = IconStatusBanner(card)
        self.notice_banner.setObjectName("planning_notice_banner")
        card.body_layout.addWidget(self.notice_banner)

        # Live status, progress bar, and single-line middle-elided filename row.
        progress_col = QVBoxLayout()
        progress_col.setContentsMargins(0, 2, 0, 0)
        progress_col.setSpacing(8)

        self.status_line = StatusLine("status_working", "Scanning folders…", parent=card)
        self.status_line.setObjectName("planning_status_line")
        status_font = QFont(self.status_line.text_label.font())
        status_font.setWeight(QFont.Weight.DemiBold)
        self.status_line.text_label.setFont(status_font)
        progress_col.addWidget(self.status_line)

        self.progress_bar = QProgressBar(card)
        self.progress_bar.setObjectName("planning_progress_bar")
        self.progress_bar.setRange(0, 0)  # Indeterminate while folder discovery runs.
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setProperty("afvDensity", "compact")
        set_accessible(
            self.progress_bar,
            "Planning progress",
            "Scanning folders.",
        )
        progress_col.addWidget(self.progress_bar)

        self.detail_label = MiddleElideLabel("", card)
        self.detail_label.setObjectName("planning_detail_label")
        self.detail_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.detail_label.setProperty("afvText", "muted")
        set_accessible(
            self.detail_label,
            "Current track",
            "Filename of the track currently being processed.",
        )
        progress_col.addWidget(self.detail_label)
        card.body_layout.addLayout(progress_col)

        # Recessed stats footer strip inside the card.
        stats_box = QFrame(card)
        stats_box.setObjectName("planning_stats_box")
        stats_box.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        stats_box.setFrameShape(QFrame.Shape.StyledPanel)
        stats_box.setFrameShadow(QFrame.Shadow.Plain)
        stats_box.setProperty("afvSurface", "card-inset")
        stats_row = QHBoxLayout(stats_box)
        stats_row.setContentsMargins(12, 8, 12, 8)
        stats_row.setSpacing(8)

        self.stats_label = make_muted_label(
            "Cache hits: 0 · MusicBrainz requests: 0",
            scale=0.92,
            parent=stats_box,
        )
        self.stats_label.setObjectName("planning_stats_label")
        set_accessible(
            self.stats_label,
            "Lookup statistics",
            "Cache hits and outbound MusicBrainz request count for the current planning run.",
        )
        stats_row.addWidget(self.stats_label, stretch=1)
        card.body_layout.addWidget(stats_box)

        # Optional error panel (for unexpected planning exceptions).
        self.error_frame = QFrame(card)
        self.error_frame.setObjectName("planning_error_frame")
        self.error_frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.error_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.error_frame.setFrameShadow(QFrame.Shadow.Plain)
        self.error_frame.setProperty("afvBanner", True)
        self.error_frame.setProperty("afvTone", "error")
        error_layout = QVBoxLayout(self.error_frame)
        error_layout.setContentsMargins(14, 12, 14, 12)
        error_layout.setSpacing(8)
        self.error_header = StatusLine(
            "status_error",
            "Planning failed",
            selectable=True,
            parent=self.error_frame,
        )
        self.error_header.setObjectName("planning_error_header")
        error_layout.addWidget(self.error_header)

        self.error_details = QPlainTextEdit(self.error_frame)
        self.error_details.setObjectName("planning_error_details")
        self.error_details.setReadOnly(True)
        self.error_details.setProperty("afvReadonly", True)
        self.error_details.setFont(make_mono_font(self.error_details))
        self.error_details.setMaximumHeight(96)
        self.error_details.setToolTip("Diagnostic details captured for the planning error.")
        set_accessible(
            self.error_details,
            "Planning error details",
            "Diagnostic details for the error encountered during planning.",
        )
        error_layout.addWidget(self.error_details)

        err_buttons = QHBoxLayout()
        err_buttons.setContentsMargins(0, 0, 0, 0)
        err_buttons.setSpacing(8)
        self.copy_details_button = QPushButton("Copy &details", self.error_frame)
        self.copy_details_button.setObjectName("planning_copy_details_button")
        apply_secondary_button_role(self.copy_details_button, icon_name="copy")
        self.copy_details_button.setToolTip(
            "Copy the error summary and diagnostic details to the clipboard."
        )
        set_accessible(
            self.copy_details_button,
            "Copy error details",
            "Copy the error message and details to the clipboard.",
        )
        self.open_log_button = QPushButton("Open &log folder", self.error_frame)
        self.open_log_button.setObjectName("planning_open_log_button")
        apply_secondary_button_role(self.open_log_button, icon_name="folder")
        self.open_log_button.setToolTip("Open the application log folder in your file manager.")
        set_accessible(
            self.open_log_button,
            "Open log folder",
            "Open the application log directory in the file manager.",
        )
        self._copy_feedback_label = make_muted_label("", scale=0.88, parent=self.error_frame)
        self._copy_feedback_label.setObjectName("planning_copy_feedback_label")
        self._copy_feedback_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._copy_feedback_label.setVisible(False)
        self._copy_feedback_timer = QTimer(self)
        self._copy_feedback_timer.setSingleShot(True)
        self._copy_feedback_timer.setInterval(2_500)
        self._copy_feedback_timer.timeout.connect(self._clear_copy_feedback)

        err_buttons.addWidget(self.copy_details_button)
        err_buttons.addWidget(self.open_log_button)
        err_buttons.addWidget(self._copy_feedback_label)
        err_buttons.addStretch(1)
        error_layout.addLayout(err_buttons)

        # Dedicated feedback line so "Open log folder" failures never overwrite
        # provider outage or offline warnings on self.notice_banner.
        self.log_feedback_label = StatusLine(
            "status_info",
            "",
            selectable=True,
            parent=self.error_frame,
        )
        self.log_feedback_label.setObjectName("planning_log_feedback_label")
        set_accessible(
            self.log_feedback_label,
            "Log folder feedback",
            "Status message shown if opening the application log folder fails.",
        )
        set_accessible(
            self.log_feedback_label.text_label,
            "Log folder feedback",
            "Status message shown if opening the application log folder fails.",
        )
        self.log_feedback_label.setVisible(False)
        error_layout.addWidget(self.log_feedback_label)

        self.error_frame.setVisible(False)
        card.body_layout.addWidget(self.error_frame)

        # Card footer divider and Cancel action.
        card.body_layout.addWidget(make_divider(card))

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addStretch(1)
        self.cancel_button = QPushButton("&Cancel", card)
        self.cancel_button.setObjectName("planning_cancel_button")
        apply_secondary_button_role(self.cancel_button)
        self.cancel_button.setToolTip(
            "Stop probing and metadata lookup and return to the Inputs screen."
        )
        set_accessible(
            self.cancel_button,
            "Cancel planning",
            "Stop probing and metadata lookup and return to the Inputs screen.",
        )
        footer.addWidget(self.cancel_button)
        card.body_layout.addLayout(footer)

        scroll.setWidget(body)
        outer_layout.addWidget(scroll)

        self._active_phase_idx: int | None = None
        self._set_phase("scan")

        # Wire widget events and controller signals via bound methods only.
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self.copy_details_button.clicked.connect(self._on_copy_details)
        self.open_log_button.clicked.connect(self._on_open_log)

        self._controller.event_received.connect(self._on_event_received)
        self._controller.stats_changed.connect(self._on_stats_changed)
        self._controller.notice.connect(self._on_notice)
        self._controller.state_changed.connect(self._on_state_changed)

        self.setTabOrder(self.error_details, self.copy_details_button)
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

    def _set_progress_density(self, *, indeterminate: bool) -> None:
        """Switch the progress bar between compact indeterminate and labeled determinate style."""
        if indeterminate:
            self.progress_bar.setTextVisible(False)
            set_widget_role(self.progress_bar, "afvDensity", "compact")
        else:
            self.progress_bar.setTextVisible(True)
            set_widget_role(self.progress_bar, "afvDensity", "labeled")

    def _set_phase(self, phase: str) -> None:
        """Update the 3-step sub-phase indicator strip inside the planning card."""
        active_order = {"scan": 0, "probe": 1, "match": 2, "done": 3}
        active_idx = active_order.get(phase, 0)
        if active_idx == self._active_phase_idx:
            return
        self._active_phase_idx = active_idx
        steps: tuple[tuple[str, str, str], ...] = (
            ("scan", "1. Scan", "Scan"),
            ("probe", "2. Probe", "Probe"),
            ("match", "3. Match", "Match"),
        )
        for idx, (key, numbered, short_name) in enumerate(steps):
            pill = self._phase_labels[key]
            if idx < active_idx:
                pill.setText(f"✓ {short_name}")
                set_widget_role(pill, "afvStepState", "done")
            elif idx == active_idx:
                pill.setText(numbered)
                set_widget_role(pill, "afvStepState", "active")
            else:
                pill.setText(numbered)
                set_widget_role(pill, "afvStepState", "pending")
            pill.adjustSize()
            pill.updateGeometry()

    @Slot()
    def _clear_copy_feedback(self) -> None:
        """Hide the temporary clipboard copy confirmation label."""
        self._copy_feedback_timer.stop()
        self._copy_feedback_label.setText("")
        self._copy_feedback_label.setVisible(False)

    def set_total_tracks(self, total: int | None) -> None:
        """Hint the total number of tracks being planned and switch to a determinate bar."""
        if total is not None and total > 0:
            self._expected_total = total
            self._set_progress_density(indeterminate=False)
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

    def _clear_log_feedback(self) -> None:
        """Clear and hide the inline log-folder status message."""
        self.log_feedback_label.set_status("status_info", "")
        set_accessible(
            self.log_feedback_label,
            "Log folder feedback",
            "Status message shown if opening the application log folder fails.",
        )
        set_accessible(
            self.log_feedback_label.text_label,
            "Log folder feedback",
            "Status message shown if opening the application log folder fails.",
        )
        self.log_feedback_label.setVisible(False)

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
        self._active_phase_idx = None
        self._set_phase("scan")
        self.status_line.set_status("status_working", "Scanning folders…")
        self.detail_label.setText("")
        self.progress_bar.setRange(0, 0)
        self._set_progress_density(indeterminate=True)
        self.progress_bar.setAccessibleDescription("Scanning folders.")
        self.stats_label.setText("Cache hits: 0 · MusicBrainz requests: 0")
        self.notice_banner.clear_message()
        self._clear_copy_feedback()
        self._clear_log_feedback()
        self.error_details.clear()
        self.error_frame.setVisible(False)
        self.cancel_button.setEnabled(True)

    def show_error(self, title: str, details: str) -> None:
        """Display an inline planning error panel with copy and log actions."""
        heading = title.strip() or "Planning failed"
        self.error_header.set_status("status_error", heading)
        self.error_details.setPlainText(details)
        self._clear_copy_feedback()
        self._clear_log_feedback()
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
                    self._set_phase("probe")
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
            self._set_phase("probe")
            self._set_progress_density(indeterminate=False)
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
            self._set_phase("match")
            self._set_progress_density(indeterminate=False)
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
            self._set_phase("match")
            self.status_line.set_status("status_info", "Waiting for your answer…")
            self.detail_label.setText(filename)
            return

        if stage == TrackStage.PLANNED:
            self._planned.add(event.index)
            total = self._effective_total()
            done = len(self._planned)
            self._set_phase("done" if done >= total else "match")
            self._set_progress_density(indeterminate=False)
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(min(done, total))
            noun = "track" if total == 1 else "tracks"
            self.status_line.set_status(
                "status_ok",
                f"Planned {done} of {total} {noun}.",
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
        self._copy_feedback_label.setText("Copied to clipboard")
        self._copy_feedback_label.setVisible(True)
        self._copy_feedback_timer.start()

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
            self.log_feedback_label.set_status("status_info", text)
            set_accessible(self.log_feedback_label, "Log folder notice", text)
            set_accessible(self.log_feedback_label.text_label, "Log folder notice", text)
            self.log_feedback_label.setVisible(True)
            return
        self._clear_log_feedback()
