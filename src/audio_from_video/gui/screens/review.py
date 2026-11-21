"""Plan review screen displaying the planned tracks, banners, filters, and details pane."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QEvent,
    QItemSelection,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    QSignalBlocker,
    QSize,
    Qt,
    QTimer,
    Slot,
)
from PySide6.QtGui import QFont, QHideEvent, QKeyEvent, QResizeEvent, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from audio_from_video.gui.controller import ControllerState
from audio_from_video.gui.delegates import PaddedTextDelegate, StatusPillDelegate
from audio_from_video.gui.models.plan_table import (
    COL_ACTION,
    COL_DESTINATION,
    COL_MATCH,
    COL_SOURCE,
    COL_STATUS,
    COL_WARNINGS,
    PlanFilterMode,
    PlanFilterProxyModel,
    PlanTableModel,
    format_action_text,
    format_destination,
    is_failed_action,
    is_needs_attention,
    is_skipped_action,
)
from audio_from_video.gui.screens._common import (
    ActionBarFooter,
    IconStatusBanner,
    MiddleElideLabel,
    SectionCard,
    ThemedSvgLabel,
    apply_filter_pill_role,
    apply_primary_button_role,
    apply_secondary_button_role,
    configure_table_view,
    make_divider,
    make_heading_label,
    make_muted_label,
    set_accessible,
)
from audio_from_video.gui.theme import set_widget_role
from audio_from_video.models import PlanAction, PlanItem
from audio_from_video.presentation import (
    MAX_DISPLAY_CANDIDATES,
    format_candidate_date,
    format_candidate_track,
)

if TYPE_CHECKING:
    from pathlib import Path

    from audio_from_video.gui.controller import RunController


def _format_item_details(item: PlanItem, output_root: Path | None) -> str:
    """Format full, untruncated metadata, warnings, candidates, and conflicts for one item."""
    lines: list[str] = [
        f"Source: {item.media_input.path}",
        f"Action: {format_action_text(item)}",
        f"Destination: {format_destination(item.output_path, output_root)}",
    ]
    if item.output_path is not None:
        lines.append(f"Full destination path: {item.output_path}")
    if item.conflict_note:
        lines.append(f"Conflict note: {item.conflict_note}")
    if item.error:
        lines.append(f"Error: {item.error}")

    if item.selected_stream is not None:
        stream = item.selected_stream
        stream_bits = [f"#{stream.audio_index}", stream.codec_name, f"{stream.channels} ch"]
        if stream.sample_rate:
            stream_bits.append(f"{stream.sample_rate} Hz")
        if stream.language:
            stream_bits.append(f"lang={stream.language}")
        if stream.title:
            stream_bits.append(f"title={stream.title!r}")
        lines.append("Audio stream: " + ", ".join(stream_bits))

    if item.metadata is not None:
        meta = item.metadata
        lines.append("")
        lines.append("Metadata:")
        lines.append(
            f"  Status: {meta.match_status.value} ({meta.source_description or 'unknown'})"
        )
        if meta.artist:
            lines.append(f"  Artist: {meta.artist}")
        if meta.album_artist:
            lines.append(f"  Album artist: {meta.album_artist}")
        if meta.featuring:
            lines.append(f"  Featuring: {', '.join(meta.featuring)}")
        if meta.title:
            lines.append(f"  Title: {meta.title}")
        if meta.album:
            lines.append(f"  Album: {meta.album}")
        if meta.track_number is not None:
            if meta.track_total:
                lines.append(f"  Track: {meta.track_number}/{meta.track_total}")
            else:
                lines.append(f"  Track: {meta.track_number}")
        if meta.disc_number is not None:
            if meta.disc_total:
                lines.append(f"  Disc: {meta.disc_number}/{meta.disc_total}")
            else:
                lines.append(f"  Disc: {meta.disc_number}")
        if meta.date:
            lines.append(f"  Date: {meta.date}")
        if meta.genre:
            lines.append(f"  Genre: {meta.genre}")
        if meta.mb_recording_id:
            lines.append(f"  MusicBrainz recording ID: {meta.mb_recording_id}")
        if meta.mb_release_id:
            lines.append(f"  MusicBrainz release ID: {meta.mb_release_id}")

    if item.warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in item.warnings:
            lines.append(f"  - {warning}")

    if item.top_candidates:
        lines.append("")
        lines.append("Top candidates:")
        for idx, cand in enumerate(item.top_candidates[:MAX_DISPLAY_CANDIDATES], start=1):
            album = cand.album or "—"
            date = format_candidate_date(cand)
            track = format_candidate_track(cand)
            lines.append(
                f"  {idx}. [{cand.score:.2f}] {cand.artist} — {cand.title} "
                f"(album: {album}, date: {date}, track: {track}, source: {cand.source})"
            )

    return "\n".join(lines)


_REFRESH_COALESCE_MS = 20

# Balanced baseline interactive column widths (sum 744 px) fitting inside the
# default 980x720 window width with zero horizontal overflow and generous space
# left for the stretched Warnings column.
_BASE_COLUMN_WIDTHS: tuple[tuple[int, int], ...] = (
    (COL_STATUS, 130),
    (COL_SOURCE, 150),
    (COL_DESTINATION, 176),
    (COL_ACTION, 122),
    (COL_MATCH, 166),
)
_BASE_TABLE_VIEWPORT_WIDTH = 920
_EXTRA_WIDTH_SHARES: tuple[tuple[int, float], ...] = (
    (COL_SOURCE, 0.26),
    (COL_DESTINATION, 0.32),
    (COL_ACTION, 0.04),
    (COL_MATCH, 0.20),
)

_RowFlags = tuple[bool, bool, bool, bool]

_EMPTY_FILTER_MESSAGES: dict[PlanFilterMode, str] = {
    PlanFilterMode.NEEDS_ATTENTION: (
        "No tracks need attention. Select All to view every planned track."
    ),
    PlanFilterMode.SKIPPED: "No tracks are skipped in this plan.",
    PlanFilterMode.FAILED: "No tracks failed during planning.",
}


class _ReviewSplitter(QSplitter):
    """Vertical ``QSplitter`` whose preferred height stays at its safe minimum floor.

    Keeps the enclosing ``QScrollArea`` (``widgetResizable=True``) scroll-free
    at standard heights such as ``980x720`` so the table and track-details card
    fill the viewport directly, while scrolling smoothly on compact ``640x480``
    windows whenever stacked banners reduce vertical room below the floor.
    """

    def sizeHint(self) -> QSize:
        base = super().sizeHint()
        min_h = max(self.minimumHeight(), self.minimumSizeHint().height())
        return QSize(base.width(), min_h)


class ReviewScreen(QWidget):
    """Plan review screen widget (listens to ``banner`` and ``notice``; reads ``model``)."""

    def __init__(
        self,
        controller: RunController,
        model: PlanTableModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("review_screen")
        self._controller = controller
        self._model = model
        self._banners: list[str] = []
        self._notices: list[str] = []
        self._row_flags: list[_RowFlags] = []
        self._extract_count = 0
        self._skipped_count = 0
        self._failed_count = 0
        self._attention_count = 0
        self._last_summary_snapshot: tuple[int, int, int, int, int] | None = None
        self._stale_while_hidden = False
        self._summary_dirty = False
        self._details_dirty = False
        self._cached_selected_source_row: int | None = None
        self._updating_columns = False
        self._user_resized_columns = False

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(_REFRESH_COALESCE_MS)
        self._refresh_timer.timeout.connect(self._on_refresh_timeout)

        set_accessible(
            self,
            "Plan review screen",
            "Review planned audio tracks, inspect metadata and warnings, filter rows, "
            "and choose whether to skip individual tracks before extraction.",
        )

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self._scroll_area = QScrollArea(self)
        self._scroll_area.setObjectName("review_scroll_area")
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll_area.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        set_accessible(
            self._scroll_area,
            "Plan review content",
            "Scrollable container holding the header, banners, track filters, table, and details "
            "pane.",
        )

        content = QWidget(self._scroll_area)
        content.setObjectName("review_content")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 12, 20, 10)
        layout.setSpacing(8)

        # Header strip with title, muted subtitle, and right-aligned counts chip.
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(12)

        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(2)
        self.title_label = make_heading_label("Review Extraction Plan", parent=content)
        self.title_label.setObjectName("review_title_label")
        title_col.addWidget(self.title_label)

        self._subtitle_label = make_muted_label(
            "Verify matches, warnings, and output paths before extracting.",
            scale=0.92,
            parent=content,
        )
        self._subtitle_label.setObjectName("review_subtitle_label")
        title_col.addWidget(self._subtitle_label)
        header_row.addLayout(title_col, stretch=1)

        self.counts_label = QLabel("Extract: 0 · Skipped: 0 · Failed: 0", content)
        self.counts_label.setObjectName("review_counts_chip")
        self.counts_label.setTextFormat(Qt.TextFormat.PlainText)
        self.counts_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.counts_label.setProperty("afvSurface", "chip")
        self.counts_label.setProperty("afvTone", "neutral")
        set_accessible(
            self.counts_label,
            "Plan summary counts",
            "Counts of tracks ready to extract, skipped, and failed.",
        )
        header_row.addWidget(self.counts_label, alignment=Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(header_row)

        # Batch-warning, cache-warning, and provider-notice banners.
        self.banner_widget = IconStatusBanner(content)
        layout.addWidget(self.banner_widget)

        self.notice_widget = IconStatusBanner(content)
        layout.addWidget(self.notice_widget)

        self.empty_banner = IconStatusBanner(content)
        layout.addWidget(self.empty_banner)

        # Segmented filter pill bar (relocating skip_checkbox to the inspector card
        # header prevents overflowing 640 px widths even with four-digit counts).
        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.setSpacing(8)
        filter_label = make_muted_label("Filter:", scale=0.9, word_wrap=False, parent=content)
        filter_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        filter_row.addWidget(filter_label, alignment=Qt.AlignmentFlag.AlignVCenter)

        filter_capsule = QFrame(content)
        filter_capsule.setObjectName("review_filter_capsule")
        filter_capsule.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        filter_capsule.setFrameShape(QFrame.Shape.NoFrame)
        filter_capsule.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        filter_capsule.setProperty("afvSurface", "segmented")
        capsule_layout = QHBoxLayout(filter_capsule)
        capsule_layout.setContentsMargins(3, 3, 3, 3)
        capsule_layout.setSpacing(2)

        self.filter_group = QButtonGroup(self)
        self.filter_group.setExclusive(True)

        self.filter_all_button = QPushButton("&All (0)", filter_capsule)
        self.filter_attention_button = QPushButton("Needs a&ttention (0)", filter_capsule)
        self.filter_skipped_button = QPushButton("S&kipped (0)", filter_capsule)
        self.filter_failed_button = QPushButton("&Failed (0)", filter_capsule)

        filter_specs = (
            (
                self.filter_all_button,
                "Filter all rows",
                "Show every planned track.",
                "Show all planned tracks.",
            ),
            (
                self.filter_attention_button,
                "Filter needs attention",
                "Show tracks with warnings, conflicts, failures, or uncertain matches.",
                "Show tracks with warnings, conflicts, failures, or uncertain matches.",
            ),
            (
                self.filter_skipped_button,
                "Filter skipped rows",
                "Show skipped tracks only.",
                "Show tracks skipped manually or automatically.",
            ),
            (
                self.filter_failed_button,
                "Filter failed rows",
                "Show failed tracks only.",
                "Show tracks that failed during planning.",
            ),
        )
        for btn, acc_name, acc_desc, tooltip in filter_specs:
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            apply_filter_pill_role(btn)
            set_accessible(btn, acc_name, acc_desc)
            self.filter_group.addButton(btn)
            capsule_layout.addWidget(btn)

        self.filter_all_button.setChecked(True)
        filter_row.addWidget(filter_capsule, alignment=Qt.AlignmentFlag.AlignVCenter)
        filter_row.addStretch(1)
        layout.addLayout(filter_row)

        # Quiet cue shown only when a filter yields zero rows on a non-empty plan.
        self._empty_filter_strip = QFrame(content)
        self._empty_filter_strip.setObjectName("review_empty_filter_strip")
        self._empty_filter_strip.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._empty_filter_strip.setFrameShape(QFrame.Shape.NoFrame)
        self._empty_filter_strip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._empty_filter_strip.setProperty("afvSurface", "card-inset")
        strip_layout = QHBoxLayout(self._empty_filter_strip)
        strip_layout.setContentsMargins(12, 6, 12, 6)
        strip_layout.setSpacing(8)
        strip_icon = ThemedSvgLabel("status_info", size=14, parent=self._empty_filter_strip)
        strip_layout.addWidget(strip_icon, alignment=Qt.AlignmentFlag.AlignVCenter)
        self._empty_filter_label = make_muted_label(
            "No tracks match the active filter.",
            scale=0.92,
            parent=self._empty_filter_strip,
        )
        set_accessible(
            self._empty_filter_label,
            "Active filter empty notice",
            "Indicates when no planned tracks match the active filter.",
        )
        strip_layout.addWidget(self._empty_filter_label, stretch=1)
        self._empty_filter_strip.setVisible(False)
        layout.addWidget(self._empty_filter_strip)

        # Proxy model wired directly to the shared PlanTableModel.
        self.proxy_model = PlanFilterProxyModel(self)
        self.proxy_model.setSourceModel(self._model)

        # Public splitter so MainWindow can persist review/splitter in QSettings.
        self.splitter = _ReviewSplitter(Qt.Orientation.Vertical, content)
        self.splitter.setObjectName("review_splitter")
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(8)
        set_accessible(
            self.splitter,
            "Plan review splitter",
            "Splits the plan table and the selected track details pane.",
        )

        self.table_view = QTableView(self.splitter)
        self.table_view.setObjectName("review_table_view")
        self.table_view.setModel(self.proxy_model)
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
        # Balanced initial widths summing to 744 px so Warnings stays visibly stretched
        # with zero horizontal scrollbar at the default 980x720 window width.
        self._updating_columns = True
        try:
            for col_idx, width in _BASE_COLUMN_WIDTHS:
                header.resizeSection(col_idx, width)
        finally:
            self._updating_columns = False
        header.sectionResized.connect(self._on_header_section_resized)
        self.table_view.setMinimumHeight(140)
        self.table_view.installEventFilter(self)
        table_vp = self.table_view.viewport()
        if table_vp is not None:
            table_vp.installEventFilter(self)
        set_accessible(
            self.table_view,
            "Planned tracks table",
            "Table listing status, source, destination, action, match, and warnings for each "
            "planned track. Press Space to toggle skipping the selected track.",
        )
        self.splitter.addWidget(self.table_view)

        # Elevated Track Details inspector card hosting active track readout and skip toggle.
        details_card = SectionCard(
            margins=(14, 9, 14, 10),
            spacing=7,
            parent=self.splitter,
        )
        details_card.setObjectName("review_details_card")

        inspector_header = QHBoxLayout()
        inspector_header.setContentsMargins(0, 0, 0, 0)
        inspector_header.setSpacing(8)

        details_heading = QLabel("Track Details", details_card)
        details_heading.setTextFormat(Qt.TextFormat.PlainText)
        details_heading.setProperty("afvText", "section")
        heading_font = QFont(details_heading.font())
        heading_font.setWeight(QFont.Weight.DemiBold)
        details_heading.setFont(heading_font)
        details_heading.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        inspector_header.addWidget(details_heading, alignment=Qt.AlignmentFlag.AlignVCenter)

        self._selected_track_label = MiddleElideLabel("— No track selected", details_card)
        self._selected_track_label.setObjectName("review_selected_track_label")
        self._selected_track_label.setProperty("afvText", "muted")
        self._selected_track_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        set_accessible(
            self._selected_track_label,
            "Selected track",
            "Filename of the track currently shown in the details pane.",
        )
        inspector_header.addWidget(
            self._selected_track_label,
            stretch=1,
            alignment=Qt.AlignmentFlag.AlignVCenter,
        )

        self.skip_checkbox = QCheckBox("Ski&p selected track", details_card)
        self.skip_checkbox.setObjectName("review_skip_checkbox")
        self.skip_checkbox.setEnabled(False)
        self.skip_checkbox.setToolTip("Select a track to toggle skipping.")
        set_accessible(
            self.skip_checkbox,
            "Skip selected track",
            "Toggle skipping the currently selected track without re-planning.",
        )
        inspector_header.addWidget(self.skip_checkbox, alignment=Qt.AlignmentFlag.AlignVCenter)
        details_card.body_layout.addLayout(inspector_header)
        details_card.body_layout.addWidget(make_divider(details_card))

        self.details_text = QPlainTextEdit(details_card)
        self.details_text.setObjectName("review_details_text")
        self.details_text.setReadOnly(True)
        self.details_text.setProperty("afvReadonly", True)
        self.details_text.document().setDocumentMargin(6)
        self.details_text.setMinimumHeight(56)
        self.details_text.setPlainText(
            "Select a row to inspect full metadata, warnings, and top candidates."
        )
        set_accessible(
            self.details_text,
            "Track details",
            "Full untruncated metadata, warnings, conflict notes, and top candidates for the "
            "selected row.",
        )
        details_card.body_layout.addWidget(self.details_text, stretch=1)

        self.splitter.addWidget(details_card)
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setMinimumHeight(264)
        self.splitter.setSizes([410, 150])
        layout.addWidget(self.splitter, stretch=1)

        self._scroll_area.setWidget(content)
        outer_layout.addWidget(self._scroll_area, stretch=1)

        # Sticky hairline-separated footer bar.
        footer = ActionBarFooter(self, margins=(20, 10, 20, 12), spacing=10)
        footer.setObjectName("review_footer")
        self.back_button = QPushButton("&Back", footer)
        self.back_button.setObjectName("review_back_button")
        apply_secondary_button_role(self.back_button, icon_name="arrow_left")
        self.back_button.setToolTip(
            "Discard this plan, close the active session, and return to the Inputs screen."
        )
        set_accessible(
            self.back_button,
            "Back to inputs",
            "Discard this plan, close the active session, and return to the Inputs screen.",
        )
        footer.content_layout.addWidget(self.back_button)

        self._footer_status_label = MiddleElideLabel("No tracks queued", footer)
        self._footer_status_label.setObjectName("review_footer_status")
        self._footer_status_label.setProperty("afvText", "muted")
        self._footer_status_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._footer_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.content_layout.addWidget(
            self._footer_status_label,
            stretch=1,
            alignment=Qt.AlignmentFlag.AlignVCenter,
        )

        self.start_button = QPushButton("&Start extraction", footer)
        self.start_button.setObjectName("review_start_button")
        apply_primary_button_role(self.start_button, icon_name="play")
        self.start_button.setDefault(True)
        self.start_button.setToolTip(
            "Begin extracting audio and writing tagged MP3 files for the planned tracks."
        )
        set_accessible(
            self.start_button,
            "Start extraction",
            "Begin extracting audio and writing tagged MP3 files for the planned tracks.",
        )
        footer.content_layout.addWidget(self.start_button)
        outer_layout.addWidget(footer)

        # Connect filters, controls, model, and controller using bound methods only.
        self.filter_all_button.clicked.connect(self._on_filter_all)
        self.filter_attention_button.clicked.connect(self._on_filter_attention)
        self.filter_skipped_button.clicked.connect(self._on_filter_skipped)
        self.filter_failed_button.clicked.connect(self._on_filter_failed)
        self.skip_checkbox.clicked.connect(self._on_skip_toggled)
        self.back_button.clicked.connect(self._on_back_clicked)
        self.start_button.clicked.connect(self._on_start_clicked)

        selection_model = self.table_view.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(self._on_selection_changed)

        self._model.modelReset.connect(self._on_model_changed)
        self._model.rowsInserted.connect(self._on_model_rows_changed)
        self._model.rowsRemoved.connect(self._on_model_rows_changed)
        self._model.dataChanged.connect(self._on_model_data_changed)

        self._controller.banner.connect(self._on_banner)
        self._controller.notice.connect(self._on_notice)
        self._controller.state_changed.connect(self._on_state_changed)

        # Tab order following natural visual flow across filters, table, inspector, and footer.
        tab_chain: tuple[QWidget, ...] = (
            self.filter_all_button,
            self.filter_attention_button,
            self.filter_skipped_button,
            self.filter_failed_button,
            self.table_view,
            self.skip_checkbox,
            self.details_text,
            self.back_button,
            self.start_button,
        )
        for prev_widget, next_widget in itertools.pairwise(tab_chain):
            self.setTabOrder(prev_widget, next_widget)

        self._refresh_summary()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Handle Space-to-skip in the table view and rebalance wide columns on viewport resize."""
        if watched is self.table_view.viewport() and event.type() == QEvent.Type.Resize:
            self._distribute_table_columns()
            return super().eventFilter(watched, event)
        if (
            watched is self.table_view
            and event.type() == QEvent.Type.KeyPress
            and isinstance(event, QKeyEvent)
            and event.key() == Qt.Key.Key_Space
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
            and self.skip_checkbox.isEnabled()
        ):
            self.skip_checkbox.click()
            return True
        return super().eventFilter(watched, event)

    def _distribute_table_columns(self) -> None:
        """Proportionally expand interactive text columns on wide displays without overwriting
        user-dragged widths.
        """
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

    @property
    def banners(self) -> tuple[str, ...]:
        """Return unique batch and cache warning messages shown in the banner area."""
        return tuple(self._banners)

    @property
    def notices(self) -> tuple[str, ...]:
        """Return unique notice messages shown in the notice area."""
        return tuple(self._notices)

    def active_filter(self) -> PlanFilterMode:
        """Return the currently active ``PlanFilterMode``."""
        return self.proxy_model.filter_mode

    def set_filter(self, mode: PlanFilterMode) -> None:
        """Select a table filter mode and sync the corresponding filter button."""
        if self._stale_while_hidden or self._summary_dirty:
            self.flush_updates()
        self.proxy_model.set_filter_mode(mode)
        button_map = {
            PlanFilterMode.ALL: self.filter_all_button,
            PlanFilterMode.NEEDS_ATTENTION: self.filter_attention_button,
            PlanFilterMode.SKIPPED: self.filter_skipped_button,
            PlanFilterMode.FAILED: self.filter_failed_button,
        }
        target = button_map[mode]
        with QSignalBlocker(target):
            target.setChecked(True)
        self._update_empty_filter_strip()
        self._ensure_valid_selection()
        self._cached_selected_source_row = self.selected_source_row()
        self._refresh_details()

    def clear_banners(self) -> None:
        """Clear accumulated warning banners and notices."""
        self._banners.clear()
        self._notices.clear()
        self.banner_widget.clear_message()
        self.notice_widget.clear_message()

    def selected_source_row(self) -> int | None:
        """Return the 0-based source model row currently selected, or ``None``."""
        selection_model = self.table_view.selectionModel()
        if selection_model is None:
            return None
        rows = selection_model.selectedRows()
        if not rows:
            current = self.table_view.currentIndex()
            if not current.isValid():
                return None
            source_index = self.proxy_model.mapToSource(current)
            return source_index.row() if source_index.isValid() else None
        source_index = self.proxy_model.mapToSource(rows[0])
        return source_index.row() if source_index.isValid() else None

    def _update_empty_filter_strip(self) -> None:
        total = self._model.rowCount()
        visible = self.proxy_model.rowCount()
        should_show = total > 0 and visible == 0
        if should_show:
            msg = _EMPTY_FILTER_MESSAGES.get(
                self.proxy_model.filter_mode,
                "No tracks match the active filter.",
            )
            self._empty_filter_label.setText(msg)
        if should_show and self._empty_filter_strip.isHidden():
            self._empty_filter_strip.setVisible(True)
        elif not should_show and not self._empty_filter_strip.isHidden():
            self._empty_filter_strip.setVisible(False)

    def _ensure_valid_selection(self) -> None:
        if self.proxy_model.rowCount() == 0:
            self.table_view.clearSelection()
            self._cached_selected_source_row = None
            return
        current = self.selected_source_row()
        if current is None:
            self.table_view.selectRow(0)
            current = self.selected_source_row()
        self._cached_selected_source_row = current

    def _sync_output_root(self) -> None:
        ctrl_root = self._controller.output_root
        if ctrl_root is not None and self._model.output_root != ctrl_root:
            self._model.set_output_root(ctrl_root)

    def _row_flags_for(self, row: int) -> _RowFlags:
        item = self._model.item_at(row)
        if item is None:
            return (False, False, False, False)
        outcome = self._model.outcome_at(row)
        return (
            item.action == PlanAction.EXTRACT,
            is_skipped_action(item.action),
            is_failed_action(item.action),
            is_needs_attention(item, outcome=outcome),
        )

    def _rebuild_row_counts(self) -> None:
        total = self._model.rowCount()
        flags: list[_RowFlags] = []
        extract_count = 0
        skipped_count = 0
        failed_count = 0
        attention_count = 0
        for row in range(total):
            row_flag = self._row_flags_for(row)
            flags.append(row_flag)
            if row_flag[0]:
                extract_count += 1
            if row_flag[1]:
                skipped_count += 1
            if row_flag[2]:
                failed_count += 1
            if row_flag[3]:
                attention_count += 1
        self._row_flags = flags
        self._extract_count = extract_count
        self._skipped_count = skipped_count
        self._failed_count = failed_count
        self._attention_count = attention_count

    def _apply_summary_widgets(self) -> None:
        total = len(self._row_flags)
        extract_count = self._extract_count
        attention_count = self._attention_count
        skipped_count = self._skipped_count
        failed_count = self._failed_count
        snapshot = (total, extract_count, attention_count, skipped_count, failed_count)
        if snapshot == self._last_summary_snapshot:
            self._update_empty_filter_strip()
            return
        self._last_summary_snapshot = snapshot

        self.filter_all_button.setText(f"&All ({total})")
        self.filter_attention_button.setText(f"Needs a&ttention ({attention_count})")
        self.filter_skipped_button.setText(f"S&kipped ({skipped_count})")
        self.filter_failed_button.setText(f"&Failed ({failed_count})")
        self.counts_label.setText(
            f"Extract: {extract_count} · Skipped: {skipped_count} · Failed: {failed_count}"
        )

        if total == 0:
            chip_tone = "neutral"
        elif failed_count > 0:
            chip_tone = "error"
        elif attention_count > 0:
            chip_tone = "warn"
        elif extract_count == 0:
            chip_tone = "neutral"
        else:
            chip_tone = "ok"
        set_widget_role(self.counts_label, "afvTone", chip_tone)

        if total == 0:
            self.empty_banner.set_message(
                "status_info",
                "Nothing to extract — no tracks are queued.",
                role_label="Empty plan notice",
            )
            self.start_button.setEnabled(False)
            self.start_button.setToolTip("No tracks are queued for extraction.")
            self._footer_status_label.setText("No tracks queued")
        elif extract_count == 0:
            self.empty_banner.set_message(
                "status_info",
                "Nothing to extract — every row is skipped or failed.",
                role_label="Empty plan notice",
            )
            self.start_button.setEnabled(False)
            self.start_button.setToolTip(
                "All tracks are skipped or failed. Un-skip a track to enable extraction."
            )
            self._footer_status_label.setText("Nothing queued to extract")
        else:
            self.empty_banner.clear_message()
            self.start_button.setEnabled(True)
            noun = "track" if extract_count == 1 else "tracks"
            self.start_button.setToolTip(
                f"Begin extracting audio and writing tagged MP3 files for {extract_count} {noun}."
            )
            self._footer_status_label.setText(f"{extract_count} {noun} ready to extract")

        self._update_empty_filter_strip()

    def _schedule_refresh(self) -> None:
        if not self.isVisible():
            return
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def flush_updates(self) -> None:
        """Apply any pending deferred counts, banner, button, and detail updates immediately."""
        if self._refresh_timer.isActive():
            self._refresh_timer.stop()
        self._sync_output_root()
        if self._stale_while_hidden:
            self._rebuild_row_counts()
            self._stale_while_hidden = False
            self._summary_dirty = True
            self._details_dirty = True
        if self._summary_dirty:
            prev_selected = self._cached_selected_source_row
            self._apply_summary_widgets()
            self._ensure_valid_selection()
            if self._cached_selected_source_row != prev_selected:
                self._details_dirty = True
            self._summary_dirty = False
        if self._details_dirty:
            self._refresh_details()
            self._details_dirty = False

    def _refresh_summary(self) -> None:
        if self._refresh_timer.isActive():
            self._refresh_timer.stop()
        self._sync_output_root()
        self._rebuild_row_counts()
        self._stale_while_hidden = False
        self._summary_dirty = False
        self._details_dirty = False
        self._apply_summary_widgets()
        self._ensure_valid_selection()
        self._refresh_details()

    def showEvent(self, event: QShowEvent) -> None:
        """Flush any updates deferred while the screen was hidden and rebalance columns."""
        super().showEvent(event)
        if self._stale_while_hidden or self._summary_dirty or self._details_dirty:
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

    def _refresh_details(self) -> None:
        row = self.selected_source_row()
        self._cached_selected_source_row = row
        item = self._model.item_at(row) if row is not None else None
        if item is None:
            self._selected_track_label.setText("— No track selected")
            self._selected_track_label.set_full_tooltip(None)
            with QSignalBlocker(self.skip_checkbox):
                self.skip_checkbox.setChecked(False)
                self.skip_checkbox.setEnabled(False)
                self.skip_checkbox.setToolTip("Select a track to toggle skipping.")
            self.details_text.setPlainText(
                "Select a row to inspect full metadata, warnings, and top candidates."
            )
            return

        source_path = item.media_input.path
        self._selected_track_label.setText(f"— {source_path.name}")
        self._selected_track_label.set_full_tooltip(str(source_path))

        root = self._model.output_root or self._controller.output_root
        self.details_text.setPlainText(_format_item_details(item, root))

        toggleable = item.action in (PlanAction.EXTRACT, PlanAction.SKIP_USER)
        with QSignalBlocker(self.skip_checkbox):
            self.skip_checkbox.setEnabled(toggleable)
            self.skip_checkbox.setChecked(item.action == PlanAction.SKIP_USER)
            if toggleable:
                self.skip_checkbox.setToolTip(
                    "Mark or unmark this track as skipped for this extraction run (Space)."
                )
            elif is_skipped_action(item.action):
                self.skip_checkbox.setToolTip(
                    "This track was skipped automatically during planning."
                )
            else:
                self.skip_checkbox.setToolTip(
                    "This track failed during planning and cannot be extracted."
                )

    @Slot()
    def _on_filter_all(self) -> None:
        self.set_filter(PlanFilterMode.ALL)

    @Slot()
    def _on_filter_attention(self) -> None:
        self.set_filter(PlanFilterMode.NEEDS_ATTENTION)

    @Slot()
    def _on_filter_skipped(self) -> None:
        self.set_filter(PlanFilterMode.SKIPPED)

    @Slot()
    def _on_filter_failed(self) -> None:
        self.set_filter(PlanFilterMode.FAILED)

    @Slot(bool)
    def _on_skip_toggled(self, checked: bool) -> None:
        row = self.selected_source_row()
        if row is None:
            return
        self._controller.set_row_skipped(row, bool(checked))
        if self._stale_while_hidden or self._summary_dirty or self._details_dirty:
            self.flush_updates()

    @Slot()
    def _on_back_clicked(self) -> None:
        self._controller.back()

    @Slot()
    def _on_start_clicked(self) -> None:
        self._controller.start_run()

    @Slot(QItemSelection, QItemSelection)
    def _on_selection_changed(
        self,
        selected: QItemSelection,
        deselected: QItemSelection,
    ) -> None:
        _ = selected, deselected
        self._refresh_details()

    @Slot()
    def _on_model_changed(self) -> None:
        self._refresh_summary()

    @Slot(QModelIndex, int, int)
    def _on_model_rows_changed(
        self,
        parent: QModelIndex | QPersistentModelIndex,
        first: int,
        last: int,
    ) -> None:
        _ = parent, first, last
        self._refresh_summary()

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
            self._refresh_summary()
            return
        if not top_left.isValid() or not bottom_right.isValid():
            self._stale_while_hidden = True
            self._schedule_refresh()
            return
        if self._stale_while_hidden or len(self._row_flags) != row_count:
            self._stale_while_hidden = True
            self._schedule_refresh()
            return

        first = max(0, top_left.row())
        last = min(row_count - 1, bottom_right.row())
        dest_only = (
            top_left.column() == COL_DESTINATION and bottom_right.column() == COL_DESTINATION
        )
        if not dest_only:
            for row in range(first, last + 1):
                old = self._row_flags[row]
                new = self._row_flags_for(row)
                if new != old:
                    self._row_flags[row] = new
                    self._extract_count += int(new[0]) - int(old[0])
                    self._skipped_count += int(new[1]) - int(old[1])
                    self._failed_count += int(new[2]) - int(old[2])
                    self._attention_count += int(new[3]) - int(old[3])
                    self._summary_dirty = True

        selected_row = self._cached_selected_source_row
        if selected_row is not None and first <= selected_row <= last:
            self._details_dirty = True

        if self._summary_dirty or self._details_dirty:
            self._schedule_refresh()

    @Slot(object)
    def _on_state_changed(self, state: object) -> None:
        if state == ControllerState.PLANNING:
            self.clear_banners()
        elif state == ControllerState.REVIEW:
            self._refresh_summary()

    @Slot(str)
    def _on_banner(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        if cleaned not in self._banners:
            self._banners.append(cleaned)
        self.banner_widget.set_message(
            "status_warn",
            "\n".join(self._banners),
            role_label="Plan warning",
        )

    @Slot(str)
    def _on_notice(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        if cleaned not in self._notices:
            self._notices.append(cleaned)
        lowered = cleaned.casefold()
        icon = (
            "status_warn"
            if any(token in lowered for token in ("unreachable", "unavailable", "warning", "error"))
            else "status_info"
        )
        self.notice_widget.set_message(
            icon,
            "\n".join(self._notices),
            role_label="Plan notice",
        )
