"""Window-modal dialog for selecting an audio stream on multi-track input files."""

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

from audio_from_video.gui.bridge import PromptKind, StreamPrompt
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

if TYPE_CHECKING:
    from audio_from_video.gui.bridge import PromptRequest
    from audio_from_video.models import AudioStreamInfo

_STREAM_COLUMNS = ("Index", "Language", "Codec", "Channels", "Sample rate", "Title", "Default")
_COMPACT_COLUMNS = (0, 1, 2, 3, 4, 6)
_TITLE_COL_INDEX = 5
_MIN_TITLE_COL_WIDTH = 180
_BUTTON_ICON_SIZE = QSize(14, 14)
_TABLE_ICON_SIZE = QSize(15, 15)
_KEY_MODIFIER_MASK = (
    Qt.KeyboardModifier.ShiftModifier
    | Qt.KeyboardModifier.ControlModifier
    | Qt.KeyboardModifier.AltModifier
    | Qt.KeyboardModifier.MetaModifier
)


def _format_channels(stream: AudioStreamInfo) -> str:
    """Format a stream's channel count and optional layout label."""
    if stream.channels <= 0:
        return "?"
    if stream.channel_layout:
        return f"{stream.channels} ({stream.channel_layout})"
    return str(stream.channels)


def _stream_row_matches_default(stream: AudioStreamInfo, default: AudioStreamInfo) -> bool:
    """Return True when ``stream`` identifies the prompt's default stream."""
    return stream == default or stream.audio_index == default.audio_index


def _format_stream_count(count: int) -> str:
    """Return a singular or plural stream count label."""
    if count == 0:
        return "No streams"
    if count == 1:
        return "1 stream"
    return f"{count} streams"


def _format_stream_summary(stream: AudioStreamInfo, *, is_default: bool) -> str:
    """Build a single-line plain-text summary of the selected audio stream."""
    title = stream.title.strip() if stream.title else "Untitled stream"
    parts: list[str] = [
        f"Index {stream.audio_index}",
        stream.language or "und",
        stream.codec_name or "unknown",
        f"{_format_channels(stream)} ch",
    ]
    if stream.sample_rate:
        parts.append(f"{stream.sample_rate} Hz")
    if stream.is_commentary:
        parts.append("Commentary")
    if is_default:
        parts.append("Default")
    return f"{title}  {EM_DASH}  {' · '.join(parts)}"


def _make_caption_font(widget: QWidget) -> QFont:
    """Return a slightly smaller font relative to ``widget`` for muted captions."""
    return make_heading_font(widget, scale=0.92, bold=False)


class StreamDialog(QDialog):
    """Prompt dialog for choosing an ``AudioStreamInfo`` (opened with ``open()``)."""

    answered = Signal(int, object)  # request_id, AudioStreamInfo

    def __init__(self, request: PromptRequest, parent: QWidget) -> None:
        super().__init__(parent)
        if request.kind != PromptKind.STREAM:
            raise ValueError(f"StreamDialog requires PromptKind.STREAM, got {request.kind!r}")
        if not isinstance(request.payload, StreamPrompt):
            raise TypeError(
                f"StreamDialog requires a StreamPrompt payload, got {request.payload!r}"
            )

        self._request = request
        self._payload: StreamPrompt = request.payload
        self._streams: tuple[AudioStreamInfo, ...] = self._payload.streams
        self._answered = False
        self._withdrawn = False

        self.setObjectName("stream_dialog")
        self.setWindowTitle("Select Audio Stream")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumWidth(520)
        self.setMinimumHeight(385)
        self.resize(780, 425)
        configure_accessible(
            self,
            "Select audio stream",
            f"Choose which audio stream to extract from {request.media_input.path.name}.",
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(12)

        # -- Header (status emblem, title, and subtitle) -------------------------
        self._heading_label = QLabel("Multiple audio streams found", self)
        self._heading_label.setObjectName("stream_heading_label")
        self._heading_label.setFont(make_heading_font(self._heading_label, scale=1.28))
        self._heading_label.setProperty("afvText", "title")
        self._heading_label.setTextFormat(Qt.TextFormat.PlainText)
        self._heading_label.setWordWrap(True)
        configure_accessible(
            self._heading_label,
            "Dialog heading",
            "Multiple audio streams found in this media file.",
        )

        self._instruction_label = QLabel(
            "Select the audio stream to extract, or accept the highlighted default.",
            self,
        )
        self._instruction_label.setObjectName("stream_subtitle_label")
        self._instruction_label.setFont(_make_caption_font(self._instruction_label))
        self._instruction_label.setWordWrap(True)
        self._instruction_label.setTextFormat(Qt.TextFormat.PlainText)
        configure_accessible(
            self._instruction_label,
            "Dialog instructions",
            "Select the audio stream to extract, or accept the highlighted default.",
        )

        header_row, self._header_icon = make_dialog_header(
            self,
            icon_name="status_info",
            heading_label=self._heading_label,
            subtitle_label=self._instruction_label,
            icon_size=26,
        )
        configure_accessible(
            self._header_icon,
            "Audio stream icon",
            "Informational badge indicating multiple audio streams.",
        )
        root.addLayout(header_row)

        # -- Context card showing source filename and stream count ---------------
        context_card = make_context_card(self, object_name="stream_context_card")
        context_row = QHBoxLayout(context_card)
        context_row.setContentsMargins(12, 8, 12, 8)
        context_row.setSpacing(12)

        self._file_label = MiddleElideLabel(f"File: {request.media_input.path.name}", context_card)
        self._file_label.setObjectName("stream_file_label")
        self._file_label.set_full_tooltip(str(request.media_input.path))
        configure_accessible(
            self._file_label,
            "Source file",
            f"Source file name: {request.media_input.path.name}.",
        )
        context_row.addWidget(self._file_label, stretch=1)

        self._count_label = QLabel(_format_stream_count(len(self._streams)), context_card)
        self._count_label.setObjectName("stream_count_label")
        self._count_label.setFont(_make_caption_font(self._count_label))
        self._count_label.setProperty("afvText", "caption")
        self._count_label.setTextFormat(Qt.TextFormat.PlainText)
        configure_accessible(
            self._count_label,
            "Stream count",
            f"Found {_format_stream_count(len(self._streams)).lower()}.",
        )
        context_row.addWidget(self._count_label)
        repolish(context_card, include_children=True)
        root.addWidget(context_card)

        # -- Empty-state banner when no selectable streams were provided ---------
        self._empty_banner, self._empty_banner_label = make_notice_banner(
            self,
            object_name="stream_empty_banner",
            pixmap_kind=QStyle.StandardPixmap.SP_MessageBoxInformation,
        )
        empty_msg = (
            "No audio streams are listed to choose from. "
            f"Choose Use default to continue with stream index {self._payload.default.audio_index}."
        )
        self._empty_banner_label.setText(empty_msg)
        configure_accessible(self._empty_banner, "No audio streams notice", empty_msg)
        configure_accessible(self._empty_banner_label, "No audio streams message", empty_msg)
        self._empty_banner.setVisible(not self._streams)
        root.addWidget(self._empty_banner)

        # -- Audio stream table --------------------------------------------------
        self._table = QTableWidget(self)
        self._table.setObjectName("stream_table")
        self._table.setProperty("afvDensity", "compact")
        repolish(self._table, include_children=True)
        self._table.setMinimumHeight(136)
        self._table.setIconSize(_TABLE_ICON_SIZE)
        configure_readonly_table(
            self._table,
            _STREAM_COLUMNS,
            accessible_name="Audio streams",
            accessible_description=(
                "Available audio streams showing index, language, codec, channels, "
                "sample rate, title, and default flag."
            ),
        )
        self._populate_table()
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.cellDoubleClicked.connect(self._on_row_double_clicked)
        self._table.installEventFilter(self)
        root.addWidget(self._table, stretch=1)

        # -- Selected-stream detail readout strip --------------------------------
        self._detail_card = QFrame(self)
        self._detail_card.setObjectName("stream_detail_card")
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
        self._detail_label.setObjectName("stream_detail_label")
        self._detail_label.setFont(_make_caption_font(self._detail_label))
        self._detail_label.setProperty("afvText", "muted")
        self._detail_label.setTextFormat(Qt.TextFormat.PlainText)
        self._detail_label.setWordWrap(True)
        self._detail_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._detail_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        configure_accessible(
            self._detail_label,
            "Selected stream details",
            "Full metadata summary for the highlighted audio stream.",
        )
        detail_layout.addWidget(self._detail_label, stretch=1)
        repolish(self._detail_card, include_children=True)
        self._detail_card.setVisible(bool(self._streams))
        root.addWidget(self._detail_card)

        root.addWidget(make_divider_line(self))

        # -- Footer action bar ---------------------------------------------------
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(8)

        self._hint_label = QLabel("Double-click or press Enter to confirm.", self)
        self._hint_label.setObjectName("stream_hint_label")
        self._hint_label.setFont(_make_caption_font(self._hint_label))
        self._hint_label.setProperty("afvText", "caption")
        self._hint_label.setWordWrap(True)
        self._hint_label.setTextFormat(Qt.TextFormat.PlainText)
        configure_accessible(
            self._hint_label,
            "Shortcut hint",
            "Double-click a row or press Enter to select the highlighted audio stream.",
        )
        self._hint_label.setVisible(bool(self._streams))
        if self._streams:
            buttons.addWidget(self._hint_label, stretch=1)
        else:
            buttons.addWidget(self._hint_label)
            buttons.addStretch(1)

        has_streams = bool(self._streams)
        self._use_default_button = QPushButton("Use &default", self)
        self._use_default_button.setObjectName("stream_use_default_button")
        self._use_default_button.setAutoDefault(False)
        self._use_default_button.setDefault(not has_streams)
        self._use_default_button.setToolTip(
            f"Use the default audio stream (index {self._payload.default.audio_index})."
        )
        configure_accessible(
            self._use_default_button,
            "Use default stream",
            "Accept the recommended default audio stream.",
        )
        set_button_role(self._use_default_button, "secondary" if has_streams else "primary")
        self._use_default_button.clicked.connect(self._on_use_default)
        buttons.addWidget(self._use_default_button)

        self._use_selected_button = QPushButton("&Use selected stream", self)
        self._use_selected_button.setObjectName("stream_use_selected_button")
        self._use_selected_button.setAutoDefault(False)
        self._use_selected_button.setDefault(has_streams)
        self._use_selected_button.setEnabled(has_streams)
        self._use_selected_button.setIconSize(_BUTTON_ICON_SIZE)
        self._use_selected_button.setToolTip("Extract the highlighted audio stream.")
        configure_accessible(
            self._use_selected_button,
            "Use selected stream",
            "Extract the currently selected audio stream.",
        )
        set_button_role(self._use_selected_button, "primary" if has_streams else "secondary")
        self._use_selected_button.setIcon(dialog_button_icon("check_mark", on_primary=has_streams))
        self._use_selected_button.clicked.connect(self._on_use_selected)
        buttons.addWidget(self._use_selected_button)

        root.addLayout(buttons)

        self._update_detail_readout()

        self.setTabOrder(self._table, self._use_selected_button)
        self.setTabOrder(self._use_selected_button, self._use_default_button)
        if self._streams:
            self._table.setFocus()
        else:
            self._use_default_button.setFocus()

    def _populate_table(self) -> None:
        self._table.setRowCount(len(self._streams))
        default_row = 0
        default_found = False
        align_right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        bold_font = QFont(self._table.font())
        bold_font.setBold(True)
        default_icon = load_icon("status_ok")
        commentary_icon = load_icon("commentary")

        for row, stream in enumerate(self._streams):
            is_default = _stream_row_matches_default(stream, self._payload.default)
            if is_default and not default_found:
                default_row = row
                default_found = True

            index_text = str(stream.audio_index)
            index_tip = f"Audio index {stream.audio_index} (container stream {stream.stream_index})"
            language_text = stream.language or "und"
            codec_text = stream.codec_name or "unknown"
            channels_text = _format_channels(stream)
            rate_text = f"{stream.sample_rate} Hz" if stream.sample_rate else EM_DASH
            title_text = stream.title or EM_DASH
            title_tip = stream.title or "No stream title"
            if stream.is_commentary:
                title_tip = f"{title_tip} (commentary)"
            default_text = "Default" if is_default else ""

            row_items = (
                make_table_item(index_text, tooltip=index_tip, alignment=align_right),
                make_table_item(language_text),
                make_table_item(codec_text),
                make_table_item(channels_text, alignment=align_right),
                make_table_item(rate_text, alignment=align_right),
                make_table_item(title_text, tooltip=title_tip),
                make_table_item(default_text, tooltip="Recommended default" if is_default else ""),
            )
            if stream.is_commentary:
                row_items[5].setIcon(commentary_icon)
            if is_default:
                row_items[6].setIcon(default_icon)
                for item in row_items:
                    item.setFont(bold_font)
            for col, item in enumerate(row_items):
                self._table.setItem(row, col, item)

        header = self._table.horizontalHeader()
        header.setMinimumSectionSize(32)
        for col in range(len(_STREAM_COLUMNS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
        self._table.resizeColumnsToContents()
        for col in _COMPACT_COLUMNS:
            natural = max(header.sectionSize(col), 32)
            if col == 6:
                natural += 12
            header.resizeSection(col, natural)
        self._sync_title_column_width()

        if self._streams:
            self._table.selectRow(default_row)
            self._table.setCurrentCell(default_row, 0)

    def _sync_title_column_width(self) -> None:
        """Keep the ``Title`` column wide enough to remain legible across dialog widths."""
        viewport_w = self._table.viewport().width()
        if viewport_w <= 0:
            viewport_w = max(0, self.width() - 42)
        other_width = sum(self._table.columnWidth(col) for col in _COMPACT_COLUMNS)
        title_w = max(_MIN_TITLE_COL_WIDTH, viewport_w - other_width)
        self._table.setColumnWidth(_TITLE_COL_INDEX, title_w)

    def _update_detail_readout(self) -> None:
        """Show full unelided metadata for the currently selected audio stream."""
        if not self._streams:
            self._detail_label.setText("")
            return
        row = self._table.currentRow()
        if not (0 <= row < len(self._streams)):
            row = 0
        stream = self._streams[row]
        is_default = _stream_row_matches_default(stream, self._payload.default)
        summary = _format_stream_summary(stream, is_default=is_default)
        self._detail_label.setText(summary)
        self._detail_label.setToolTip(summary)
        configure_accessible(
            self._detail_label,
            "Selected stream details",
            summary,
        )

    @Slot()
    def _on_selection_changed(self) -> None:
        self._update_detail_readout()

    def showEvent(self, event: QShowEvent) -> None:
        """Size the Title column to match the live viewport once shown."""
        super().showEvent(event)
        self._sync_title_column_width()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Reflow the Title column cleanly while protecting its minimum width."""
        super().resizeEvent(event)
        self._sync_title_column_width()

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
                dialog_button_icon("check_mark", on_primary=bool(self._streams))
            )

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Confirm the highlighted stream on Return/Enter while the table has focus."""
        if (
            watched is self._table
            and event.type() == QEvent.Type.KeyPress
            and isinstance(event, QKeyEvent)
            and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and not (event.modifiers() & _KEY_MODIFIER_MASK)
            and self._streams
        ):
            self._on_use_selected()
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def _emit_answer(self, stream: AudioStreamInfo, *, accepted: bool) -> None:
        if self._answered or self._withdrawn:
            return
        self._answered = True
        if accepted:
            super().accept()
        else:
            super().reject()
        self.answered.emit(self._request.request_id, stream)

    @Slot()
    def _on_use_selected(self) -> None:
        row = self._table.currentRow()
        if 0 <= row < len(self._streams):
            self._emit_answer(self._streams[row], accepted=True)
        else:
            self._emit_answer(self._payload.default, accepted=True)

    @Slot()
    def _on_use_default(self) -> None:
        self._emit_answer(self._payload.default, accepted=True)

    @Slot(int, int)
    def _on_row_double_clicked(self, row: int, column: int) -> None:
        _ = column
        if 0 <= row < len(self._streams):
            self._emit_answer(self._streams[row], accepted=True)

    def withdraw(self) -> None:
        """Close the dialog programmatically without emitting ``answered``."""
        if self._withdrawn:
            return
        self._withdrawn = True
        if self.isVisible():
            super().reject()

    def reject(self) -> None:
        """Handle Escape or window dismissal by choosing the safe default stream."""
        if self._withdrawn or self._answered:
            super().reject()
            return
        self._emit_answer(self._payload.default, accepted=False)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Emit the safe default stream once if closed directly while open."""
        if self._withdrawn or self._answered:
            event.accept()
            return
        if self.isVisible():
            self._emit_answer(self._payload.default, accepted=False)
        event.accept()
