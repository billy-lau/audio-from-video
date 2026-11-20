"""Window-modal dialog for selecting an audio stream on multi-track input files."""

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

from audio_from_video.gui.bridge import PromptKind, StreamPrompt
from audio_from_video.gui.dialogs._common import (
    EM_DASH,
    MiddleElideLabel,
    configure_accessible,
    configure_readonly_table,
    make_heading_font,
    make_table_item,
)

if TYPE_CHECKING:
    from audio_from_video.gui.bridge import PromptRequest
    from audio_from_video.models import AudioStreamInfo

_STREAM_COLUMNS = ("Index", "Language", "Codec", "Channels", "Sample rate", "Title", "Default")


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
        self.resize(680, 340)
        configure_accessible(
            self,
            "Select audio stream",
            f"Choose which audio stream to extract from {request.media_input.path.name}.",
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        self._heading_label = QLabel("Multiple audio streams found", self)
        self._heading_label.setObjectName("stream_heading_label")
        self._heading_label.setFont(make_heading_font(self._heading_label))
        self._heading_label.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self._heading_label)

        self._file_label = MiddleElideLabel(f"File: {request.media_input.path.name}", self)
        self._file_label.setObjectName("stream_file_label")
        self._file_label.set_full_tooltip(str(request.media_input.path))
        configure_accessible(
            self._file_label,
            "Source file",
            f"Source file name: {request.media_input.path.name}.",
        )
        root.addWidget(self._file_label)

        self._instruction_label = QLabel(
            "Select the audio stream to extract, or accept the highlighted default.",
            self,
        )
        self._instruction_label.setWordWrap(True)
        self._instruction_label.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self._instruction_label)

        self._table = QTableWidget(self)
        self._table.setObjectName("stream_table")
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
        self._table.cellDoubleClicked.connect(self._on_row_double_clicked)
        root.addWidget(self._table, stretch=1)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch(1)

        self._use_default_button = QPushButton("Use &default", self)
        self._use_default_button.setObjectName("stream_use_default_button")
        self._use_default_button.setAutoDefault(False)
        self._use_default_button.setToolTip(
            f"Use the default audio stream (index {self._payload.default.audio_index})."
        )
        configure_accessible(
            self._use_default_button,
            "Use default stream",
            "Accept the recommended default audio stream.",
        )
        self._use_default_button.clicked.connect(self._on_use_default)
        buttons.addWidget(self._use_default_button)

        self._use_selected_button = QPushButton("&Use selected stream", self)
        self._use_selected_button.setObjectName("stream_use_selected_button")
        self._use_selected_button.setDefault(True)
        self._use_selected_button.setEnabled(bool(self._streams))
        self._use_selected_button.setToolTip("Extract the highlighted audio stream.")
        configure_accessible(
            self._use_selected_button,
            "Use selected stream",
            "Extract the currently selected audio stream.",
        )
        self._use_selected_button.clicked.connect(self._on_use_selected)
        buttons.addWidget(self._use_selected_button)

        root.addLayout(buttons)

        self.setTabOrder(self._table, self._use_selected_button)
        self.setTabOrder(self._use_selected_button, self._use_default_button)
        if self._streams:
            self._table.setFocus()

    def _populate_table(self) -> None:
        self._table.setRowCount(len(self._streams))
        default_row = 0
        default_found = False
        align_right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

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

            self._table.setItem(
                row, 0, make_table_item(index_text, tooltip=index_tip, alignment=align_right)
            )
            self._table.setItem(row, 1, make_table_item(language_text))
            self._table.setItem(row, 2, make_table_item(codec_text))
            self._table.setItem(row, 3, make_table_item(channels_text, alignment=align_right))
            self._table.setItem(row, 4, make_table_item(rate_text, alignment=align_right))
            self._table.setItem(row, 5, make_table_item(title_text, tooltip=title_tip))
            self._table.setItem(row, 6, make_table_item(default_text))

        header = self._table.horizontalHeader()
        for col in (0, 1, 2, 3, 4, 6):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)

        if self._streams:
            self._table.selectRow(default_row)
            self._table.setCurrentCell(default_row, 0)

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
