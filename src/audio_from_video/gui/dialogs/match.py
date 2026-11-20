"""Window-modal dialog for resolving uncertain, unmatched, or manual-confirm tracks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QObject, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QFont, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QStyle,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from audio_from_video.gui.bridge import MatchPrompt, PromptKind
from audio_from_video.gui.dialogs._common import (
    EM_DASH,
    MiddleElideLabel,
    configure_accessible,
    configure_readonly_table,
    is_text_input_widget,
    make_heading_font,
    make_notice_banner,
    make_table_item,
)
from audio_from_video.models import Candidate, MatchDecision, ParsedHints
from audio_from_video.planner import CandidateAction, CandidatePromptResponse
from audio_from_video.presentation import (
    MAX_DISPLAY_CANDIDATES,
    build_manual_metadata,
    format_candidate_date,
    format_candidate_track,
)

if TYPE_CHECKING:
    from audio_from_video.gui.bridge import PromptRequest

_CANDIDATE_COLUMNS = (
    "#",
    "Score",
    "Artist",
    "Title",
    "Album",
    "Date",
    "Track",
    "Source",
)

_DIGIT_KEYS: dict[Qt.Key, int] = {
    Qt.Key.Key_1: 1,
    Qt.Key.Key_2: 2,
    Qt.Key.Key_3: 3,
    Qt.Key.Key_4: 4,
    Qt.Key.Key_5: 5,
    Qt.Key.Key_6: 6,
    Qt.Key.Key_7: 7,
    Qt.Key.Key_8: 8,
    Qt.Key.Key_9: 9,
}

# Allow Shift so AZERTY and similar keyboard layouts that require Shift for
# digits 1 to 9 can still trigger candidate shortcuts.
_DISALLOWED_MODIFIERS = (
    Qt.KeyboardModifier.ControlModifier
    | Qt.KeyboardModifier.AltModifier
    | Qt.KeyboardModifier.MetaModifier
)

_KEY_MODIFIER_MASK = (
    Qt.KeyboardModifier.ShiftModifier
    | Qt.KeyboardModifier.ControlModifier
    | Qt.KeyboardModifier.AltModifier
    | Qt.KeyboardModifier.MetaModifier
)


def _reason_header(decision: MatchDecision, *, has_candidates: bool) -> tuple[str, str]:
    """Return a heading and brief explanation matching ``MatchDecision``."""
    if decision == MatchDecision.AUTO_ACCEPT:
        return (
            "Confirm match",
            (
                "Review the matched candidate below, pick another candidate, search again, "
                "enter custom tags, or use fallback metadata."
            ),
        )
    if decision == MatchDecision.AMBIGUOUS:
        return (
            "Uncertain match",
            (
                "Pick the matching candidate, search again, enter custom tags, "
                "use fallback metadata, or skip this file."
            ),
        )
    if decision == MatchDecision.SKIPPED:
        return (
            "Hintless file (no lookup run)",
            (
                "No usable filename or container hints were found. Search again, "
                "enter custom tags, use fallback metadata, or skip this file."
            ),
        )
    if has_candidates:
        return (
            "No confident match",
            (
                "No candidate passed automatic confirmation. Pick a candidate below, "
                "search again, enter custom tags, or use fallback metadata."
            ),
        )
    return (
        "No match found",
        (
            "No candidates were returned. Search again, enter custom tags, "
            "use fallback metadata, or skip this file."
        ),
    )


def _format_hints_summary(hints: ParsedHints) -> str:
    """Build a compact plain-text summary of parsed filename and container hints."""
    parts: list[str] = []
    if hints.artist:
        parts.append(f"Artist: {hints.artist}")
    if hints.title:
        parts.append(f"Title: {hints.title}")
    if hints.album:
        parts.append(f"Album: {hints.album}")
    if hints.track_number is not None:
        if hints.track_total:
            parts.append(f"Track: {hints.track_number}/{hints.track_total}")
        else:
            parts.append(f"Track: {hints.track_number}")
    if hints.date:
        parts.append(f"Date: {hints.date}")
    if not parts:
        return "Parsed hints: none"
    return "Parsed hints: " + "  |  ".join(parts)


def _parse_optional_track_number(raw: str) -> tuple[int | None, str | None]:
    """Parse an optional positive integer track number and return ``(value, error)``."""
    cleaned = raw.strip()
    if not cleaned:
        return None, None
    if cleaned.isascii() and cleaned.isdigit():
        num = int(cleaned)
        if num >= 1:
            return num, None
    return None, "Track number must be a positive integer."


def _candidate_artist_tooltip(cand: Candidate) -> str:
    """Return a rich plain-text tooltip for a candidate's artist field."""
    if cand.featuring:
        feat = ", ".join(cand.featuring)
        return f"{cand.artist} (feat. {feat})"
    return cand.artist


def _set_button_emphasized(button: QPushButton, emphasized: bool) -> None:
    """Apply or clear bold emphasis derived from the system font (no hard-coded sizes)."""
    font = QFont(button.font())
    font.setBold(emphasized)
    button.setFont(font)


class MatchDialog(QDialog):
    """Single-instance-per-file match dialog returning ``CandidatePromptResponse``.

    Stays open in a busy state after a ``MANUAL_SEARCH`` answer so the next
    ``MATCH`` prompt for the same file can reuse and refill this dialog.
    """

    answered = Signal(int, object)  # request_id, CandidatePromptResponse

    def __init__(self, request: PromptRequest, parent: QWidget) -> None:
        super().__init__(parent)
        if request.kind != PromptKind.MATCH:
            raise ValueError(f"MatchDialog requires PromptKind.MATCH, got {request.kind!r}")
        if not isinstance(request.payload, MatchPrompt):
            raise TypeError(f"MatchDialog requires a MatchPrompt payload, got {request.payload!r}")

        self._request: PromptRequest = request
        self._payload: MatchPrompt = request.payload
        self._candidates: tuple[Candidate, ...] = ()
        self._busy = False
        self._answered = False
        self._withdrawn = False
        self._notice_while_busy = False

        self.setObjectName("match_dialog")
        self.setWindowTitle("Resolve Match")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumWidth(680)
        self.resize(860, 520)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        # -- Header (reason, file name, and parsed hints) -------------------------
        self._reason_label = QLabel(self)
        self._reason_label.setObjectName("match_reason_label")
        self._reason_label.setFont(make_heading_font(self._reason_label))
        self._reason_label.setTextFormat(Qt.TextFormat.PlainText)
        self._reason_label.setWordWrap(True)
        root.addWidget(self._reason_label)

        self._subtitle_label = QLabel(self)
        self._subtitle_label.setObjectName("match_subtitle_label")
        self._subtitle_label.setTextFormat(Qt.TextFormat.PlainText)
        self._subtitle_label.setWordWrap(True)
        root.addWidget(self._subtitle_label)

        self._file_label = MiddleElideLabel("", self)
        self._file_label.setObjectName("match_file_label")
        configure_accessible(self._file_label, "Source file", "Media file being matched.")
        root.addWidget(self._file_label)

        self._hints_label = MiddleElideLabel("", self)
        self._hints_label.setObjectName("match_hints_label")
        configure_accessible(
            self._hints_label,
            "Parsed hints",
            "Metadata hints parsed from the filename and container tags.",
        )
        root.addWidget(self._hints_label)

        # -- Inline notice banner (hidden until show_notice is called) ------------
        self._notice_banner, self._notice_label = make_notice_banner(
            self,
            object_name="match_notice_banner",
            pixmap_kind=QStyle.StandardPixmap.SP_MessageBoxWarning,
        )
        configure_accessible(
            self._notice_banner,
            "Match notice",
            "Inline notice displayed inside the match dialog.",
        )
        root.addWidget(self._notice_banner)

        # -- Busy searching bar (hidden unless set_busy(True)) --------------------
        self._busy_frame = QFrame(self)
        self._busy_frame.setObjectName("match_busy_bar")
        self._busy_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._busy_frame.setFrameShadow(QFrame.Shadow.Plain)
        busy_layout = QHBoxLayout(self._busy_frame)
        busy_layout.setContentsMargins(10, 6, 10, 6)
        busy_layout.setSpacing(10)

        self._busy_progress = QProgressBar(self._busy_frame)
        self._busy_progress.setObjectName("match_busy_progress")
        self._busy_progress.setRange(0, 0)
        self._busy_progress.setTextVisible(False)
        self._busy_progress.setMaximumWidth(140)
        configure_accessible(
            self._busy_progress,
            "Search progress",
            "Manual online search in progress.",
        )
        busy_layout.addWidget(self._busy_progress)

        self._busy_label = QLabel("Searching…", self._busy_frame)
        self._busy_label.setObjectName("match_busy_label")
        self._busy_label.setTextFormat(Qt.TextFormat.PlainText)
        configure_accessible(
            self._busy_label,
            "Searching status",
            "Searching online providers for matching tracks.",
        )
        busy_layout.addWidget(self._busy_label, stretch=1)

        self.cancel_run_button = QPushButton("&Cancel run", self._busy_frame)
        self.cancel_run_button.setObjectName("match_cancel_run_button")
        self.cancel_run_button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.cancel_run_button.setAutoDefault(False)
        self.cancel_run_button.setDefault(False)
        self.cancel_run_button.setToolTip(
            "Stop the manual online search and cancel the current planning run."
        )
        configure_accessible(
            self.cancel_run_button,
            "Cancel run",
            "Cancel the active manual search and stop planning.",
        )
        self.cancel_run_button.setVisible(False)
        self.cancel_run_button.setEnabled(False)
        busy_layout.addWidget(self.cancel_run_button)

        self._busy_frame.setVisible(False)
        root.addWidget(self._busy_frame)

        # -- Candidate table or empty state ("No candidates") ---------------------
        self._candidates_stack = QStackedWidget(self)
        self._candidates_stack.setObjectName("match_candidates_stack")

        self._table = QTableWidget(self._candidates_stack)
        self._table.setObjectName("match_candidates_table")
        configure_readonly_table(
            self._table,
            _CANDIDATE_COLUMNS,
            accessible_name="Match candidates",
            accessible_description=(
                "Candidate recordings with score, artist, title, album, date, track, "
                "and source. Press digits 1 to 9 to pick a candidate when not typing."
            ),
        )
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.cellDoubleClicked.connect(self._on_row_double_clicked)
        self._table.installEventFilter(self)
        self._candidates_stack.addWidget(self._table)

        self._empty_page = QFrame(self._candidates_stack)
        self._empty_page.setObjectName("match_empty_state")
        self._empty_page.setFrameShape(QFrame.Shape.StyledPanel)
        self._empty_page.setFrameShadow(QFrame.Shadow.Plain)
        empty_layout = QVBoxLayout(self._empty_page)
        empty_layout.setContentsMargins(16, 20, 16, 20)
        empty_layout.setSpacing(6)
        empty_layout.addStretch(1)

        self._empty_title_label = QLabel("No candidates", self._empty_page)
        self._empty_title_label.setObjectName("match_empty_title_label")
        self._empty_title_label.setFont(make_heading_font(self._empty_title_label, scale=1.05))
        self._empty_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_title_label.setTextFormat(Qt.TextFormat.PlainText)
        empty_layout.addWidget(self._empty_title_label)

        self._empty_hint_label = QLabel(
            "Use Search again to query providers with different text, "
            "or choose Enter tags to provide custom metadata.",
            self._empty_page,
        )
        self._empty_hint_label.setObjectName("match_empty_hint_label")
        self._empty_hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_hint_label.setWordWrap(True)
        self._empty_hint_label.setTextFormat(Qt.TextFormat.PlainText)
        empty_layout.addWidget(self._empty_hint_label)
        empty_layout.addStretch(1)
        self._candidates_stack.addWidget(self._empty_page)

        root.addWidget(self._candidates_stack, stretch=1)

        # -- Shortcut hint line ---------------------------------------------------
        self._shortcut_hint = QLabel(
            "Shortcuts: digits 1 to 9 pick a candidate when focus is outside a text box; "
            "Escape hides an open form or uses fallback metadata.",
            self,
        )
        self._shortcut_hint.setObjectName("match_shortcut_hint")
        self._shortcut_hint.setWordWrap(True)
        self._shortcut_hint.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self._shortcut_hint)

        # -- Inline Search again form ---------------------------------------------
        self._search_panel = QGroupBox("Search again", self)
        self._search_panel.setObjectName("match_search_panel")
        configure_accessible(
            self._search_panel,
            "Search again form",
            "Enter artist, title, and album to run another online metadata search.",
        )
        search_outer = QVBoxLayout(self._search_panel)
        search_outer.setSpacing(6)

        search_form = QFormLayout()
        search_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        search_form.setSpacing(6)

        search_artist_label = QLabel("&Artist (optional):", self._search_panel)
        self._search_artist_edit = QLineEdit(self._search_panel)
        self._search_artist_edit.setObjectName("match_search_artist_edit")
        self._search_artist_edit.setPlaceholderText("Optional artist name")
        configure_accessible(
            self._search_artist_edit,
            "Search artist",
            "Optional artist name for manual online lookup.",
        )
        search_artist_label.setBuddy(self._search_artist_edit)
        self._search_artist_edit.returnPressed.connect(self._on_search_return_pressed)
        search_form.addRow(search_artist_label, self._search_artist_edit)

        search_title_label = QLabel("&Title (required):", self._search_panel)
        self._search_title_edit = QLineEdit(self._search_panel)
        self._search_title_edit.setObjectName("match_search_title_edit")
        self._search_title_edit.setPlaceholderText("Required track title")
        configure_accessible(
            self._search_title_edit,
            "Search title",
            "Required track title for manual online lookup.",
        )
        search_title_label.setBuddy(self._search_title_edit)
        self._search_title_edit.textChanged.connect(self._on_search_fields_changed)
        self._search_title_edit.returnPressed.connect(self._on_search_return_pressed)
        search_form.addRow(search_title_label, self._search_title_edit)

        search_album_label = QLabel("Al&bum (optional):", self._search_panel)
        self._search_album_edit = QLineEdit(self._search_panel)
        self._search_album_edit.setObjectName("match_search_album_edit")
        self._search_album_edit.setPlaceholderText("Optional album title")
        configure_accessible(
            self._search_album_edit,
            "Search album",
            "Optional album title for manual online lookup.",
        )
        search_album_label.setBuddy(self._search_album_edit)
        self._search_album_edit.returnPressed.connect(self._on_search_return_pressed)
        search_form.addRow(search_album_label, self._search_album_edit)
        search_outer.addLayout(search_form)

        self._search_error_label = QLabel("", self._search_panel)
        self._search_error_label.setObjectName("match_search_error_label")
        self._search_error_label.setWordWrap(True)
        self._search_error_label.setTextFormat(Qt.TextFormat.PlainText)
        self._search_error_label.setVisible(False)
        search_outer.addWidget(self._search_error_label)

        search_buttons = QHBoxLayout()
        search_buttons.addStretch(1)
        self._search_hide_button = QPushButton("&Hide", self._search_panel)
        self._search_hide_button.setObjectName("match_search_hide_button")
        self._search_hide_button.setAutoDefault(False)
        configure_accessible(
            self._search_hide_button,
            "Hide search form",
            "Collapse the inline search form.",
        )
        self._search_hide_button.clicked.connect(self._on_hide_search_panel)
        search_buttons.addWidget(self._search_hide_button)

        self._search_submit_button = QPushButton("Ru&n search", self._search_panel)
        self._search_submit_button.setObjectName("match_search_submit_button")
        self._search_submit_button.setAutoDefault(False)
        self._search_submit_button.setEnabled(False)
        configure_accessible(
            self._search_submit_button,
            "Run search",
            "Run an online lookup using the entered artist, title, and album.",
        )
        self._search_submit_button.clicked.connect(self._on_search_submit)
        search_buttons.addWidget(self._search_submit_button)
        search_outer.addLayout(search_buttons)

        self._search_panel.setVisible(False)
        root.addWidget(self._search_panel)

        # -- Inline Enter tags form -----------------------------------------------
        self._tags_panel = QGroupBox("Enter custom tags", self)
        self._tags_panel.setObjectName("match_tags_panel")
        configure_accessible(
            self._tags_panel,
            "Enter custom tags form",
            "Enter artist, title, album, and track number to set manual tags.",
        )
        tags_outer = QVBoxLayout(self._tags_panel)
        tags_outer.setSpacing(6)

        tags_form = QFormLayout()
        tags_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        tags_form.setSpacing(6)

        tags_artist_label = QLabel("&Artist (optional):", self._tags_panel)
        self._tags_artist_edit = QLineEdit(self._tags_panel)
        self._tags_artist_edit.setObjectName("match_tags_artist_edit")
        self._tags_artist_edit.setPlaceholderText("Optional artist name")
        configure_accessible(
            self._tags_artist_edit,
            "Custom artist",
            "Optional manual artist tag (also used as album artist).",
        )
        tags_artist_label.setBuddy(self._tags_artist_edit)
        self._tags_artist_edit.returnPressed.connect(self._on_tags_return_pressed)
        tags_form.addRow(tags_artist_label, self._tags_artist_edit)

        tags_title_label = QLabel("&Title (required):", self._tags_panel)
        self._tags_title_edit = QLineEdit(self._tags_panel)
        self._tags_title_edit.setObjectName("match_tags_title_edit")
        self._tags_title_edit.setPlaceholderText("Required track title")
        configure_accessible(
            self._tags_title_edit,
            "Custom title",
            "Required manual track title tag.",
        )
        tags_title_label.setBuddy(self._tags_title_edit)
        self._tags_title_edit.textChanged.connect(self._on_tags_fields_changed)
        self._tags_title_edit.returnPressed.connect(self._on_tags_return_pressed)
        tags_form.addRow(tags_title_label, self._tags_title_edit)

        tags_album_label = QLabel("Al&bum (optional):", self._tags_panel)
        self._tags_album_edit = QLineEdit(self._tags_panel)
        self._tags_album_edit.setObjectName("match_tags_album_edit")
        self._tags_album_edit.setPlaceholderText("Optional album name")
        configure_accessible(
            self._tags_album_edit,
            "Custom album",
            "Optional manual album tag.",
        )
        tags_album_label.setBuddy(self._tags_album_edit)
        self._tags_album_edit.returnPressed.connect(self._on_tags_return_pressed)
        tags_form.addRow(tags_album_label, self._tags_album_edit)

        tags_track_label = QLabel("Track &number (optional):", self._tags_panel)
        self._tags_track_edit = QLineEdit(self._tags_panel)
        self._tags_track_edit.setObjectName("match_tags_track_edit")
        self._tags_track_edit.setPlaceholderText("Optional positive integer")
        configure_accessible(
            self._tags_track_edit,
            "Custom track number",
            "Optional positive integer track number.",
        )
        tags_track_label.setBuddy(self._tags_track_edit)
        self._tags_track_edit.textChanged.connect(self._on_tags_fields_changed)
        self._tags_track_edit.returnPressed.connect(self._on_tags_return_pressed)
        tags_form.addRow(tags_track_label, self._tags_track_edit)
        tags_outer.addLayout(tags_form)

        self._tags_error_label = QLabel("", self._tags_panel)
        self._tags_error_label.setObjectName("match_tags_error_label")
        self._tags_error_label.setWordWrap(True)
        self._tags_error_label.setTextFormat(Qt.TextFormat.PlainText)
        self._tags_error_label.setVisible(False)
        tags_outer.addWidget(self._tags_error_label)

        tags_buttons = QHBoxLayout()
        tags_buttons.addStretch(1)
        self._tags_hide_button = QPushButton("&Hide", self._tags_panel)
        self._tags_hide_button.setObjectName("match_tags_hide_button")
        self._tags_hide_button.setAutoDefault(False)
        configure_accessible(
            self._tags_hide_button,
            "Hide custom tags form",
            "Collapse the inline custom tags form.",
        )
        self._tags_hide_button.clicked.connect(self._on_hide_tags_panel)
        tags_buttons.addWidget(self._tags_hide_button)

        self._tags_submit_button = QPushButton("A&pply tags", self._tags_panel)
        self._tags_submit_button.setObjectName("match_tags_submit_button")
        self._tags_submit_button.setAutoDefault(False)
        self._tags_submit_button.setEnabled(False)
        configure_accessible(
            self._tags_submit_button,
            "Apply tags",
            "Save and apply the manually entered track metadata.",
        )
        self._tags_submit_button.clicked.connect(self._on_tags_submit)
        tags_buttons.addWidget(self._tags_submit_button)
        tags_outer.addLayout(tags_buttons)

        self._tags_panel.setVisible(False)
        root.addWidget(self._tags_panel)

        # -- Primary action row ---------------------------------------------------
        actions = QHBoxLayout()
        actions.setSpacing(8)

        self._skip_button = QPushButton("S&kip file", self)
        self._skip_button.setObjectName("match_skip_button")
        self._skip_button.setAutoDefault(False)
        self._skip_button.setToolTip("Skip extracting this file.")
        configure_accessible(self._skip_button, "Skip file", "Exclude this file from extraction.")
        self._skip_button.clicked.connect(self._on_skip_file)
        actions.addWidget(self._skip_button)

        self._fallback_all_button = QPushButton("Use fallback for the &rest", self)
        self._fallback_all_button.setObjectName("match_fallback_all_button")
        self._fallback_all_button.setAutoDefault(False)
        self._fallback_all_button.setToolTip(
            "Use offline fallback metadata for this and all remaining uncertain tracks."
        )
        configure_accessible(
            self._fallback_all_button,
            "Use fallback for the rest",
            "Stop prompting and apply offline fallback metadata to remaining uncertain tracks.",
        )
        self._fallback_all_button.clicked.connect(self._on_use_fallback_all)
        actions.addWidget(self._fallback_all_button)

        self._fallback_button = QPushButton("Use &fallback", self)
        self._fallback_button.setObjectName("match_fallback_button")
        self._fallback_button.setAutoDefault(False)
        self._fallback_button.setToolTip(
            "Use offline fallback metadata from the file's tags and filename (Escape)."
        )
        configure_accessible(
            self._fallback_button,
            "Use fallback",
            "Use offline fallback metadata for this track.",
        )
        self._fallback_button.clicked.connect(self._on_use_fallback)
        actions.addWidget(self._fallback_button)

        actions.addStretch(1)

        self._tags_button = QPushButton("&Enter tags", self)
        self._tags_button.setObjectName("match_tags_button")
        self._tags_button.setAutoDefault(False)
        self._tags_button.setToolTip("Enter artist, title, album, and track tags manually.")
        configure_accessible(
            self._tags_button,
            "Enter tags",
            "Open the inline form to enter custom track metadata.",
        )
        self._tags_button.clicked.connect(self._on_open_tags_panel)
        actions.addWidget(self._tags_button)

        self._search_button = QPushButton("&Search again", self)
        self._search_button.setObjectName("match_search_button")
        self._search_button.setAutoDefault(False)
        self._search_button.setToolTip(
            "Run a new online search with custom artist, title, or album."
        )
        configure_accessible(
            self._search_button,
            "Search again",
            "Open the inline form to run another online metadata search.",
        )
        self._search_button.clicked.connect(self._on_open_search_panel)
        actions.addWidget(self._search_button)

        self._use_selected_button = QPushButton("&Use selected", self)
        self._use_selected_button.setObjectName("match_use_selected_button")
        self._use_selected_button.setAutoDefault(False)
        self._use_selected_button.setToolTip("Apply the highlighted candidate recording.")
        configure_accessible(
            self._use_selected_button,
            "Use selected",
            "Apply the highlighted candidate match from the table.",
        )
        self._use_selected_button.clicked.connect(self._on_use_selected)
        actions.addWidget(self._use_selected_button)

        root.addLayout(actions)

        self._search_edits: tuple[QLineEdit, ...] = (
            self._search_artist_edit,
            self._search_title_edit,
            self._search_album_edit,
        )
        self._tags_edits: tuple[QLineEdit, ...] = (
            self._tags_artist_edit,
            self._tags_title_edit,
            self._tags_album_edit,
            self._tags_track_edit,
        )
        for line_edit in (*self._search_edits, *self._tags_edits):
            line_edit.installEventFilter(self)

        self._action_buttons: tuple[QPushButton, ...] = (
            self._use_selected_button,
            self._search_button,
            self._tags_button,
            self._fallback_button,
            self._fallback_all_button,
            self._skip_button,
            self._search_hide_button,
            self._search_submit_button,
            self._tags_hide_button,
            self._tags_submit_button,
        )
        for action_btn in self._action_buttons:
            action_btn.installEventFilter(self)
        self._busy_frame.installEventFilter(self)
        self._busy_label.installEventFilter(self)
        self.cancel_run_button.installEventFilter(self)

        # Install logical keyboard tab order.
        self.setTabOrder(self.cancel_run_button, self._table)
        self.setTabOrder(self._table, self._search_artist_edit)
        self.setTabOrder(self._search_artist_edit, self._search_title_edit)
        self.setTabOrder(self._search_title_edit, self._search_album_edit)
        self.setTabOrder(self._search_album_edit, self._search_submit_button)
        self.setTabOrder(self._search_submit_button, self._search_hide_button)
        self.setTabOrder(self._search_hide_button, self._tags_artist_edit)
        self.setTabOrder(self._tags_artist_edit, self._tags_title_edit)
        self.setTabOrder(self._tags_title_edit, self._tags_album_edit)
        self.setTabOrder(self._tags_album_edit, self._tags_track_edit)
        self.setTabOrder(self._tags_track_edit, self._tags_submit_button)
        self.setTabOrder(self._tags_submit_button, self._tags_hide_button)
        self.setTabOrder(self._tags_hide_button, self._use_selected_button)
        self.setTabOrder(self._use_selected_button, self._search_button)
        self.setTabOrder(self._search_button, self._tags_button)
        self.setTabOrder(self._tags_button, self._fallback_button)
        self.setTabOrder(self._fallback_button, self._fallback_all_button)
        self.setTabOrder(self._fallback_all_button, self._skip_button)

        self._apply_request(request, preserve_forms=False)

    @property
    def is_busy(self) -> bool:
        """Return True while the dialog is awaiting a manual search refill."""
        return self._busy

    @property
    def request(self) -> PromptRequest:
        """Return the currently displayed ``PromptRequest``."""
        return self._request

    def _clear_notice(self) -> None:
        self._notice_label.setText("")
        self._notice_banner.setVisible(False)
        configure_accessible(
            self._notice_banner,
            "Match notice",
            "Inline notice displayed inside the match dialog.",
        )

    def _apply_request(self, request: PromptRequest, *, preserve_forms: bool) -> None:
        payload = request.payload
        assert isinstance(payload, MatchPrompt)
        self._request = request
        self._payload = payload
        self._candidates = tuple(payload.match.top_candidates[:MAX_DISPLAY_CANDIDATES])
        has_candidates = bool(self._candidates)

        heading, subtitle = _reason_header(payload.match.decision, has_candidates=has_candidates)
        self._reason_label.setText(heading)
        self._subtitle_label.setText(subtitle)

        file_name = request.media_input.path.name
        self._file_label.setText(f"File: {file_name}")
        self._file_label.set_full_tooltip(str(request.media_input.path))
        configure_accessible(
            self._file_label,
            "Source file",
            f"Source file name: {file_name}.",
        )

        hints_text = _format_hints_summary(payload.hints)
        self._hints_label.setText(hints_text)
        configure_accessible(self._hints_label, "Parsed hints", hints_text)

        configure_accessible(
            self,
            f"Resolve match: {heading}",
            f"{heading} for {file_name}. {subtitle}",
        )

        if not preserve_forms:
            self._search_artist_edit.clear()
            self._search_title_edit.clear()
            self._search_album_edit.clear()
            self._search_error_label.setText("")
            self._search_error_label.setVisible(False)
            self._tags_artist_edit.clear()
            self._tags_title_edit.clear()
            self._tags_album_edit.clear()
            self._tags_track_edit.clear()
            self._tags_error_label.setText("")
            self._tags_error_label.setVisible(False)

        self._populate_candidates_table()

        if not preserve_forms:
            self._search_panel.setVisible(False)
            self._tags_panel.setVisible(False)

        if has_candidates:
            self._candidates_stack.setCurrentWidget(self._table)
            self._shortcut_hint.setVisible(True)
            _set_button_emphasized(self._search_button, False)
            _set_button_emphasized(self._tags_button, False)
        else:
            self._candidates_stack.setCurrentWidget(self._empty_page)
            self._shortcut_hint.setVisible(False)
            _set_button_emphasized(self._search_button, True)
            _set_button_emphasized(self._tags_button, True)

        self._sync_default_button()
        self._sync_enabled_states()

        if not self._has_open_inline_form():
            self._focus_default_target()
        elif not self._search_panel.isHidden():
            self._search_title_edit.setFocus()
        elif not self._tags_panel.isHidden():
            self._tags_title_edit.setFocus()

    def _populate_candidates_table(self) -> None:
        self._table.setRowCount(len(self._candidates))
        align_right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        for row, cand in enumerate(self._candidates):
            artist_tip = _candidate_artist_tooltip(cand)
            album_text = cand.album or EM_DASH
            album_tip = cand.album or "No album"
            date_text = format_candidate_date(cand)
            track_text = format_candidate_track(cand)
            source_text = cand.source or EM_DASH

            self._table.setItem(row, 0, make_table_item(str(row + 1), alignment=align_right))
            self._table.setItem(
                row,
                1,
                make_table_item(f"{cand.score:.2f}", alignment=align_right),
            )
            self._table.setItem(row, 2, make_table_item(cand.artist, tooltip=artist_tip))
            self._table.setItem(row, 3, make_table_item(cand.title))
            self._table.setItem(row, 4, make_table_item(album_text, tooltip=album_tip))
            self._table.setItem(row, 5, make_table_item(date_text))
            self._table.setItem(row, 6, make_table_item(track_text, alignment=align_right))
            self._table.setItem(row, 7, make_table_item(source_text))

        header = self._table.horizontalHeader()
        for col in (0, 1, 5, 6, 7):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        for col in (2, 3, 4):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)

        if self._candidates:
            self._table.selectRow(0)
            self._table.setCurrentCell(0, 0)

    def _sync_default_button(self) -> None:
        default_auto = (
            self._payload.always_prompt
            and self._payload.match.decision == MatchDecision.AUTO_ACCEPT
            and bool(self._candidates)
        )
        self._use_selected_button.setDefault(default_auto)
        self._fallback_button.setDefault(not default_auto)
        self.cancel_run_button.setAutoDefault(False)
        self.cancel_run_button.setDefault(False)

    def _focus_default_target(self) -> None:
        """Set keyboard focus to match the dialog's active safe default action."""
        if self._use_selected_button.isDefault() and self._candidates:
            self._table.setFocus()
        else:
            self._fallback_button.setFocus()

    def _has_open_inline_form(self) -> bool:
        """Return True when the inline search or custom-tags form is currently open."""
        return (not self._search_panel.isHidden()) or (not self._tags_panel.isHidden())

    def _dismiss_open_inline_form(self) -> bool:
        """Hide any open inline form while preserving draft input text."""
        dismissed = False
        if not self._search_panel.isHidden():
            self._on_hide_search_panel()
            dismissed = True
        if not self._tags_panel.isHidden():
            self._on_hide_tags_panel()
            dismissed = True
        return dismissed

    def _sync_enabled_states(self) -> None:
        active = not self._busy
        has_candidates = bool(self._candidates)
        has_selection = has_candidates and 0 <= self._table.currentRow() < len(self._candidates)

        self._table.setEnabled(active)
        self._use_selected_button.setEnabled(active and has_selection)
        self._search_button.setEnabled(active)
        self._tags_button.setEnabled(active)
        self._fallback_button.setEnabled(active)
        self._fallback_all_button.setEnabled(active)
        self._skip_button.setEnabled(active)

        self._search_artist_edit.setEnabled(active)
        self._search_title_edit.setEnabled(active)
        self._search_album_edit.setEnabled(active)
        self._search_hide_button.setEnabled(active)
        can_search = active and bool(self._search_title_edit.text().strip())
        self._search_submit_button.setEnabled(can_search)

        self._tags_artist_edit.setEnabled(active)
        self._tags_title_edit.setEnabled(active)
        self._tags_album_edit.setEnabled(active)
        self._tags_track_edit.setEnabled(active)
        self._tags_hide_button.setEnabled(active)
        _, track_err = _parse_optional_track_number(self._tags_track_edit.text())
        can_apply_tags = active and bool(self._tags_title_edit.text().strip()) and track_err is None
        self._tags_submit_button.setEnabled(can_apply_tags)

    def set_busy(self, busy: bool) -> None:
        """Toggle the busy 'Searching…' state while a manual search executes."""
        was_busy = self._busy
        self._busy = busy
        if busy:
            if not was_busy:
                self._clear_notice()
                self._notice_while_busy = False
                # Keep cancel_run_button hidden and disabled while clearing focus and
                # disabling all other controls first so Qt never shifts focus along the
                # focus chain onto Cancel run.
                self.cancel_run_button.setEnabled(False)
                self.cancel_run_button.setVisible(False)
                current_focus = self.focusWidget()
                if current_focus is not None:
                    current_focus.clearFocus()
                self._sync_enabled_states()
                self._busy_frame.setVisible(True)
                self.cancel_run_button.setVisible(True)
                self.cancel_run_button.setEnabled(True)
            else:
                self._sync_enabled_states()
                self._busy_frame.setVisible(True)
                self.cancel_run_button.setVisible(True)
            # Park focus on the status label (never on cancel_run_button) after
            # updating widget enabled states so a second Enter press or key auto-repeat
            # never triggers a cancel.
            self._busy_label.setFocus(Qt.FocusReason.OtherFocusReason)
            return

        self.cancel_run_button.setEnabled(False)
        self.cancel_run_button.setVisible(False)
        self._busy_frame.setVisible(False)
        self._sync_enabled_states()
        if was_busy:
            if not self._search_panel.isHidden():
                self._search_title_edit.setFocus()
            elif not self._tags_panel.isHidden():
                self._tags_title_edit.setFocus()
            else:
                self._focus_default_target()

    def refill(self, request: PromptRequest) -> None:
        """Replace displayed match candidates with ``request`` and clear the busy state."""
        if request.kind != PromptKind.MATCH:
            raise ValueError(f"MatchDialog.refill requires PromptKind.MATCH, got {request.kind!r}")
        if not isinstance(request.payload, MatchPrompt):
            raise TypeError(
                f"MatchDialog.refill requires a MatchPrompt payload, got {request.payload!r}"
            )

        keep_notice = self._notice_while_busy
        self._notice_while_busy = False
        same_file = request.media_input.path == self._request.media_input.path

        self._answered = False
        self._withdrawn = False
        self._busy = False
        self._busy_frame.setVisible(False)
        self.cancel_run_button.setVisible(False)
        self.cancel_run_button.setEnabled(False)
        self._apply_request(request, preserve_forms=same_file)

        if keep_notice:
            # Keep the inline notice (for example, "Manual search found no candidates")
            # visible so the user can see it and adjust their search query directly.
            self._notice_banner.setVisible(True)
        else:
            self._clear_notice()
            if same_file and self._candidates:
                self._search_panel.setVisible(False)
                self._tags_panel.setVisible(False)
                self._search_error_label.setText("")
                self._search_error_label.setVisible(False)
                self._tags_error_label.setText("")
                self._tags_error_label.setVisible(False)
                self._focus_default_target()

    def show_notice(self, message: str) -> None:
        """Display an inline notice inside the dialog (for example, no search results)."""
        cleaned = message.strip()
        if not cleaned:
            self._clear_notice()
            return
        self._notice_label.setText(cleaned)
        self._notice_label.setToolTip(cleaned)
        configure_accessible(self._notice_banner, "Match notice", cleaned)
        configure_accessible(self._notice_label, "Notice message", cleaned)
        self._notice_banner.setVisible(True)
        if self._busy:
            self._notice_while_busy = True

    def withdraw(self) -> None:
        """Close the dialog programmatically without emitting ``answered``."""
        if self._withdrawn:
            return
        self._withdrawn = True
        self._busy = False
        self._busy_frame.setVisible(False)
        self.cancel_run_button.setVisible(False)
        self.cancel_run_button.setEnabled(False)
        if self.isVisible():
            super().reject()

    def _emit_answer(
        self, response: CandidatePromptResponse, *, close: bool, accepted: bool
    ) -> None:
        if self._withdrawn or self._answered or self._busy:
            return
        self._answered = True
        if close:
            if accepted:
                super().accept()
            else:
                super().reject()
        self.answered.emit(self._request.request_id, response)

    def _submit_candidate(self, candidate: Candidate) -> None:
        self._emit_answer(
            CandidatePromptResponse(
                action=CandidateAction.SELECT_CANDIDATE,
                candidate=candidate,
            ),
            close=True,
            accepted=True,
        )

    @Slot()
    def _on_selection_changed(self) -> None:
        self._sync_enabled_states()

    @Slot(int, int)
    def _on_row_double_clicked(self, row: int, column: int) -> None:
        _ = column
        if self._busy or self._answered or self._withdrawn:
            return
        if 0 <= row < len(self._candidates):
            self._submit_candidate(self._candidates[row])

    @Slot()
    def _on_use_selected(self) -> None:
        row = self._table.currentRow()
        if 0 <= row < len(self._candidates):
            self._submit_candidate(self._candidates[row])

    @Slot()
    def _on_open_search_panel(self) -> None:
        if self._busy:
            return
        self._tags_error_label.setText("")
        self._tags_error_label.setVisible(False)
        self._tags_panel.setVisible(False)
        self._search_panel.setVisible(True)
        self._search_title_edit.setFocus()

    @Slot()
    def _on_hide_search_panel(self) -> None:
        self._search_error_label.setText("")
        self._search_error_label.setVisible(False)
        self._search_panel.setVisible(False)
        self._focus_default_target()

    @Slot(str)
    def _on_search_fields_changed(self, text: str) -> None:
        _ = text
        if self._search_title_edit.text().strip():
            self._search_error_label.setText("")
            self._search_error_label.setVisible(False)
        self._sync_enabled_states()

    @Slot()
    def _on_search_return_pressed(self) -> None:
        self._on_search_submit()

    @Slot()
    def _on_search_submit(self) -> None:
        if self._busy or self._answered or self._withdrawn:
            return
        title = self._search_title_edit.text().strip()
        if not title:
            self._search_error_label.setText("Title is required for a manual search.")
            self._search_error_label.setVisible(True)
            self._search_title_edit.setFocus()
            return
        artist = self._search_artist_edit.text().strip() or None
        album = self._search_album_edit.text().strip() or None
        self._search_error_label.setText("")
        self._search_error_label.setVisible(False)
        self._emit_answer(
            CandidatePromptResponse(
                action=CandidateAction.MANUAL_SEARCH,
                search_artist=artist,
                search_title=title,
                search_album=album,
            ),
            close=False,
            accepted=False,
        )

    @Slot()
    def _on_open_tags_panel(self) -> None:
        if self._busy:
            return
        self._search_error_label.setText("")
        self._search_error_label.setVisible(False)
        self._search_panel.setVisible(False)
        self._tags_panel.setVisible(True)
        _, track_err = _parse_optional_track_number(self._tags_track_edit.text())
        if track_err is not None:
            self._tags_error_label.setText(track_err)
            self._tags_error_label.setVisible(True)
        else:
            self._tags_error_label.setText("")
            self._tags_error_label.setVisible(False)
        self._tags_title_edit.setFocus()

    @Slot()
    def _on_hide_tags_panel(self) -> None:
        self._tags_error_label.setText("")
        self._tags_error_label.setVisible(False)
        self._tags_panel.setVisible(False)
        self._focus_default_target()

    @Slot(str)
    def _on_tags_fields_changed(self, text: str) -> None:
        _ = text
        _, track_err = _parse_optional_track_number(self._tags_track_edit.text())
        if track_err is not None:
            self._tags_error_label.setText(track_err)
            self._tags_error_label.setVisible(True)
        elif self._tags_title_edit.text().strip():
            self._tags_error_label.setText("")
            self._tags_error_label.setVisible(False)
        self._sync_enabled_states()

    @Slot()
    def _on_tags_return_pressed(self) -> None:
        self._on_tags_submit()

    @Slot()
    def _on_tags_submit(self) -> None:
        if self._busy or self._answered or self._withdrawn:
            return
        title = self._tags_title_edit.text().strip()
        if not title:
            self._tags_error_label.setText("Title is required for custom tags.")
            self._tags_error_label.setVisible(True)
            self._tags_title_edit.setFocus()
            return
        track_num, track_err = _parse_optional_track_number(self._tags_track_edit.text())
        if track_err is not None:
            self._tags_error_label.setText(track_err)
            self._tags_error_label.setVisible(True)
            self._tags_track_edit.setFocus()
            return
        artist = self._tags_artist_edit.text().strip() or None
        album = self._tags_album_edit.text().strip() or None
        try:
            metadata = build_manual_metadata(artist, title, album, track_num)
        except ValueError as exc:
            self._tags_error_label.setText(str(exc))
            self._tags_error_label.setVisible(True)
            return
        self._tags_error_label.setText("")
        self._tags_error_label.setVisible(False)
        self._emit_answer(
            CandidatePromptResponse(
                action=CandidateAction.CUSTOM_TAGS,
                custom_metadata=metadata,
            ),
            close=True,
            accepted=True,
        )

    @Slot()
    def _on_use_fallback(self) -> None:
        self._emit_answer(
            CandidatePromptResponse(action=CandidateAction.USE_FALLBACK),
            close=True,
            accepted=True,
        )

    @Slot()
    def _on_use_fallback_all(self) -> None:
        self._emit_answer(
            CandidatePromptResponse(action=CandidateAction.USE_FALLBACK_ALL),
            close=True,
            accepted=True,
        )

    @Slot()
    def _on_skip_file(self) -> None:
        self._emit_answer(
            CandidatePromptResponse(action=CandidateAction.SKIP_TRACK),
            close=True,
            accepted=True,
        )

    def _has_text_input_focus(self) -> bool:
        return is_text_input_widget(QApplication.focusWidget()) or is_text_input_widget(
            self.focusWidget()
        )

    def _try_handle_digit_key(self, event: QKeyEvent) -> bool:
        if self._busy or self._answered or self._withdrawn:
            return False
        if self._has_text_input_focus():
            return False
        if event.modifiers() & _DISALLOWED_MODIFIERS:
            return False
        try:
            key_enum = Qt.Key(event.key())
        except ValueError:
            return False
        digit = _DIGIT_KEYS.get(key_enum)
        if digit is None:
            return False
        if 1 <= digit <= len(self._candidates):
            row = digit - 1
            self._table.selectRow(row)
            self._table.setCurrentCell(row, 0)
            self._submit_candidate(self._candidates[row])
            event.accept()
            return True
        return False

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Route inline Return and Escape presses, table Enter, and digit shortcuts."""
        if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            if event.key() == Qt.Key.Key_Escape and not (event.modifiers() & _KEY_MODIFIER_MASK):
                if self._busy and not self._withdrawn:
                    event.accept()
                    return True
                if (
                    not self._busy
                    and not self._answered
                    and not self._withdrawn
                    and self._dismiss_open_inline_form()
                ):
                    event.accept()
                    return True
                return super().eventFilter(watched, event)

            if watched is self.cancel_run_button and event.key() in (
                Qt.Key.Key_Return,
                Qt.Key.Key_Enter,
            ):
                # Never activate Cancel run with Return/Enter; mouse click or Space
                # after tabbing must be used explicitly.
                event.accept()
                return True

            if self._busy and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                event.accept()
                return True

            if watched in self._search_edits:
                if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
                    event.modifiers() & _DISALLOWED_MODIFIERS
                ):
                    self._on_search_submit()
                    event.accept()
                    return True
                return super().eventFilter(watched, event)

            if watched in self._tags_edits:
                if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
                    event.modifiers() & _DISALLOWED_MODIFIERS
                ):
                    self._on_tags_submit()
                    event.accept()
                    return True
                return super().eventFilter(watched, event)

            if isinstance(watched, QWidget) and is_text_input_widget(watched):
                return super().eventFilter(watched, event)

            if self._try_handle_digit_key(event):
                return True

            if (
                watched is self._table
                and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and not (event.modifiers() & _KEY_MODIFIER_MASK)
                and not self._busy
                and not self._answered
                and not self._withdrawn
            ):
                if (
                    self._use_selected_button.isDefault()
                    and self._use_selected_button.isEnabled()
                    and self._candidates
                    and 0 <= self._table.currentRow() < len(self._candidates)
                ):
                    self._on_use_selected()
                elif self._fallback_button.isDefault() and self._fallback_button.isEnabled():
                    self._on_use_fallback()
                event.accept()
                return True

            if (
                watched in self._action_buttons
                and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and not (event.modifiers() & _KEY_MODIFIER_MASK)
            ):
                if (
                    not self._busy
                    and not self._answered
                    and not self._withdrawn
                    and isinstance(watched, QPushButton)
                    and watched.isEnabled()
                ):
                    watched.click()
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle Escape (hide open forms first), busy guards, and digit shortcuts."""
        if event.key() == Qt.Key.Key_Escape:
            if self._busy and not self._withdrawn:
                event.accept()
                return
            if (
                not self._answered
                and not self._withdrawn
                and not (event.modifiers() & _KEY_MODIFIER_MASK)
                and self._dismiss_open_inline_form()
            ):
                event.accept()
                return
            super().keyPressEvent(event)
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and (
            self._busy or self._answered or self._has_text_input_focus()
        ):
            # Inline line edits handle Return directly; while busy, swallow Enter so
            # a second press or key auto-repeat never falls through or cancels the run.
            event.accept()
            return
        if self._try_handle_digit_key(event):
            return
        super().keyPressEvent(event)

    def reject(self) -> None:
        """Handle Escape or window dismissal with ``USE_FALLBACK`` unless busy or withdrawn."""
        if self._withdrawn:
            super().reject()
            return
        if self._busy:
            return
        if self._answered:
            super().reject()
            return
        self._emit_answer(
            CandidatePromptResponse(action=CandidateAction.USE_FALLBACK),
            close=True,
            accepted=False,
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        """Block direct close while searching; otherwise fall back safely once."""
        if self._withdrawn:
            event.accept()
            return
        if self._busy:
            event.ignore()
            return
        if self._answered:
            event.accept()
            return
        if self.isVisible():
            self._emit_answer(
                CandidatePromptResponse(action=CandidateAction.USE_FALLBACK),
                close=True,
                accepted=False,
            )
        event.accept()
