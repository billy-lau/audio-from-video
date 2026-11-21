"""Window-modal dialog for supplying an English alias for non-Latin metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QObject, QSize, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QFont, QFontMetrics, QKeyEvent
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from audio_from_video.gui.bridge import AliasPrompt, PromptKind
from audio_from_video.gui.dialogs._common import (
    ElidingPushButton,
    MiddleElideLabel,
    configure_accessible,
    dialog_button_icon,
    escape_mnemonic,
    make_context_card,
    make_dialog_header,
    make_divider_line,
    make_heading_font,
    set_button_role,
)
from audio_from_video.gui.theme import is_theme_updating, refresh_themed_widgets, repolish
from audio_from_video.planner import AliasAction, AliasPromptResponse

if TYPE_CHECKING:
    from audio_from_video.gui.bridge import PromptRequest

_BUTTON_ICON_SIZE = QSize(14, 14)


def _format_field_kind(field_kind: str) -> str:
    """Turn an internal field name such as ``album_artist`` into readable text."""
    return field_kind.replace("_", " ").strip() or "value"


def _make_caption_font(widget: QWidget) -> QFont:
    """Return a slightly smaller font relative to ``widget`` for subtle helper lines."""
    return make_heading_font(widget, scale=0.92, bold=False)


class AliasDialog(QDialog):
    """Prompt dialog returning an ``AliasPromptResponse`` (opened with ``open()``)."""

    answered = Signal(int, object)  # request_id, AliasPromptResponse

    def __init__(self, request: PromptRequest, parent: QWidget) -> None:
        super().__init__(parent)
        if request.kind != PromptKind.ALIAS:
            raise ValueError(f"AliasDialog requires PromptKind.ALIAS, got {request.kind!r}")
        if not isinstance(request.payload, AliasPrompt):
            raise TypeError(f"AliasDialog requires an AliasPrompt payload, got {request.payload!r}")

        self._request = request
        self._payload: AliasPrompt = request.payload
        self._answered = False
        self._withdrawn = False

        field_label = _format_field_kind(self._payload.field_kind)
        scope_label = self._payload.scope_label.strip() or "this artist or album"

        self.setObjectName("alias_dialog")
        self.setWindowTitle("English Name Alias")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumWidth(480)
        self.setMinimumHeight(370)
        self.resize(600, 395)
        configure_accessible(
            self,
            "English name alias",
            (
                f"Enter an English alias for {field_label} {self._payload.original_value}, "
                "or keep the native name."
            ),
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(12)

        # -- Header (icon badge, heading, and subtitle) --------------------------
        self._heading_label = QLabel(f"Missing English alias for {field_label}", self)
        self._heading_label.setObjectName("alias_heading_label")
        self._heading_label.setFont(make_heading_font(self._heading_label, scale=1.28))
        self._heading_label.setProperty("afvText", "title")
        self._heading_label.setTextFormat(Qt.TextFormat.PlainText)
        self._heading_label.setWordWrap(True)
        configure_accessible(
            self._heading_label,
            "Dialog heading",
            f"Missing English alias for {field_label}.",
        )

        self._subtitle_label = QLabel(
            "Enter an English name to pair with the native text, "
            "or choose how widely to keep the native name.",
            self,
        )
        self._subtitle_label.setObjectName("alias_subtitle_label")
        self._subtitle_label.setFont(_make_caption_font(self._subtitle_label))
        self._subtitle_label.setWordWrap(True)
        self._subtitle_label.setTextFormat(Qt.TextFormat.PlainText)
        configure_accessible(
            self._subtitle_label,
            "Dialog instructions",
            "Guidance for entering an English alias or keeping the native name.",
        )

        header_row, self._header_icon = make_dialog_header(
            self,
            icon_name="tag",
            heading_label=self._heading_label,
            subtitle_label=self._subtitle_label,
            icon_size=26,
        )
        configure_accessible(
            self._header_icon,
            "Alias dialog icon",
            "Icon indicating a metadata tag alias prompt.",
        )
        root.addLayout(header_row)

        # -- Recessed context card (field kind, native value, and source file) ---
        context_card = make_context_card(self, object_name="alias_context_card")
        context_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        context_layout = QVBoxLayout(context_card)
        context_layout.setContentsMargins(14, 12, 14, 12)
        context_layout.setSpacing(8)

        meta_grid = QGridLayout()
        meta_grid.setContentsMargins(0, 0, 0, 0)
        meta_grid.setHorizontalSpacing(14)
        meta_grid.setVerticalSpacing(6)
        meta_grid.setColumnStretch(1, 1)

        field_key_label = QLabel("Field:", context_card)
        field_key_label.setFont(_make_caption_font(field_key_label))
        field_key_label.setProperty("afvText", "muted")
        field_key_label.setTextFormat(Qt.TextFormat.PlainText)
        meta_grid.addWidget(field_key_label, 0, 0, Qt.AlignmentFlag.AlignLeft)

        self._kind_value_label = QLabel(field_label, context_card)
        self._kind_value_label.setObjectName("alias_field_kind_label")
        self._kind_value_label.setTextFormat(Qt.TextFormat.PlainText)
        self._kind_value_label.setWordWrap(False)
        configure_accessible(
            self._kind_value_label,
            "Metadata field",
            f"Metadata field kind: {field_label}.",
        )
        meta_grid.addWidget(self._kind_value_label, 0, 1, Qt.AlignmentFlag.AlignLeft)

        native_key_label = QLabel("Native value:", context_card)
        native_key_label.setFont(_make_caption_font(native_key_label))
        native_key_label.setProperty("afvText", "muted")
        native_key_label.setTextFormat(Qt.TextFormat.PlainText)
        meta_grid.addWidget(
            native_key_label,
            1,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )

        self._native_value_label = QLabel(self._payload.original_value, context_card)
        self._native_value_label.setObjectName("alias_original_value_label")
        self._native_value_label.setFont(
            make_heading_font(self._native_value_label, scale=1.08, bold=True)
        )
        self._native_value_label.setTextFormat(Qt.TextFormat.PlainText)
        self._native_value_label.setWordWrap(True)
        self._native_value_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self._native_value_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._native_value_label.setToolTip(
            f"{self._payload.original_value} (drag or double-click to select and copy)"
        )
        configure_accessible(
            self._native_value_label,
            "Native value",
            f"Original non-Latin {field_label}: {self._payload.original_value}.",
        )
        meta_grid.addWidget(self._native_value_label, 1, 1)
        context_layout.addLayout(meta_grid)

        self._file_label = MiddleElideLabel(f"File: {request.media_input.path.name}", context_card)
        self._file_label.setObjectName("alias_file_label")
        self._file_label.setFont(_make_caption_font(self._file_label))
        self._file_label.setProperty("afvText", "muted")
        self._file_label.set_full_tooltip(str(request.media_input.path))
        configure_accessible(
            self._file_label,
            "Source file",
            f"Source file name: {request.media_input.path.name}.",
        )
        context_layout.addWidget(self._file_label)

        repolish(context_card, include_children=True)
        root.addWidget(context_card)

        # -- English alias entry block -------------------------------------------
        entry_col = QVBoxLayout()
        entry_col.setContentsMargins(0, 2, 0, 0)
        entry_col.setSpacing(8)

        english_buddy = QLabel("&English name:", self)
        english_buddy.setProperty("afvText", "section")
        buddy_font = QFont(english_buddy.font())
        buddy_font.setWeight(QFont.Weight.DemiBold)
        english_buddy.setFont(buddy_font)
        entry_col.addWidget(english_buddy)

        self._alias_edit = QLineEdit(self)
        self._alias_edit.setObjectName("alias_english_edit")
        self._alias_edit.setMinimumHeight(32)
        self._alias_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._alias_edit.setClearButtonEnabled(True)
        self._alias_edit.setPlaceholderText("Enter English name or alias")
        self._alias_edit.setToolTip(
            f"Enter an English name to pair with {self._payload.original_value}."
        )
        configure_accessible(
            self._alias_edit,
            "English name",
            f"English name or alias for {field_label} {self._payload.original_value}.",
        )
        english_buddy.setBuddy(self._alias_edit)
        self._alias_edit.textChanged.connect(self._on_text_changed)
        self._alias_edit.returnPressed.connect(self._on_return_pressed)
        self._alias_edit.installEventFilter(self)
        entry_col.addWidget(self._alias_edit)

        self._note_label = QLabel(
            "Combined tags pair both names together (for example, English {native}). "
            "Leave blank to keep the native value for this item, matching scope, or batch.",
            self,
        )
        self._note_label.setObjectName("alias_note_label")
        self._note_label.setFont(_make_caption_font(self._note_label))
        self._note_label.setProperty("afvText", "caption")
        self._note_label.setWordWrap(True)
        self._note_label.setTextFormat(Qt.TextFormat.PlainText)
        self._note_label.setToolTip(
            f"Keep options include this item only, {scope_label}, or the rest of the run."
        )
        configure_accessible(
            self._note_label,
            "Alias options hint",
            "Explains bilingual tag pairing and retention options for native names.",
        )
        entry_col.addWidget(self._note_label)
        root.addLayout(entry_col)

        root.addStretch(1)
        root.addWidget(make_divider_line(self))

        # -- Grouped footer: quiet scope retention row and primary decision row --
        footer_col = QVBoxLayout()
        footer_col.setContentsMargins(0, 0, 0, 0)
        footer_col.setSpacing(8)

        scope_row = QHBoxLayout()
        scope_row.setContentsMargins(0, 0, 0, 0)
        scope_row.setSpacing(8)

        escaped_scope = escape_mnemonic(scope_label)
        plain_scope_text = f"Keep for {scope_label}"
        self._keep_scope_button = ElidingPushButton(
            f"Keep &for {escaped_scope}",
            self,
            max_hint_width=240,
        )
        # Allow enough width past button horizontal padding so short scope labels
        # never prematurely middle-elide inside ElidingPushButton.paintEvent.
        scope_text_px = QFontMetrics(self._keep_scope_button.font()).horizontalAdvance(
            plain_scope_text
        )
        desired_scope_w = min(scope_text_px + 38, 232)
        self._keep_scope_button.set_max_hint_width(max(desired_scope_w, 120))
        self._keep_scope_button.setMinimumWidth(min(desired_scope_w, 224))
        self._keep_scope_button.setObjectName("alias_keep_scope_button")
        self._keep_scope_button.setAutoDefault(False)
        self._keep_scope_button.setToolTip(
            f"Keep the native name for {scope_label} without asking again in this run."
        )
        configure_accessible(
            self._keep_scope_button,
            f"Keep for {scope_label}",
            f"Keep native names for {scope_label} in this run.",
        )
        set_button_role(self._keep_scope_button, "ghost")
        self._keep_scope_button.clicked.connect(self._on_keep_scope)
        scope_row.addWidget(self._keep_scope_button)

        self._keep_all_button = QPushButton("Keep for the &rest of the run", self)
        self._keep_all_button.setObjectName("alias_keep_all_button")
        self._keep_all_button.setAutoDefault(False)
        self._keep_all_button.setToolTip(
            "Keep native non-Latin names for all remaining missing aliases in this batch."
        )
        configure_accessible(
            self._keep_all_button,
            "Keep for the rest of the run",
            "Keep native names without prompting again for the rest of the run.",
        )
        set_button_role(self._keep_all_button, "ghost")
        self._keep_all_button.clicked.connect(self._on_keep_all)
        scope_row.addWidget(self._keep_all_button)
        scope_row.addStretch(1)
        footer_col.addLayout(scope_row)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(8)
        action_row.addStretch(1)

        self._keep_item_button = QPushButton("Keep for this &item", self)
        self._keep_item_button.setObjectName("alias_keep_item_button")
        self._keep_item_button.setAutoDefault(False)
        self._keep_item_button.setDefault(True)
        self._keep_item_button.setToolTip(
            "Keep the native name for this track only (Escape, or Enter when blank)."
        )
        configure_accessible(
            self._keep_item_button,
            "Keep for this item",
            "Keep the native name for this item only.",
        )
        self._keep_item_button.clicked.connect(self._on_keep_item)
        action_row.addWidget(self._keep_item_button)

        self._use_english_button = QPushButton("&Use English name", self)
        self._use_english_button.setObjectName("alias_use_english_button")
        self._use_english_button.setEnabled(False)
        self._use_english_button.setAutoDefault(False)
        self._use_english_button.setIconSize(_BUTTON_ICON_SIZE)
        self._use_english_button.setToolTip(
            "Apply the entered English name together with the native value."
        )
        configure_accessible(
            self._use_english_button,
            "Use English name",
            "Use the typed English name as the alias for this native value.",
        )
        self._use_english_button.clicked.connect(self._on_use_english)
        action_row.addWidget(self._use_english_button)
        footer_col.addLayout(action_row)

        root.addLayout(footer_col)

        self._sync_action_roles(has_text=False)

        root.activate()
        needed_min_h = max(350, root.totalMinimumHeightForWidth(480) + 4)
        self.setMinimumHeight(needed_min_h)
        target_default_h = max(365, root.totalMinimumHeightForWidth(600) + 12)
        self.resize(self.width(), max(needed_min_h, target_default_h))

        self.setTabOrder(self._alias_edit, self._use_english_button)
        self.setTabOrder(self._use_english_button, self._keep_item_button)
        self.setTabOrder(self._keep_item_button, self._keep_scope_button)
        self.setTabOrder(self._keep_scope_button, self._keep_all_button)
        self._alias_edit.setFocus()

    def _sync_action_roles(self, *, has_text: bool) -> None:
        """Align visual primary emphasis with whichever button is currently ``default``."""
        set_button_role(self._use_english_button, "primary" if has_text else "secondary")
        set_button_role(self._keep_item_button, "secondary" if has_text else "primary")
        self._use_english_button.setIcon(dialog_button_icon("check_mark", on_primary=has_text))

    def _refresh_themed_visuals(self) -> None:
        """Refresh the primary button icon contrast on theme updates."""
        if hasattr(self, "_alias_edit"):
            has_text = bool(self._alias_edit.text().strip())
            self._use_english_button.setIcon(dialog_button_icon("check_mark", on_primary=has_text))

    def changeEvent(self, event: QEvent) -> None:
        """Refresh button icon contrast on live theme or palette changes."""
        super().changeEvent(event)
        if (
            not is_theme_updating()
            and event.type()
            in (
                QEvent.Type.PaletteChange,
                QEvent.Type.ApplicationPaletteChange,
                QEvent.Type.StyleChange,
            )
            and hasattr(self, "_alias_edit")
        ):
            refresh_themed_widgets(self)

    def _emit_answer(self, response: AliasPromptResponse, *, accepted: bool) -> None:
        if self._answered or self._withdrawn:
            return
        self._answered = True
        if accepted:
            super().accept()
        else:
            super().reject()
        self.answered.emit(self._request.request_id, response)

    @Slot(str)
    def _on_text_changed(self, text: str) -> None:
        has_text = bool(text.strip())
        self._use_english_button.setEnabled(has_text)
        self._use_english_button.setDefault(has_text)
        self._keep_item_button.setDefault(not has_text)
        self._sync_action_roles(has_text=has_text)

    @Slot()
    def _on_return_pressed(self) -> None:
        cleaned = self._alias_edit.text().strip()
        if cleaned:
            self._emit_answer(
                AliasPromptResponse(action=AliasAction.USE_ENGLISH, english_value=cleaned),
                accepted=True,
            )

    @Slot()
    def _on_use_english(self) -> None:
        cleaned = self._alias_edit.text().strip()
        if not cleaned:
            return
        self._emit_answer(
            AliasPromptResponse(action=AliasAction.USE_ENGLISH, english_value=cleaned),
            accepted=True,
        )

    @Slot()
    def _on_keep_item(self) -> None:
        self._emit_answer(AliasPromptResponse(action=AliasAction.KEEP_ITEM), accepted=True)

    @Slot()
    def _on_keep_scope(self) -> None:
        self._emit_answer(AliasPromptResponse(action=AliasAction.KEEP_SCOPE), accepted=True)

    @Slot()
    def _on_keep_all(self) -> None:
        self._emit_answer(AliasPromptResponse(action=AliasAction.KEEP_ALL), accepted=True)

    def withdraw(self) -> None:
        """Close the dialog programmatically without emitting ``answered``."""
        if self._withdrawn:
            return
        self._withdrawn = True
        if self.isVisible():
            super().reject()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Handle Return inside the English-name edit so whitespace cannot dismiss the dialog."""
        if (
            watched is self._alias_edit
            and event.type() == QEvent.Type.KeyPress
            and isinstance(event, QKeyEvent)
            and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        ):
            raw = self._alias_edit.text()
            if raw.strip():
                self._on_use_english()
                event.accept()
                return True
            if raw:
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def reject(self) -> None:
        """Handle Escape or window dismissal by keeping the native value for this item."""
        if self._withdrawn or self._answered:
            super().reject()
            return
        self._emit_answer(AliasPromptResponse(action=AliasAction.KEEP_ITEM), accepted=False)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Emit the safe default keep-for-this-item response once if closed directly while open."""
        if self._withdrawn or self._answered:
            event.accept()
            return
        if self.isVisible():
            self._emit_answer(AliasPromptResponse(action=AliasAction.KEEP_ITEM), accepted=False)
        event.accept()
