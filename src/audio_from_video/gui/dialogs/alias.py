"""Window-modal dialog for supplying an English alias for non-Latin metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QObject, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QKeyEvent
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from audio_from_video.gui.bridge import AliasPrompt, PromptKind
from audio_from_video.gui.dialogs._common import (
    MiddleElideLabel,
    configure_accessible,
    escape_mnemonic,
    make_heading_font,
)
from audio_from_video.planner import AliasAction, AliasPromptResponse

if TYPE_CHECKING:
    from audio_from_video.gui.bridge import PromptRequest


def _format_field_kind(field_kind: str) -> str:
    """Turn an internal field name such as ``album_artist`` into readable text."""
    return field_kind.replace("_", " ").strip() or "value"


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
        self.resize(560, 240)
        configure_accessible(
            self,
            "English name alias",
            (
                f"Enter an English alias for {field_label} {self._payload.original_value}, "
                "or keep the native name."
            ),
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        self._heading_label = QLabel(f"Missing English alias for {field_label}", self)
        self._heading_label.setObjectName("alias_heading_label")
        self._heading_label.setFont(make_heading_font(self._heading_label))
        self._heading_label.setTextFormat(Qt.TextFormat.PlainText)
        self._heading_label.setWordWrap(True)
        root.addWidget(self._heading_label)

        self._file_label = MiddleElideLabel(f"File: {request.media_input.path.name}", self)
        self._file_label.setObjectName("alias_file_label")
        self._file_label.set_full_tooltip(str(request.media_input.path))
        configure_accessible(
            self._file_label,
            "Source file",
            f"Source file name: {request.media_input.path.name}.",
        )
        root.addWidget(self._file_label)

        form = QFormLayout()
        form.setContentsMargins(0, 4, 0, 4)
        form.setSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._kind_value_label = QLabel(field_label, self)
        self._kind_value_label.setObjectName("alias_field_kind_label")
        self._kind_value_label.setTextFormat(Qt.TextFormat.PlainText)
        form.addRow("Field:", self._kind_value_label)

        self._native_value_label = MiddleElideLabel(self._payload.original_value, self)
        self._native_value_label.setObjectName("alias_original_value_label")
        self._native_value_label.setFont(make_heading_font(self._native_value_label, scale=1.0))
        configure_accessible(
            self._native_value_label,
            "Native value",
            f"Original non-Latin {field_label}: {self._payload.original_value}.",
        )
        form.addRow("Native value:", self._native_value_label)

        english_buddy = QLabel("&English name:", self)
        self._alias_edit = QLineEdit(self)
        self._alias_edit.setObjectName("alias_english_edit")
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
        form.addRow(english_buddy, self._alias_edit)

        root.addLayout(form)

        note_label = QLabel(
            "Enter an English name above, keep the native name for this item, keep it for "
            f"{scope_label}, or keep native names for the rest of the run.",
            self,
        )
        note_label.setWordWrap(True)
        note_label.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(note_label)

        scope_row = QHBoxLayout()
        scope_row.setSpacing(8)

        escaped_scope = escape_mnemonic(scope_label)
        self._keep_scope_button = QPushButton(f"Keep &for {escaped_scope}", self)
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
        self._keep_all_button.clicked.connect(self._on_keep_all)
        scope_row.addWidget(self._keep_all_button)
        scope_row.addStretch(1)
        root.addLayout(scope_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        action_row.addStretch(1)

        self._keep_item_button = QPushButton("Keep for this &item", self)
        self._keep_item_button.setObjectName("alias_keep_item_button")
        self._keep_item_button.setAutoDefault(False)
        self._keep_item_button.setDefault(True)
        self._keep_item_button.setToolTip("Keep the native name for this track only.")
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

        root.addLayout(action_row)

        self.setTabOrder(self._alias_edit, self._use_english_button)
        self.setTabOrder(self._use_english_button, self._keep_item_button)
        self.setTabOrder(self._keep_item_button, self._keep_scope_button)
        self.setTabOrder(self._keep_scope_button, self._keep_all_button)
        self._alias_edit.setFocus()

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
