"""Shared Qt helpers for prompt dialogs using system palette colors and middle elision."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont, QFontMetrics, QPainter, QPaintEvent, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSizePolicy,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QWidget,
)

EM_DASH = "—"


class MiddleElideLabel(QLabel):
    """Single-line label that paints middle-elided text while retaining full text and tooltips.

    Keeps the full unelided string in :meth:`text` for screen readers, tooltips,
    and programmatic checks, eliding visually only when width is constrained.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setWordWrap(False)
        self._explicit_tooltip: str | None = None
        if text:
            super().setToolTip(text)

    def setText(self, text: str) -> None:
        """Update the label text and refresh the default tooltip when none is overridden."""
        super().setText(text)
        if self._explicit_tooltip is None:
            super().setToolTip(text)
        self.updateGeometry()
        self.update()

    def set_full_tooltip(self, tooltip: str | None) -> None:
        """Override the hover tooltip text (pass ``None`` to follow :meth:`text`)."""
        self._explicit_tooltip = tooltip
        super().setToolTip(self.text() if tooltip is None else tooltip)

    def minimumSizeHint(self) -> QSize:
        """Allow shrinking horizontally so elision can take effect in narrow windows."""
        fm = QFontMetrics(self.font())
        return QSize(min(80, fm.horizontalAdvance(self.text())), fm.height() + 2)

    def sizeHint(self) -> QSize:
        """Report the single-line height and full text width plus contents margins."""
        fm = QFontMetrics(self.font())
        margins = self.contentsMargins()
        width = fm.horizontalAdvance(self.text()) + margins.left() + margins.right()
        height = max(fm.height(), super().sizeHint().height())
        return QSize(width, height)

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the label text elided in the middle to fit ``contentsRect``."""
        _ = event
        painter = QPainter(self)
        rect = self.contentsRect()
        fm = QFontMetrics(self.font())
        elided = fm.elidedText(self.text(), Qt.TextElideMode.ElideMiddle, max(0, rect.width()))
        painter.setPen(self.palette().color(QPalette.ColorRole.WindowText))
        alignment = self.alignment() | Qt.AlignmentFlag.AlignVCenter
        painter.drawText(rect, alignment, elided)
        painter.end()


def escape_mnemonic(text: str) -> str:
    """Escape literal ampersands so Qt labels and buttons do not treat them as mnemonics."""
    return text.replace("&", "&&")


def make_heading_font(widget: QWidget, *, scale: float = 1.15, bold: bool = True) -> QFont:
    """Return a system-derived font scaled relative to ``widget`` without hard-coded pixels."""
    font = QFont(widget.font())
    font.setBold(bold)
    point_size = font.pointSizeF()
    if point_size > 0 and scale != 1.0:
        font.setPointSizeF(point_size * scale)
    return font


def configure_accessible(widget: QWidget, name: str, description: str = "") -> None:
    """Assign plain-text screen-reader name and optional description to ``widget``."""
    widget.setAccessibleName(name)
    if description:
        widget.setAccessibleDescription(description)


def configure_readonly_table(
    table: QTableWidget,
    headers: Sequence[str],
    *,
    accessible_name: str,
    accessible_description: str,
) -> None:
    """Apply standard read-only, row-selection, middle-elision, and accessibility settings."""
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setAlternatingRowColors(True)
    table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
    table.setWordWrap(False)
    table.setTabKeyNavigation(False)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setHighlightSections(False)
    table.horizontalHeader().setStretchLastSection(False)
    configure_accessible(table, accessible_name, accessible_description)


def make_table_item(
    text: str,
    *,
    tooltip: str | None = None,
    alignment: Qt.AlignmentFlag | None = None,
) -> QTableWidgetItem:
    """Create a read-only ``QTableWidgetItem`` with tooltip and accessible text."""
    item = QTableWidgetItem(text)
    full_tip = text if tooltip is None else tooltip
    item.setToolTip(full_tip)
    item.setData(Qt.ItemDataRole.AccessibleTextRole, full_tip)
    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    if alignment is not None:
        item.setTextAlignment(alignment)
    return item


def make_notice_banner(
    parent: QWidget,
    *,
    object_name: str,
    pixmap_kind: QStyle.StandardPixmap = QStyle.StandardPixmap.SP_MessageBoxWarning,
) -> tuple[QFrame, QLabel]:
    """Create a system-palette notice banner carrying an icon alongside wrapping text."""
    frame = QFrame(parent)
    frame.setObjectName(object_name)
    frame.setFrameShape(QFrame.Shape.StyledPanel)
    frame.setFrameShadow(QFrame.Shadow.Plain)
    frame.setBackgroundRole(QPalette.ColorRole.AlternateBase)
    frame.setAutoFillBackground(True)

    layout = QHBoxLayout(frame)
    layout.setContentsMargins(10, 8, 10, 8)
    layout.setSpacing(8)

    icon_label = QLabel(frame)
    icon_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    icon_size = max(16, frame.style().pixelMetric(QStyle.PixelMetric.PM_SmallIconSize, None, frame))
    icon = frame.style().standardIcon(pixmap_kind)
    icon_label.setPixmap(icon.pixmap(icon_size, icon_size))
    icon_label.setAccessibleName("Notice icon")
    layout.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignTop)

    text_label = QLabel(frame)
    text_label.setObjectName(f"{object_name}_text")
    text_label.setWordWrap(True)
    text_label.setTextFormat(Qt.TextFormat.PlainText)
    text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    configure_accessible(text_label, "Notice message", "Status notice shown inside this dialog.")
    layout.addWidget(text_label, stretch=1)

    frame.setVisible(False)
    return frame, text_label


def is_text_input_widget(widget: QWidget | None) -> bool:
    """Return True when ``widget`` accepts free-form text or numeric typing."""
    if widget is None:
        return False
    if isinstance(widget, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
        return True
    return isinstance(widget, QComboBox) and widget.isEditable()


def safe_disconnect(signal: object, slot: object) -> None:
    """Disconnect a Qt signal/slot pair, ignoring missing or already-deleted targets."""
    try:
        disconnect = getattr(signal, "disconnect", None)
        if callable(disconnect):
            disconnect(slot)
    except (RuntimeError, TypeError):
        pass
