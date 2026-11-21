"""Shared Qt helpers for prompt dialogs using semantic theme roles and middle elision."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import (
    QFont,
    QFontMetrics,
    QGuiApplication,
    QIcon,
    QPainter,
    QPaintEvent,
    QPalette,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyleOptionButton,
    QStylePainter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from audio_from_video.gui.resources import render_svg_pixmap
from audio_from_video.gui.theme import (
    is_dark_palette,
    is_theme_updating,
    repolish,
    themed_button_icon,
)

EM_DASH = "—"

_STANDARD_PIXMAP_TO_SVG: dict[QStyle.StandardPixmap, tuple[str, str]] = {
    QStyle.StandardPixmap.SP_MessageBoxWarning: ("status_warn", "warn"),
    QStyle.StandardPixmap.SP_MessageBoxCritical: ("status_error", "error"),
    QStyle.StandardPixmap.SP_MessageBoxInformation: ("status_info", "info"),
    QStyle.StandardPixmap.SP_MessageBoxQuestion: ("status_unmatched", "unmatched"),
}


def _scale_font(base: QFont, *, scale: float, bold: bool | None = None) -> QFont:
    """Return a copy of ``base`` scaled by ``scale``, handling pixel-sized fonts."""
    font = QFont(base)
    if bold is not None:
        font.setBold(bold)
    if scale != 1.0:
        point_size = font.pointSizeF()
        if point_size > 0:
            font.setPointSizeF(round(point_size * scale, 1))
        else:
            pixel_size = font.pixelSize()
            if pixel_size > 0:
                font.setPixelSize(max(1, round(pixel_size * scale)))
    return font


def _strip_qt_mnemonic(text: str) -> str:
    """Convert Qt button/label mnemonic markup into plain text for width measurement."""
    out: list[str] = []
    idx = 0
    length = len(text)
    while idx < length:
        ch = text[idx]
        if ch == "&":
            if idx + 1 < length and text[idx + 1] == "&":
                out.append("&")
                idx += 2
                continue
            idx += 1
            continue
        out.append(ch)
        idx += 1
    return "".join(out)


class ReactiveSvgLabel(QLabel):
    """Non-focusable SVG pixmap label that refreshes on palette and DPR changes."""

    def __init__(
        self,
        icon_name: str,
        *,
        icon_size: int = 18,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._icon_name = icon_name
        self._icon_size = max(1, int(icon_size))
        self._rendered_key: tuple[str, float, bool, str] | None = None
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._refresh_pixmap(force=True)

    @property
    def icon_name(self) -> str:
        """Return the active SVG icon stem displayed by this label."""
        return self._icon_name

    def set_icon_name(self, icon_name: str) -> None:
        """Switch the rendered SVG icon stem and repaint immediately."""
        if icon_name != self._icon_name:
            self._icon_name = icon_name
            self._refresh_pixmap(force=True)

    def _refresh_pixmap(self, *, force: bool = False) -> None:
        dpr = max(1.0, float(self.devicePixelRatioF()))
        pal = self.palette()
        if not self.testAttribute(Qt.WidgetAttribute.WA_SetPalette):
            app = QGuiApplication.instance()
            if isinstance(app, QGuiApplication):
                pal = app.palette()
        is_dark = is_dark_palette(pal)
        fg_key = pal.color(QPalette.ColorGroup.Active, QPalette.ColorRole.ButtonText).name()
        key = (self._icon_name, dpr, is_dark, fg_key)
        if not force and self._rendered_key == key:
            return
        self._rendered_key = key
        self.setPixmap(
            render_svg_pixmap(
                self._icon_name,
                size=self._icon_size,
                device_pixel_ratio=dpr,
                dark=is_dark,
            )
        )

    def _refresh_themed_visuals(self) -> None:
        """Refresh the cached SVG pixmap after an application palette or theme update."""
        self._refresh_pixmap(force=True)

    def changeEvent(self, event: QEvent) -> None:
        """Re-render the vector pixmap when display scale, palette, or style changes."""
        super().changeEvent(event)
        etype = event.type()
        if etype == QEvent.Type.DevicePixelRatioChange or (
            not is_theme_updating()
            and etype
            in (
                QEvent.Type.PaletteChange,
                QEvent.Type.ApplicationPaletteChange,
                QEvent.Type.StyleChange,
            )
        ):
            self._refresh_pixmap(force=True)


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
        """Allow shrinking horizontally while reserving room for tall glyphs and margins."""
        fm = QFontMetrics(self.font())
        margins = self.contentsMargins()
        text_w = min(80, fm.horizontalAdvance(self.text()))
        height = max(fm.height() + 4, fm.lineSpacing() + 2) + margins.top() + margins.bottom()
        return QSize(text_w + margins.left() + margins.right(), height)

    def sizeHint(self) -> QSize:
        """Report the single-line height and full text width plus contents margins."""
        fm = QFontMetrics(self.font())
        margins = self.contentsMargins()
        width = fm.horizontalAdvance(self.text()) + margins.left() + margins.right()
        height = max(
            fm.height() + margins.top() + margins.bottom() + 4,
            super().sizeHint().height(),
        )
        return QSize(width, height)

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the label text elided in the middle inside ``contentsRect``."""
        _ = event
        painter = QPainter(self)
        rect = self.contentsRect()
        fm = QFontMetrics(self.font())
        elided = fm.elidedText(self.text(), Qt.TextElideMode.ElideMiddle, max(0, rect.width()))
        group = (
            QPalette.ColorGroup.Disabled
            if not self.isEnabled()
            else self.palette().currentColorGroup()
        )
        painter.setPen(self.palette().color(group, self.foregroundRole()))
        alignment = self.alignment() | Qt.AlignmentFlag.AlignVCenter
        painter.drawText(rect, alignment, elided)
        painter.end()


class ElidingPushButton(QPushButton):
    """``QPushButton`` that visually middle-elides overly long labels without mutating ``text()``.

    Useful for dynamic scope labels (such as bilingual album titles) where exact
    unelided mnemonic text must stay intact for Qt keyboard shortcuts, tooltips,
    and screen readers while avoiding unbounded horizontal dialog expansion.
    """

    def __init__(
        self,
        text: str = "",
        parent: QWidget | None = None,
        *,
        max_hint_width: int = 260,
    ) -> None:
        super().__init__(text, parent)
        self._max_hint_width = max(80, int(max_hint_width))
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    @property
    def max_hint_width(self) -> int:
        """Return the maximum preferred horizontal hint width in logical pixels."""
        return self._max_hint_width

    def set_max_hint_width(self, width: int) -> None:
        """Update the maximum preferred horizontal hint width and re-layout."""
        self._max_hint_width = max(80, int(width))
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        """Bound button preferred width so long labels cannot inflate dialog minimum widths."""
        hint = super().sizeHint()
        return QSize(min(hint.width(), self._max_hint_width), hint.height())

    def minimumSizeHint(self) -> QSize:
        """Allow the button to shrink cleanly inside narrow modal sheets."""
        hint = super().minimumSizeHint()
        cap = min(140, self._max_hint_width)
        return QSize(min(hint.width(), cap), hint.height())

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the styled button using middle-elided text whenever the full label overflows."""
        _ = event
        option = QStyleOptionButton()
        self.initStyleOption(option)

        content_rect = self.style().subElementRect(
            QStyle.SubElement.SE_PushButtonContents,
            option,
            self,
        )
        available = max(0, content_rect.width() - 8)
        if not option.icon.isNull():
            available = max(0, available - option.iconSize.width() - 6)

        raw_text = option.text
        plain_text = _strip_qt_mnemonic(raw_text)
        fm = QFontMetrics(self.font())
        if available > 0 and fm.horizontalAdvance(plain_text) > available:
            elided = fm.elidedText(plain_text, Qt.TextElideMode.ElideMiddle, available)
            # Escape literal ampersands inside elided copy so Qt does not turn them into mnemonics.
            option.text = escape_mnemonic(elided)

        painter = QStylePainter(self)
        painter.drawControl(QStyle.ControlElement.CE_PushButton, option)


def escape_mnemonic(text: str) -> str:
    """Escape literal ampersands so Qt labels and buttons do not treat them as mnemonics."""
    return text.replace("&", "&&")


def make_heading_font(widget: QWidget, *, scale: float = 1.15, bold: bool = True) -> QFont:
    """Return a system-derived font scaled relative to ``widget`` without hard-coded pixels."""
    return _scale_font(widget.font(), scale=scale, bold=bold)


def configure_accessible(widget: QWidget, name: str, description: str = "") -> None:
    """Assign plain-text screen-reader name and optional description to ``widget``."""
    widget.setAccessibleName(name)
    if description:
        widget.setAccessibleDescription(description)


def set_button_role(button: QPushButton, role: str) -> None:
    """Assign a semantic ``afvRole`` property (for example ``'primary'``) and repolish."""
    button.setProperty("afvRole", role)
    repolish(button)


def dialog_button_icon(icon_name: str, *, on_primary: bool) -> QIcon:
    """Return a scalable button icon tinted for primary accent or outlined dialog buttons."""
    return themed_button_icon(icon_name, on_primary=on_primary)


def make_divider_line(parent: QWidget | None = None) -> QFrame:
    """Create a sleek horizontal hairline divider themed via ``afvDivider``."""
    line = QFrame(parent)
    line.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Plain)
    line.setProperty("afvDivider", True)
    line.setFixedHeight(1)
    return line


def make_context_card(
    parent: QWidget | None = None,
    *,
    object_name: str = "",
) -> QFrame:
    """Create a recessed, rounded metadata context frame themed via ``afvSurface``."""
    frame = QFrame(parent)
    if object_name:
        frame.setObjectName(object_name)
    frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    frame.setFrameShape(QFrame.Shape.StyledPanel)
    frame.setFrameShadow(QFrame.Shadow.Plain)
    frame.setProperty("afvSurface", "context-box")
    return frame


def make_dialog_header(
    parent: QWidget,
    *,
    icon_name: str,
    heading_label: QLabel,
    subtitle_label: QLabel | None = None,
    icon_size: int = 24,
) -> tuple[QHBoxLayout, ReactiveSvgLabel]:
    """Arrange a vector SVG badge alongside ``heading_label`` and an optional subtitle."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(12)

    icon_label = ReactiveSvgLabel(icon_name, icon_size=icon_size, parent=parent)
    icon_label.setAccessibleName("Dialog status icon")
    row.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignTop)

    text_col = QVBoxLayout()
    text_col.setContentsMargins(0, 0, 0, 0)
    text_col.setSpacing(3)
    text_col.addWidget(heading_label)
    if subtitle_label is not None:
        subtitle_label.setProperty("afvText", "subtitle")
        text_col.addWidget(subtitle_label)
    row.addLayout(text_col, stretch=1)
    return row, icon_label


def configure_readonly_table(
    table: QTableWidget,
    headers: Sequence[str],
    *,
    accessible_name: str,
    accessible_description: str,
    row_height: int | None = None,
) -> None:
    """Apply standard read-only, row-selection, middle-elision, and accessibility settings."""
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setAlternatingRowColors(True)
    table.setShowGrid(False)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
    table.setWordWrap(False)
    table.setTabKeyNavigation(False)
    v_header = table.verticalHeader()
    v_header.setVisible(False)
    fm = QFontMetrics(table.font())
    resolved_height = max(32, fm.height() + 12) if row_height is None else max(24, int(row_height))
    v_header.setDefaultSectionSize(resolved_height)
    h_header = table.horizontalHeader()
    h_header.setHighlightSections(False)
    h_header.setStretchLastSection(False)
    h_header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
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
    """Create a themed semantic notice banner pairing a crisp SVG icon with wrapping text."""
    icon_name, tone = _STANDARD_PIXMAP_TO_SVG.get(pixmap_kind, ("status_warn", "warn"))

    frame = QFrame(parent)
    frame.setObjectName(object_name)
    frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    frame.setFrameShape(QFrame.Shape.StyledPanel)
    frame.setFrameShadow(QFrame.Shadow.Plain)
    frame.setBackgroundRole(QPalette.ColorRole.AlternateBase)
    frame.setProperty("afvBanner", True)
    frame.setProperty("afvTone", tone)
    repolish(frame)

    layout = QHBoxLayout(frame)
    layout.setContentsMargins(8, 7, 12, 7)
    layout.setSpacing(9)

    stripe = QFrame(frame)
    stripe.setObjectName(f"{object_name}_stripe")
    stripe.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    stripe.setFrameShape(QFrame.Shape.NoFrame)
    stripe.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    stripe.setFixedWidth(3)
    stripe.setMinimumHeight(14)
    stripe.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
    stripe.setProperty("afvBannerStripe", True)
    layout.addWidget(stripe)

    icon_label = ReactiveSvgLabel(icon_name, icon_size=18, parent=frame)
    icon_label.setAccessibleName("Notice icon")
    icon_label.setAccessibleDescription("Visual status indicator for this notice banner.")
    layout.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignTop)

    text_label = QLabel(frame)
    text_label.setObjectName(f"{object_name}_text")
    text_label.setWordWrap(True)
    text_label.setTextFormat(Qt.TextFormat.PlainText)
    text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    configure_accessible(text_label, "Notice message", "Status notice shown inside this dialog.")
    layout.addWidget(text_label, stretch=1)

    repolish(frame, include_children=True)
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
