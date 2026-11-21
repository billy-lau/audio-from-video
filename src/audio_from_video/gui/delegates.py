"""Custom ``QStyledItemDelegate`` renderers for ``PlanTableModel`` views."""

from __future__ import annotations

from PySide6.QtCore import (
    QAbstractProxyModel,
    QModelIndex,
    QPersistentModelIndex,
    QRectF,
    QSize,
    Qt,
)
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionViewItem, QWidget

from audio_from_video.gui.models.plan_table import (
    COL_ACTION,
    COL_WARNINGS,
    PLAN_ITEM_ROLE,
    is_failed_action,
    is_skipped_action,
)
from audio_from_video.gui.resources import render_svg_pixmap
from audio_from_video.gui.theme import (
    is_high_contrast_mode,
    tokens_for_palette,
    tone_for_icon,
)
from audio_from_video.models import PlanItem, RunOutcome

_LABEL_TO_ICON: dict[str, str] = {
    "done": "status_ok",
    "ready": "status_extract",
    "waiting": "status_info",
    "probing": "status_working",
    "looking up": "status_working",
    "extracting": "status_working",
    "attention": "status_warn",
    "partial": "status_warn",
    "failed": "status_error",
    "unmatched": "status_unmatched",
    "done (unmatched)": "status_unmatched",
    "queued": "status_pending",
    "skipped": "status_skip",
    "skipped (no audio)": "status_skip",
    "cancelled": "status_cancelled",
    "not run": "status_cancelled",
}

_PILL_ICON_SIZE = 14
_PILL_H_PAD = 8
_PILL_V_MARGIN = 4
_PILL_ICON_GAP = 5
_CELL_H_PAD = 8


def _source_model_and_row(
    index: QModelIndex | QPersistentModelIndex,
) -> tuple[object, int]:
    """Unwrap any ``QAbstractProxyModel`` chain to return ``(source_model, source_row)``."""
    model = index.model()
    current = index
    while isinstance(model, QAbstractProxyModel):
        current = model.mapToSource(current)
        model = model.sourceModel()
    return model, current.row()


def _resolve_status_icon(
    index: QModelIndex | QPersistentModelIndex,
    display_text: str,
) -> str:
    """Resolve the bundled ``status_*`` icon stem for ``index`` without mutating the model."""
    model, source_row = _source_model_and_row(index)
    status_info_fn = getattr(model, "status_info_at", None)
    if callable(status_info_fn) and source_row >= 0:
        try:
            _label, icon_name = status_info_fn(source_row)
            if isinstance(icon_name, str) and icon_name:
                return icon_name
        except (IndexError, ValueError, TypeError):
            pass

    key = display_text.strip().casefold()
    return _LABEL_TO_ICON.get(key, "status_info")


def _resolve_row_outcome(index: QModelIndex | QPersistentModelIndex) -> RunOutcome | None:
    """Return the latest ``RunOutcome`` for ``index`` when backed by ``PlanTableModel``."""
    model, source_row = _source_model_and_row(index)
    outcome_fn = getattr(model, "outcome_at", None)
    if callable(outcome_fn) and source_row >= 0:
        try:
            outcome = outcome_fn(source_row)
            if isinstance(outcome, RunOutcome):
                return outcome
        except (IndexError, ValueError, TypeError):
            pass
    return None


def _paint_cell_background(
    painter: QPainter,
    option: QStyleOptionViewItem,
    *,
    is_alternate: bool,
    high_contrast: bool,
) -> None:
    """Paint the cell backdrop, handling hover, selection, and crisp row dividers."""
    rect = option.rect
    selected = bool(option.state & QStyle.StateFlag.State_Selected)
    hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

    if high_contrast:
        if selected:
            bg = option.palette.color(QPalette.ColorRole.Highlight)
        elif is_alternate:
            bg = option.palette.color(QPalette.ColorRole.AlternateBase)
        else:
            bg = option.palette.color(QPalette.ColorRole.Base)
        painter.fillRect(rect, bg)
        return

    tokens = tokens_for_palette(option.palette)
    if selected:
        bg = QColor(tokens.selection_bg)
    elif hovered:
        bg = QColor(tokens.bg_surface_hover)
    elif is_alternate:
        bg = QColor(tokens.bg_surface_alt)
    else:
        bg = QColor(tokens.bg_surface)
    painter.fillRect(rect, bg)

    # Subtle crisp horizontal bottom hairline joining adjacent cells uniformly.
    painter.save()
    painter.setPen(QPen(QColor(tokens.border_subtle), 1.0))
    y = rect.bottom()
    painter.drawLine(rect.left(), y, rect.right(), y)
    painter.restore()


class StatusPillDelegate(QStyledItemDelegate):
    """Paints ``COL_STATUS`` as a crisp semantic capsule pairing an SVG icon with text.

    Automatically adapts to Light, Dark, and High-Contrast themes by reading the
    active palette on every paint pass, leaving the underlying ``PlanTableModel``
    roles, labels, tooltips, and accessibility strings completely unchanged.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

    def sizeHint(
        self,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> QSize:
        """Report comfortable horizontal pill width and vertical breathing room."""
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        text = str(index.data(int(Qt.ItemDataRole.DisplayRole)) or "")
        pill_font = QFont(opt.font)
        pill_font.setWeight(QFont.Weight.DemiBold)
        fm = QFontMetrics(pill_font)
        text_w = fm.horizontalAdvance(text) + (6 if text else 0)
        pill_w = _PILL_H_PAD * 2 + _PILL_ICON_SIZE + (_PILL_ICON_GAP + text_w if text else 0)
        row_h = max(32, fm.height() + 14)
        return QSize(pill_w + _CELL_H_PAD * 2, row_h)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        """Render the capsule status pill inside ``option.rect``."""
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        is_alt = bool(opt.features & QStyleOptionViewItem.ViewItemFeature.Alternate)
        high_contrast = is_high_contrast_mode(palette=opt.palette)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        _paint_cell_background(painter, opt, is_alternate=is_alt, high_contrast=high_contrast)

        display_text = str(index.data(int(Qt.ItemDataRole.DisplayRole)) or "").strip()
        icon_name = _resolve_status_icon(index, display_text)
        tone_name = tone_for_icon(icon_name)
        tokens = tokens_for_palette(opt.palette)
        tone = tokens.tone(tone_name)

        pill_font = QFont(opt.font)
        pill_font.setWeight(QFont.Weight.DemiBold)
        fm = QFontMetrics(pill_font)

        cell = opt.rect
        avail_w = max(0, cell.width() - _CELL_H_PAD * 2)
        max_pill_h = max(18, cell.height() - _PILL_V_MARGIN * 2)
        pill_h = min(max_pill_h, max(22, fm.height() + 6))
        pill_y = cell.top() + (cell.height() - pill_h) / 2.0

        text_w = fm.horizontalAdvance(display_text) if display_text else 0
        # Add a modest rounding cushion so subpixel kerning on DemiBold never elides short labels.
        text_cushion = 6 if display_text else 0
        desired_w = (
            _PILL_H_PAD * 2
            + _PILL_ICON_SIZE
            + (_PILL_ICON_GAP + text_w + text_cushion if text_w else 0)
        )
        pill_w = min(float(avail_w), float(desired_w))

        # Tight columns: center just the vector icon if there is no room for a label.
        show_text = bool(display_text) and avail_w >= (_PILL_H_PAD * 2 + _PILL_ICON_SIZE + 16)
        if not show_text:
            pill_w = min(float(avail_w), float(_PILL_ICON_SIZE + 10))

        pill_x = float(cell.left() + _CELL_H_PAD)
        pill_rect = QRectF(pill_x, pill_y, max(16.0, pill_w), float(pill_h))
        radius = min(pill_rect.height() / 2.0, 12.0)

        selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        if high_contrast:
            bg_color = opt.palette.color(QPalette.ColorRole.Base)
            fg_color = opt.palette.color(QPalette.ColorRole.WindowText)
            border_color = opt.palette.color(
                QPalette.ColorRole.HighlightedText if selected else QPalette.ColorRole.WindowText
            )
            border_width = 1.5
        else:
            bg_color = QColor(tone.bg)
            fg_color = QColor(tone.fg)
            border_color = QColor(tone.border)
            border_width = 1.0

        path = QPainterPath()
        path.addRoundedRect(pill_rect.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        painter.fillPath(path, bg_color)
        painter.setPen(QPen(border_color, border_width))
        painter.drawPath(path)

        # Render the crisp vector status icon at the target paint device DPR.
        device = painter.device()
        dpr = 2.0
        if device is not None and hasattr(device, "devicePixelRatioF"):
            dpr = max(1.0, float(device.devicePixelRatioF()))
        pixmap = render_svg_pixmap(
            icon_name,
            size=_PILL_ICON_SIZE,
            device_pixel_ratio=dpr,
            dark=tokens.is_dark if not high_contrast else None,
        )

        icon_y = pill_rect.top() + (pill_rect.height() - _PILL_ICON_SIZE) / 2.0
        if show_text:
            icon_x = pill_rect.left() + _PILL_H_PAD
            painter.drawPixmap(round(icon_x), round(icon_y), pixmap)

            text_left = icon_x + _PILL_ICON_SIZE + _PILL_ICON_GAP
            text_avail = max(0.0, (pill_rect.left() + pill_rect.width()) - _PILL_H_PAD - text_left)
            if text_w <= text_avail:
                elided = display_text
            else:
                elided = fm.elidedText(
                    display_text,
                    Qt.TextElideMode.ElideRight,
                    max(0, int(text_avail)),
                )
            text_rect = QRectF(
                text_left,
                pill_rect.top(),
                max(0.0, text_avail + 2.0),
                pill_rect.height(),
            )
            painter.setFont(pill_font)
            painter.setPen(fg_color)
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                elided,
            )
        else:
            icon_x = pill_rect.left() + (pill_rect.width() - _PILL_ICON_SIZE) / 2.0
            painter.drawPixmap(round(icon_x), round(icon_y), pixmap)

        painter.restore()


class PaddedTextDelegate(QStyledItemDelegate):
    """Lightweight text cell delegate adding clean horizontal padding and muted states.

    Renders ``"—"`` placeholder values and ``Skipped`` rows in muted secondary
    ink while keeping active tracks and failure diagnostics sharply legible across
    light, dark, and high-contrast themes.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        elide_mode: Qt.TextElideMode = Qt.TextElideMode.ElideMiddle,
        horizontal_padding: int = _CELL_H_PAD,
    ) -> None:
        super().__init__(parent)
        self._elide_mode = elide_mode
        self._h_pad = max(2, int(horizontal_padding))

    def sizeHint(
        self,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> QSize:
        """Return comfortable single-line height matching ``StatusPillDelegate``."""
        base = super().sizeHint(option, index)
        fm = QFontMetrics(option.font)
        return QSize(base.width() + self._h_pad * 2, max(32, fm.height() + 14))

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        """Paint left-aligned, elided text with balanced row padding."""
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        is_alt = bool(opt.features & QStyleOptionViewItem.ViewItemFeature.Alternate)
        selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        high_contrast = is_high_contrast_mode(palette=opt.palette)

        painter.save()
        _paint_cell_background(painter, opt, is_alternate=is_alt, high_contrast=high_contrast)

        text = str(index.data(int(Qt.ItemDataRole.DisplayRole)) or "")
        if not text:
            painter.restore()
            return

        item = index.data(PLAN_ITEM_ROLE)
        col = index.column()
        tokens = tokens_for_palette(opt.palette)
        font = QFont(opt.font)

        outcome = _resolve_row_outcome(index)
        is_failed = outcome == RunOutcome.FAILED or (
            isinstance(item, PlanItem) and is_failed_action(item.action)
        )
        is_skipped = outcome in (RunOutcome.SKIPPED, RunOutcome.SKIPPED_NO_AUDIO) or (
            isinstance(item, PlanItem) and is_skipped_action(item.action)
        )

        if high_contrast:
            color = opt.palette.color(
                QPalette.ColorRole.HighlightedText if selected else QPalette.ColorRole.Text
            )
        elif selected:
            color = QColor(tokens.selection_fg)
        elif text == "—":
            color = QColor(tokens.text_secondary)
        elif is_failed and col == COL_WARNINGS:
            color = QColor(tokens.error.fg)
            font.setWeight(QFont.Weight.DemiBold)
        elif is_skipped or col == COL_ACTION:
            color = QColor(tokens.text_secondary)
        else:
            color = QColor(tokens.text_primary)

        painter.setFont(font)
        painter.setPen(color)
        text_rect = opt.rect.adjusted(self._h_pad, 0, -self._h_pad, 0)
        fm = QFontMetrics(font)
        elide = Qt.TextElideMode.ElideRight if col == COL_WARNINGS else self._elide_mode
        elided = fm.elidedText(text, elide, max(0, text_rect.width()))
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            elided,
        )
        painter.restore()


__all__ = [
    "PaddedTextDelegate",
    "StatusPillDelegate",
]
