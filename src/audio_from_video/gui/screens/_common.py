"""Shared accessible widget helpers and palette-safe banners for GUI screens."""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_log_dir
from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QWidget,
)

from audio_from_video.config import APP_NAME
from audio_from_video.gui.controller import SETTINGS_APPLICATION, SETTINGS_ORGANIZATION
from audio_from_video.gui.resources import render_svg_pixmap

QUALITY_PRESETS: tuple[str, ...] = (
    "320k",
    "256k",
    "192k",
    "160k",
    "128k",
    "v0",
    "v2",
    "v4",
    "v6",
)


def set_accessible(widget: QWidget, name: str, description: str) -> None:
    """Assign an accessible name and description to ``widget`` and editable sub-controls."""
    widget.setAccessibleName(name)
    widget.setAccessibleDescription(description)
    if isinstance(widget, QAbstractSpinBox):
        line_edit = widget.lineEdit()
        if line_edit is not None:
            line_edit.setAccessibleName(name)
            line_edit.setAccessibleDescription(description)
    elif isinstance(widget, QComboBox):
        view = widget.view()
        if view is not None:
            view.setAccessibleName(f"{name} options")
            view.setAccessibleDescription(description)
        if widget.isEditable():
            line_edit = widget.lineEdit()
            if line_edit is not None:
                line_edit.setAccessibleName(name)
                line_edit.setAccessibleDescription(description)


def open_ui_settings(parent: QObject | None = None) -> QSettings:
    """Open the GUI ``QSettings`` store respecting ``QSettings.defaultFormat()``.

    Using the four-argument constructor ensures tests configured with
    ``IniFormat`` and ``QSettings.setPath(...)`` stay completely isolated from
    system preferences (such as macOS ``NSUserDefaults``).
    """
    return QSettings(
        QSettings.defaultFormat(),
        QSettings.Scope.UserScope,
        SETTINGS_ORGANIZATION,
        SETTINGS_APPLICATION,
        parent,
    )


def read_bool_setting(settings: QSettings, key: str, *, default: bool = False) -> bool:
    """Read a boolean flag from ``QSettings`` handling native and INI string values."""
    raw = settings.value(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return raw != 0
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off", ""):
            return False
    return default


def resolve_existing_directory(
    path: Path,
    *,
    fallback_to_parent: bool = True,
) -> tuple[Path | None, bool]:
    """Return ``(target, used_fallback)`` for opening ``path`` locally.

    Expands ``~`` and resolves ``path`` strictly to an absolute path before
    checking existence or walking parents. If ``path`` exists, returns
    ``(resolved, False)``. When ``fallback_to_parent`` is ``True`` and the
    target is missing, walks up its parent chain to the nearest existing
    directory and returns ``(parent, True)`` — never stepping out to the
    current working directory for a non-existent relative path. Returns
    ``(None, False)`` when no valid target exists.
    """
    expanded = path.expanduser()
    was_absolute = expanded.is_absolute()
    try:
        candidate = expanded.resolve()
    except OSError:
        return (None, False)
    try:
        if candidate.exists():
            return (candidate, False)
    except OSError:
        return (None, False)
    if not fallback_to_parent:
        return (None, False)
    if not was_absolute:
        try:
            cwd = Path.cwd().resolve()
        except OSError:
            return (None, False)
        for ancestor in candidate.parents:
            if ancestor == cwd or not ancestor.is_relative_to(cwd):
                break
            try:
                if ancestor.is_dir():
                    return (ancestor, True)
            except OSError:
                continue
        return (None, False)
    for ancestor in candidate.parents:
        try:
            if ancestor.is_dir():
                return (ancestor, True)
        except OSError:
            continue
    return (None, False)


def open_local_path_with_fallback(path: Path) -> tuple[bool, Path | None, bool]:
    """Open ``path`` if present, or fall back to its nearest existing parent folder.

    Returns ``(opened, resolved_target, used_parent_fallback)``.
    """
    target, used_fallback = resolve_existing_directory(path)
    if target is None:
        return (False, None, False)
    opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
    return (opened, target, used_fallback)


def open_local_path(path: Path, *, fallback_to_parent: bool = False) -> bool:
    """Ask the desktop environment to open a local file or folder.

    By default (``fallback_to_parent=False``), this resolves ``path`` strictly
    and returns ``False`` without opening a parent folder when ``path`` does
    not exist. Pass ``fallback_to_parent=True`` (or call
    :func:`open_local_path_with_fallback`) to walk up to the nearest existing
    parent directory.
    """
    if fallback_to_parent:
        opened, _target, _used_fallback = open_local_path_with_fallback(path)
        return opened
    try:
        target = path.expanduser().resolve()
    except OSError:
        return False
    try:
        if not target.exists():
            return False
    except OSError:
        return False
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))


def log_directory() -> Path:
    """Return the user log directory for ``audio-from-video``."""
    return Path(user_log_dir(APP_NAME))


def make_heading_label(text: str, *, scale: float = 1.35, parent: QWidget | None = None) -> QLabel:
    """Create a bold heading ``QLabel`` scaled relative to the system font size."""
    label = QLabel(text, parent)
    font = QFont(label.font())
    font.setBold(True)
    point_size = font.pointSizeF()
    if point_size > 0:
        font.setPointSizeF(round(point_size * scale, 1))
    label.setFont(font)
    label.setWordWrap(True)
    return label


def make_bold_font(widget: QWidget) -> QFont:
    """Return a copy of ``widget``'s current system font with bold weight."""
    font = QFont(widget.font())
    font.setBold(True)
    return font


def format_byte_size(num_bytes: int) -> str:
    """Format a byte count cleanly for display in cache summary tables."""
    size = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def format_ttl_duration(seconds: int) -> str:
    """Format a cache TTL duration in seconds as days, hours, or seconds."""
    if seconds <= 0:
        return "0 s"
    if seconds % 86_400 == 0:
        days = seconds // 86_400
        return f"{days} day" if days == 1 else f"{days} days"
    if seconds % 3_600 == 0:
        hours = seconds // 3_600
        return f"{hours} hour" if hours == 1 else f"{hours} hours"
    return f"{seconds} s"


class IconStatusBanner(QFrame):
    """System-palette banner pairing an SVG status icon with accessible text.

    Uses only system frame styling (no hard-coded background or text colors) so
    light, dark, and Windows high-contrast themes all remain readable.
    """

    def __init__(self, parent: QWidget | None = None, *, icon_size: int = 18) -> None:
        super().__init__(parent)
        self._icon_size = icon_size
        self._icon_name = "status_info"
        self._rendered_icon_key: tuple[str, float] | None = None
        self._last_text = ""
        self._last_role_label = ""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Plain)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        self.icon_label = QLabel(self)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.icon_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.text_label = QLabel(self)
        self.text_label.setWordWrap(True)
        self.text_label.setTextFormat(Qt.TextFormat.PlainText)
        self.text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.text_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        set_accessible(self, "Status banner", "Displays status, warning, or error messages.")
        set_accessible(
            self.text_label,
            "Status message",
            "Displays status, warning, or error text.",
        )

        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label, stretch=1)
        self.setVisible(False)

    @property
    def icon_name(self) -> str:
        """Return the active SVG icon stem displayed by the banner."""
        return self._icon_name

    @property
    def text(self) -> str:
        """Return the plain text currently shown by the banner."""
        return self.text_label.text()

    def _update_pixmap(self, icon_name: str, *, force: bool = False) -> None:
        dpr = max(1.0, float(self.devicePixelRatioF()))
        key = (icon_name, dpr)
        if not force and self._rendered_icon_key == key:
            return
        self._rendered_icon_key = key
        self.icon_label.setPixmap(
            render_svg_pixmap(icon_name, size=self._icon_size, device_pixel_ratio=dpr)
        )

    def changeEvent(self, event: QEvent) -> None:
        """Refresh the banner icon when the display scale or palette changes."""
        super().changeEvent(event)
        if self._rendered_icon_key is not None and event.type() in (
            QEvent.Type.DevicePixelRatioChange,
            QEvent.Type.PaletteChange,
        ):
            self._update_pixmap(self._icon_name, force=True)

    def set_message(self, icon_name: str, text: str, *, role_label: str = "Notice") -> None:
        """Show the banner with ``icon_name`` and ``text``."""
        self._icon_name = icon_name
        self._update_pixmap(icon_name)
        if text != self._last_text or role_label != self._last_role_label:
            self._last_text = text
            self._last_role_label = role_label
            self.text_label.setText(text)
            set_accessible(self, role_label, text)
            set_accessible(self.text_label, role_label, text)
        if self.isHidden():
            self.setVisible(True)

    def clear_message(self) -> None:
        """Clear and hide the banner."""
        if self._last_text:
            self._last_text = ""
            self._last_role_label = ""
            self.text_label.setText("")
        if not self.isHidden():
            self.setVisible(False)


class StatusLine(QWidget):
    """Inline horizontal row pairing a small SVG status icon with plain text."""

    def __init__(
        self,
        icon_name: str = "status_pending",
        text: str = "",
        *,
        icon_size: int = 16,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._icon_size = icon_size
        self._icon_name = icon_name
        self._rendered_icon_key: tuple[str, float] | None = None
        self._last_text: str | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.icon_label = QLabel(self)
        self.icon_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.text_label = QLabel(self)
        self.text_label.setWordWrap(True)
        self.text_label.setTextFormat(Qt.TextFormat.PlainText)
        self.text_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label, stretch=1)
        self.set_status(icon_name, text)

    @property
    def icon_name(self) -> str:
        """Return the active icon stem."""
        return self._icon_name

    @property
    def text(self) -> str:
        """Return the current label text."""
        return self.text_label.text()

    def _update_pixmap(self, icon_name: str, *, force: bool = False) -> None:
        dpr = max(1.0, float(self.devicePixelRatioF()))
        key = (icon_name, dpr)
        if not force and self._rendered_icon_key == key:
            return
        self._rendered_icon_key = key
        self.icon_label.setPixmap(
            render_svg_pixmap(icon_name, size=self._icon_size, device_pixel_ratio=dpr)
        )

    def changeEvent(self, event: QEvent) -> None:
        """Refresh the status icon when the display scale or palette changes."""
        super().changeEvent(event)
        if self._rendered_icon_key is not None and event.type() in (
            QEvent.Type.DevicePixelRatioChange,
            QEvent.Type.PaletteChange,
        ):
            self._update_pixmap(self._icon_name, force=True)

    def set_status(self, icon_name: str, text: str) -> None:
        """Update both the SVG status badge and label text together."""
        self._icon_name = icon_name
        self._update_pixmap(icon_name)
        if text != self._last_text:
            self._last_text = text
            self.text_label.setText(text)
            self.setAccessibleName(text)
            self.setAccessibleDescription(text)
