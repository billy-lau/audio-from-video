"""Shared accessible widget helpers, themed primitives, and status banners for GUI screens."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from platformdirs import user_log_dir
from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QUrl
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QGuiApplication,
    QPalette,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableView,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from audio_from_video.config import (
    APP_NAME,
    ConfigError,
    validate_mb_contact,
    validate_mb_user_agent,
)
from audio_from_video.gui.controller import (
    SETTINGS_APPLICATION,
    SETTINGS_ORGANIZATION,
    ControllerState,
)
from audio_from_video.gui.dialogs._common import MiddleElideLabel
from audio_from_video.gui.resources import load_icon, render_svg_pixmap
from audio_from_video.gui.theme import (
    is_dark_palette,
    is_theme_updating,
    repolish,
    set_widget_role,
    themed_button_icon,
    tone_for_icon,
)

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

_WORKFLOW_STEPS: tuple[tuple[str, str], ...] = (
    ("inputs", "Inputs"),
    ("planning", "Plan"),
    ("review", "Review"),
    ("running", "Extract"),
    ("summary", "Summary"),
)


def join_oxford(items: Sequence[str]) -> str:
    """Format ``items`` as an English list using the Oxford comma."""
    cleaned = [str(item) for item in items if str(item)]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def find_invalid_env_overrides(active_env: Mapping[str, str]) -> dict[str, tuple[str, str]]:
    """Return ``{attr_path: (env_var, error)}`` for invalid ``AFV_*`` overrides."""
    issues: dict[str, tuple[str, str]] = {}
    raw_contact = active_env.get("AFV_MB_CONTACT", "").strip()
    if raw_contact:
        try:
            validate_mb_contact(raw_contact)
        except ConfigError as exc:
            issues["musicbrainz.contact"] = ("AFV_MB_CONTACT", str(exc))
    raw_ua = active_env.get("AFV_MB_USER_AGENT", "").strip()
    if raw_ua:
        try:
            validate_mb_user_agent(raw_ua)
        except ConfigError as exc:
            issues["musicbrainz.user_agent_override"] = ("AFV_MB_USER_AGENT", str(exc))
    return issues


def _scale_font(
    base: QFont,
    *,
    scale: float,
    weight: QFont.Weight | None = None,
    bold: bool | None = None,
) -> QFont:
    """Return a copy of ``base`` scaled cleanly with ``pixelSize`` fallback."""
    font = QFont(base)
    if bold is not None:
        font.setBold(bold)
    elif weight is not None:
        font.setWeight(weight)
    if scale != 1.0:
        point_size = font.pointSizeF()
        if point_size > 0:
            font.setPointSizeF(round(point_size * scale, 1))
        else:
            pixel_size = font.pixelSize()
            if pixel_size > 0:
                font.setPixelSize(max(1, round(pixel_size * scale)))
    return font


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
    label.setFont(_scale_font(label.font(), scale=scale, bold=True))
    label.setWordWrap(True)
    label.setProperty("afvText", "hero" if scale >= 1.5 else "title")
    return label


def make_section_label(text: str, *, scale: float = 1.05, parent: QWidget | None = None) -> QLabel:
    """Create a semi-bold section title ``QLabel`` scaled cleanly to the system font."""
    label = QLabel(text, parent)
    label.setFont(_scale_font(label.font(), scale=scale, weight=QFont.Weight.DemiBold))
    label.setWordWrap(True)
    label.setProperty("afvText", "section")
    return label


def make_muted_label(
    text: str = "",
    *,
    scale: float = 0.92,
    word_wrap: bool = True,
    parent: QWidget | None = None,
) -> QLabel:
    """Create a subtle secondary-tone ``QLabel`` suitable for subtitles and helper text."""
    label = QLabel(text, parent)
    label.setFont(_scale_font(label.font(), scale=scale))
    label.setWordWrap(word_wrap)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setProperty("afvText", "muted")
    return label


def make_bold_font(widget: QWidget) -> QFont:
    """Return a copy of ``widget``'s current system font with bold weight."""
    font = QFont(widget.font())
    font.setBold(True)
    return font


def make_mono_font(widget: QWidget | None = None) -> QFont:
    """Return the OS monospace system font matched to ``widget``'s text size."""
    mono = QFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
    if widget is not None:
        base = widget.font()
        point_size = base.pointSizeF()
        if point_size > 0:
            mono.setPointSizeF(max(8.0, round(point_size * 0.92, 1)))
        elif base.pixelSize() > 0:
            mono.setPixelSize(max(10, round(base.pixelSize() * 0.92)))
    return mono


def make_divider(parent: QWidget | None = None) -> QFrame:
    """Create a crisp horizontal hairline separator tagged for centralized theming."""
    line = QFrame(parent)
    line.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Plain)
    line.setProperty("afvDivider", True)
    line.setFixedHeight(1)
    return line


def _effective_palette(widget: QWidget) -> QPalette:
    """Return the live application palette unless ``widget`` has an explicit palette override."""
    if not widget.testAttribute(Qt.WidgetAttribute.WA_SetPalette):
        app = QGuiApplication.instance()
        if isinstance(app, QGuiApplication):
            return app.palette()
    return widget.palette()


def apply_primary_button_role(
    button: QPushButton,
    *,
    icon_name: str | None = None,
) -> None:
    """Style ``button`` as the primary accented CTA button without inline QSS.

    Assigns bold font, tags the dynamic ``afvRole="primary"`` property handled
    by the application-level stylesheet, and (when ``icon_name`` is provided)
    attaches a scalable icon matching the filled accent foreground (white in
    Studio Slate or ``HighlightedText`` in OS high-contrast mode).
    """
    button.setFont(make_bold_font(button))
    set_widget_role(button, "afvRole", "primary")
    if icon_name is not None:
        button.setProperty("afvPrimaryIconName", icon_name)
        button.setIcon(themed_button_icon(icon_name, on_primary=True))


def apply_secondary_button_role(
    button: QPushButton,
    *,
    icon_name: str | None = None,
) -> None:
    """Style ``button`` as a crisp outlined secondary button."""
    if icon_name is not None:
        button.setIcon(load_icon(icon_name))
    set_widget_role(button, "afvRole", "secondary")


def apply_danger_button_role(
    button: QPushButton,
    *,
    icon_name: str | None = None,
) -> None:
    """Style ``button`` as an outlined destructive action button."""
    set_widget_role(button, "afvRole", "danger")
    if icon_name is not None:
        button.setProperty("afvDangerIconName", icon_name)
        button.setIcon(themed_button_icon(icon_name, on_primary=False, on_danger=True))


def apply_ghost_button_role(
    button: QPushButton,
    *,
    icon_name: str | None = None,
) -> None:
    """Style ``button`` as a quiet subtle outline/ghost button."""
    if icon_name is not None:
        button.setIcon(load_icon(icon_name))
    set_widget_role(button, "afvRole", "ghost")


def apply_filter_pill_role(button: QPushButton) -> None:
    """Style a checkable ``QPushButton`` as a crisp segmented filter pill."""
    set_widget_role(button, "afvRole", "filter-pill")


def configure_table_view(
    table: QTableView | QTableWidget,
    *,
    row_height: int | None = None,
) -> None:
    """Apply cross-platform scroll, density, alignment, and focus polish to a table."""
    table.setTabKeyNavigation(False)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.setShowGrid(False)
    table.setAlternatingRowColors(True)
    table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
    table.setWordWrap(False)

    h_header = table.horizontalHeader()
    if h_header is not None:
        h_header.setHighlightSections(False)
        h_header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    v_header = table.verticalHeader()
    if v_header is not None:
        v_header.setVisible(False)
        fm = QFontMetrics(table.font())
        resolved = max(34, fm.height() + 14) if row_height is None else max(24, int(row_height))
        v_header.setDefaultSectionSize(resolved)


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


class ThemedSvgLabel(QLabel):
    """DPR- and theme-reactive ``QLabel`` rendering a bundled SVG icon.

    Automatically re-rasterizes its pixmap whenever the display scale factor,
    palette, or application style changes, fixing stale light/dark icon colors.
    """

    def __init__(
        self,
        icon_name: str,
        *,
        size: int = 20,
        color: QColor | str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._icon_name = icon_name
        self._size = max(1, int(size))
        self._color = color
        self._rendered_key: tuple[str, int, float, bool, str] | None = None
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._refresh_pixmap(force=True)

    @property
    def icon_name(self) -> str:
        """Return the active SVG icon stem."""
        return self._icon_name

    def set_icon_name(self, icon_name: str) -> None:
        """Switch the displayed SVG icon stem and repaint immediately."""
        if icon_name != self._icon_name:
            self._icon_name = icon_name
            self._refresh_pixmap(force=True)

    def set_icon_size(self, size: int) -> None:
        """Resize the rendered logical icon dimensions in pixels."""
        new_size = max(1, int(size))
        if new_size != self._size:
            self._size = new_size
            self._refresh_pixmap(force=True)

    def set_icon_color(self, color: QColor | str | None) -> None:
        """Override the tint color (pass ``None`` to follow the active theme palette)."""
        if color != self._color:
            self._color = color
            self._refresh_pixmap(force=True)

    def _refresh_pixmap(self, *, force: bool = False) -> None:
        dpr = max(1.0, float(self.devicePixelRatioF()))
        pal = _effective_palette(self)
        dark = is_dark_palette(pal)
        fg_key = (
            str(self._color)
            if self._color is not None
            else pal.color(QPalette.ColorGroup.Active, QPalette.ColorRole.ButtonText).name()
        )
        key = (self._icon_name, self._size, dpr, dark, fg_key)
        if not force and self._rendered_key == key:
            return
        self._rendered_key = key
        self.setPixmap(
            render_svg_pixmap(
                self._icon_name,
                size=self._size,
                device_pixel_ratio=dpr,
                color=self._color,
                dark=dark,
            )
        )

    def _refresh_themed_visuals(self) -> None:
        """Refresh the cached SVG pixmap after an application palette or theme update."""
        self._refresh_pixmap(force=True)

    def changeEvent(self, event: QEvent) -> None:
        """Refresh the SVG pixmap on DPR, palette, or style updates."""
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


class SectionCard(QFrame):
    """Elevated rounded surface card with optional title, subtitle, and accessory row."""

    def __init__(
        self,
        title: str = "",
        *,
        subtitle: str = "",
        icon_name: str | None = None,
        surface: str = "card",
        margins: tuple[int, int, int, int] = (16, 14, 16, 14),
        spacing: int = 10,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Plain)
        self.setProperty("afvSurface", surface)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(*margins)
        self._root_layout.setSpacing(spacing)

        self.header_layout: QHBoxLayout | None = None
        self.title_icon_label: ThemedSvgLabel | None = None
        self.title_label: QLabel | None = None
        self.subtitle_label: QLabel | None = None

        if title or subtitle or icon_name:
            header_row = QHBoxLayout()
            header_row.setContentsMargins(0, 0, 0, 0)
            header_row.setSpacing(8)

            if icon_name:
                self.title_icon_label = ThemedSvgLabel(icon_name, size=18, parent=self)
                header_row.addWidget(self.title_icon_label, alignment=Qt.AlignmentFlag.AlignTop)

            text_col = QVBoxLayout()
            text_col.setContentsMargins(0, 0, 0, 0)
            text_col.setSpacing(2)
            if title:
                self.title_label = make_section_label(title, parent=self)
                text_col.addWidget(self.title_label)
            if subtitle:
                self.subtitle_label = make_muted_label(subtitle, parent=self)
                text_col.addWidget(self.subtitle_label)
            header_row.addLayout(text_col, stretch=1)

            self.header_layout = header_row
            self._root_layout.addLayout(header_row)

        self.body_layout = QVBoxLayout()
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(spacing)
        self._root_layout.addLayout(self.body_layout, stretch=1)

    def add_header_accessory(self, widget: QWidget) -> None:
        """Attach a right-aligned accessory widget to the card's header row."""
        if self.header_layout is None:
            header_row = QHBoxLayout()
            header_row.setContentsMargins(0, 0, 0, 0)
            header_row.setSpacing(8)
            header_row.addStretch(1)
            self.header_layout = header_row
            self._root_layout.insertLayout(0, header_row)
        self.header_layout.addWidget(widget, alignment=Qt.AlignmentFlag.AlignVCenter)


class ActionBarFooter(QFrame):
    """Sticky bottom footer bar pinned outside scrollable page bodies."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        margins: tuple[int, int, int, int] = (20, 10, 20, 12),
        spacing: int = 10,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFrameShadow(QFrame.Shadow.Plain)
        self.setProperty("afvSurface", "footer-bar")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.content_layout = QHBoxLayout(self)
        self.content_layout.setContentsMargins(*margins)
        self.content_layout.setSpacing(spacing)


class WorkflowHeader(QFrame):
    """Compact, read-only top workflow header bar hosted inside ``MainWindow``.

    Shows the application emblem, wordmark, and a non-interactive linear step
    strip so users always know where they are in the batch lifecycle. Every
    widget inside stays ``FocusPolicy.NoFocus`` so keyboard traversal flows
    directly through the active screen.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("workflow_header")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFrameShadow(QFrame.Shadow.Plain)
        self.setProperty("afvSurface", "header-bar")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        set_accessible(
            self,
            "Workflow progress bar",
            "Displays the application name and the active workflow step.",
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 8, 18, 8)
        layout.setSpacing(10)

        self.brand_icon = ThemedSvgLabel("app_icon", size=22, parent=self)
        self.brand_icon.setObjectName("workflow_brand_icon")
        layout.addWidget(self.brand_icon, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.brand_label = QLabel("Audio from Video", self)
        self.brand_label.setObjectName("workflow_brand_label")
        self.brand_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.brand_label.setTextFormat(Qt.TextFormat.PlainText)
        self.brand_label.setFont(
            _scale_font(self.brand_label.font(), scale=1.0, weight=QFont.Weight.DemiBold)
        )
        layout.addWidget(self.brand_label, alignment=Qt.AlignmentFlag.AlignVCenter)

        layout.addStretch(1)

        # Single pill displayed during the one-time Setup step.
        self.setup_pill = QLabel("Setup", self)
        self.setup_pill.setObjectName("workflow_step_setup")
        self.setup_pill.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setup_pill.setTextFormat(Qt.TextFormat.PlainText)
        self.setup_pill.setFont(
            _scale_font(self.setup_pill.font(), scale=0.88, weight=QFont.Weight.DemiBold)
        )
        self.setup_pill.setProperty("afvStepState", "active")
        self.setup_pill.setVisible(False)
        layout.addWidget(self.setup_pill, alignment=Qt.AlignmentFlag.AlignVCenter)

        # Read-only linear 5-step batch progression indicator.
        self._steps_container = QWidget(self)
        self._steps_container.setObjectName("workflow_steps_container")
        self._steps_container.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        steps_row = QHBoxLayout(self._steps_container)
        steps_row.setContentsMargins(0, 0, 0, 0)
        steps_row.setSpacing(4)

        self.step_labels: dict[str, QLabel] = {}
        self._step_separators: list[QLabel] = []
        for idx, (key, title) in enumerate(_WORKFLOW_STEPS):
            if idx > 0:
                sep = ThemedSvgLabel("chevron_right", size=10, parent=self._steps_container)
                sep.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                self._step_separators.append(sep)
                steps_row.addWidget(sep, alignment=Qt.AlignmentFlag.AlignVCenter)

            pill = QLabel(title, self._steps_container)
            pill.setObjectName(f"workflow_step_{key}")
            pill.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            pill.setTextFormat(Qt.TextFormat.PlainText)
            pill.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            pill.setFont(_scale_font(pill.font(), scale=0.88))
            pill.setProperty("afvStepState", "pending")
            self.step_labels[key] = pill
            steps_row.addWidget(pill, alignment=Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self._steps_container, alignment=Qt.AlignmentFlag.AlignVCenter)
        self._current_step_key: str = ""

    def update_state(
        self,
        state: ControllerState,
        *,
        last_active_state: ControllerState | None = None,
    ) -> None:
        """Synchronize the header pills with the active ``ControllerState``."""
        effective = state
        if state == ControllerState.CANCELLING and last_active_state in (
            ControllerState.PLANNING,
            ControllerState.RUNNING,
        ):
            effective = last_active_state

        if effective == ControllerState.PREFLIGHT:
            self.setVisible(False)
            self._current_step_key = "preflight"
            return

        self.setVisible(True)
        if effective == ControllerState.SETUP:
            self.setup_pill.setVisible(True)
            self._steps_container.setVisible(False)
            self.setup_pill.setText("Setup")
            set_widget_role(self.setup_pill, "afvStepState", "active")
            self._current_step_key = "setup"
            return

        self.setup_pill.setVisible(False)
        self._steps_container.setVisible(True)

        state_to_idx: dict[ControllerState, int] = {
            ControllerState.INPUTS: 0,
            ControllerState.PLANNING: 1,
            ControllerState.REVIEW: 2,
            ControllerState.RUNNING: 3,
            ControllerState.SUMMARY: 4,
        }
        active_idx = state_to_idx.get(effective, 0)
        self._current_step_key = _WORKFLOW_STEPS[active_idx][0]

        for idx, (key, title) in enumerate(_WORKFLOW_STEPS):
            pill = self.step_labels[key]
            if idx < active_idx:
                pill.setFont(_scale_font(self.font(), scale=0.88, weight=QFont.Weight.Medium))
                pill.setText(f"✓ {title}")
                set_widget_role(pill, "afvStepState", "done")
            elif idx == active_idx:
                pill.setFont(_scale_font(self.font(), scale=0.88, weight=QFont.Weight.DemiBold))
                suffix = "…" if state == ControllerState.CANCELLING else ""
                pill.setText(f"{title}{suffix}")
                set_widget_role(pill, "afvStepState", "active")
            else:
                pill.setFont(_scale_font(self.font(), scale=0.88, weight=QFont.Weight.Normal))
                pill.setText(title)
                set_widget_role(pill, "afvStepState", "pending")
            pill.setAccessibleDescription(f"Step {idx + 1} of {len(_WORKFLOW_STEPS)}: {title}.")
            pill.adjustSize()
            pill.updateGeometry()
        self._steps_container.updateGeometry()
        steps_layout = self._steps_container.layout()
        if steps_layout is not None:
            steps_layout.invalidate()
            steps_layout.activate()
        self.updateGeometry()
        root_layout = self.layout()
        if root_layout is not None:
            root_layout.invalidate()
            root_layout.activate()


class IconStatusBanner(QFrame):
    """Semantic, palette-safe banner pairing an SVG status icon with accessible text.

    Applies dynamic properties ``afvBanner=True`` and ``afvTone`` derived from
    ``icon_name`` (``ok``, ``info``, ``working``, ``warn``, ``error``,
    ``unmatched``, or ``neutral``) so centralized theming renders crisp tinted
    surfaces with a left accent stripe in light, dark, and high-contrast modes.
    """

    def __init__(self, parent: QWidget | None = None, *, icon_size: int = 18) -> None:
        super().__init__(parent)
        self._icon_size = icon_size
        self._icon_name = "status_info"
        self._rendered_icon_key: tuple[str, float, bool] | None = None
        self._last_text = ""
        self._last_role_label = ""
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Plain)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.setProperty("afvBanner", True)
        self.setProperty("afvTone", tone_for_icon(self._icon_name))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 7, 12, 7)
        layout.setSpacing(9)

        self._accent_stripe = QFrame(self)
        self._accent_stripe.setObjectName("banner_accent_stripe")
        self._accent_stripe.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._accent_stripe.setFrameShape(QFrame.Shape.NoFrame)
        self._accent_stripe.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._accent_stripe.setFixedWidth(3)
        self._accent_stripe.setMinimumHeight(14)
        self._accent_stripe.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._accent_stripe.setProperty("afvBannerStripe", True)

        self.icon_label = QLabel(self)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.icon_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.icon_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)

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

        layout.addWidget(self._accent_stripe)
        layout.addWidget(self.icon_label, alignment=Qt.AlignmentFlag.AlignTop)
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
        dark = is_dark_palette(_effective_palette(self))
        key = (icon_name, dpr, dark)
        if not force and self._rendered_icon_key == key:
            return
        self._rendered_icon_key = key
        self.icon_label.setPixmap(
            render_svg_pixmap(icon_name, size=self._icon_size, device_pixel_ratio=dpr, dark=dark)
        )

    def _refresh_themed_visuals(self) -> None:
        """Refresh the banner status icon after a palette or theme update."""
        if self._rendered_icon_key is not None:
            self._update_pixmap(self._icon_name, force=True)

    def changeEvent(self, event: QEvent) -> None:
        """Refresh the banner icon when the display scale, palette, or style changes."""
        super().changeEvent(event)
        etype = event.type()
        if self._rendered_icon_key is not None and (
            etype == QEvent.Type.DevicePixelRatioChange
            or (
                not is_theme_updating()
                and etype
                in (
                    QEvent.Type.PaletteChange,
                    QEvent.Type.ApplicationPaletteChange,
                    QEvent.Type.StyleChange,
                )
            )
        ):
            self._update_pixmap(self._icon_name, force=True)

    def set_message(self, icon_name: str, text: str, *, role_label: str = "Notice") -> None:
        """Show the banner with ``icon_name`` and ``text``."""
        self._icon_name = icon_name
        new_tone = tone_for_icon(icon_name)
        if self.property("afvTone") != new_tone:
            self.setProperty("afvTone", new_tone)
            repolish(self, include_children=True)
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
        selectable: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._icon_size = icon_size
        self._icon_name = icon_name
        self._rendered_icon_key: tuple[str, float, bool] | None = None
        self._last_text: str | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.icon_label = QLabel(self)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.icon_label.setContentsMargins(0, 1, 0, 0)
        self.icon_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.icon_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.text_label = QLabel(self)
        self.text_label.setWordWrap(True)
        self.text_label.setTextFormat(Qt.TextFormat.PlainText)
        self.text_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        if selectable:
            self.text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        layout.addWidget(self.icon_label, alignment=Qt.AlignmentFlag.AlignTop)
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
        dark = is_dark_palette(_effective_palette(self))
        key = (icon_name, dpr, dark)
        if not force and self._rendered_icon_key == key:
            return
        self._rendered_icon_key = key
        self.icon_label.setPixmap(
            render_svg_pixmap(icon_name, size=self._icon_size, device_pixel_ratio=dpr, dark=dark)
        )

    def _refresh_themed_visuals(self) -> None:
        """Refresh the banner status icon after a palette or theme update."""
        if self._rendered_icon_key is not None:
            self._update_pixmap(self._icon_name, force=True)

    def changeEvent(self, event: QEvent) -> None:
        """Refresh the status icon when the display scale, palette, or style changes."""
        super().changeEvent(event)
        etype = event.type()
        if self._rendered_icon_key is not None and (
            etype == QEvent.Type.DevicePixelRatioChange
            or (
                not is_theme_updating()
                and etype
                in (
                    QEvent.Type.PaletteChange,
                    QEvent.Type.ApplicationPaletteChange,
                    QEvent.Type.StyleChange,
                )
            )
        ):
            self._update_pixmap(self._icon_name, force=True)

    def set_status(self, icon_name: str, text: str) -> None:
        """Update both the SVG status badge and label text together."""
        self._icon_name = icon_name
        self.setProperty("afvTone", tone_for_icon(icon_name))
        self._update_pixmap(icon_name)
        if text != self._last_text:
            self._last_text = text
            self.text_label.setText(text)
            self.setAccessibleName(text)
            self.setAccessibleDescription(text)
            self.text_label.setAccessibleName(text)
            self.text_label.setAccessibleDescription(text)


__all__ = [
    "QUALITY_PRESETS",
    "ActionBarFooter",
    "IconStatusBanner",
    "MiddleElideLabel",
    "SectionCard",
    "StatusLine",
    "ThemedSvgLabel",
    "WorkflowHeader",
    "apply_danger_button_role",
    "apply_filter_pill_role",
    "apply_ghost_button_role",
    "apply_primary_button_role",
    "apply_secondary_button_role",
    "configure_table_view",
    "find_invalid_env_overrides",
    "format_byte_size",
    "format_ttl_duration",
    "join_oxford",
    "log_directory",
    "make_bold_font",
    "make_divider",
    "make_heading_label",
    "make_mono_font",
    "make_muted_label",
    "make_section_label",
    "open_local_path",
    "open_local_path_with_fallback",
    "open_ui_settings",
    "read_bool_setting",
    "resolve_existing_directory",
    "set_accessible",
    "user_log_dir",
]
