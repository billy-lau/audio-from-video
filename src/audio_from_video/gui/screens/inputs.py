"""Inputs screen for choosing media files, folders, output directory, and run options."""

from __future__ import annotations

import itertools
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QSize, Qt, Slot
from PySide6.QtGui import (
    QDragEnterEvent,
    QDragLeaveEvent,
    QDragMoveEvent,
    QDropEvent,
    QIcon,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from audio_from_video.config import (
    AppConfig,
    ConfigError,
    default_config_path,
    default_jobs,
    load_config,
    parse_quality,
)
from audio_from_video.discovery import DEFAULT_OUTPUT_FOLDER_NAME, VIDEO_EXTENSIONS
from audio_from_video.gui.controller import ControllerState, build_fallback_config
from audio_from_video.gui.resources import load_icon
from audio_from_video.gui.screens._common import (
    QUALITY_PRESETS,
    ActionBarFooter,
    IconStatusBanner,
    MiddleElideLabel,
    SectionCard,
    StatusLine,
    ThemedSvgLabel,
    apply_primary_button_role,
    apply_secondary_button_role,
    find_invalid_env_overrides,
    join_oxford,
    make_divider,
    make_heading_label,
    make_muted_label,
    open_ui_settings,
    set_accessible,
)
from audio_from_video.gui.theme import (
    is_theme_updating,
    refresh_themed_widgets,
    set_widget_role,
)
from audio_from_video.models import ConflictPolicy, ManualOverrides
from audio_from_video.session import RunOptions

if TYPE_CHECKING:
    from audio_from_video.gui.controller import RunController

_LAST_INPUTS_FOLDER_KEY = "inputs/last_folder"
_LAST_OUTPUT_FOLDER_KEY = "output/last_folder"
INVALID_CONFIG_BANNER_TEXT = "config.toml is invalid; using built-in defaults."
INVALID_ENV_CONFIG_BANNER_TEXT = (
    "Configuration is invalid (AFV_* environment variable); using built-in defaults."
)
INVALID_ENV_WITH_FILE_BANNER_TEXT = (
    "Configuration is invalid (AFV_* environment variable); using saved file settings."
)
INVALID_UNKNOWN_CONFIG_BANNER_TEXT = (
    "Configuration is invalid (config.toml or AFV_* environment); using built-in defaults."
)
OFFLINE_SWITCH_NOTICE_TEXT = "Switched to Offline mode because start-offline is enabled."
_UNSET = object()

_DefaultsKey = tuple[str, bool, str | None, bool, ConflictPolicy, int]


@dataclass(frozen=True, slots=True)
class _FallbackConfigDiagnosis:
    """Single-pass fallback ``AppConfig`` and warning banner when ``controller.config`` is None."""

    effective_config: AppConfig
    from_file: bool
    banner_text: str | None


def _open_shortcut_text() -> str:
    """Return a crisp, cross-platform label for the Open shortcut."""
    return "Cmd+O" if sys.platform == "darwin" else "Ctrl+O"


def _classify_invalid_config_banner(
    *,
    target_path: Path | None,
    from_file: bool,
    file_error: str | None,
    show_when_none: bool,
    active_env: Mapping[str, str],
    preflight_error: str = "",
) -> str | None:
    """Build an accurate warning banner for why ``controller.config`` is ``None``."""
    env_issues = list(find_invalid_env_overrides(active_env).values())
    env_load_error: str | None = None
    if file_error is None and not env_issues and target_path is not None:
        try:
            load_config(config_path=target_path, env=active_env)
        except ConfigError as exc:
            env_load_error = str(exc)
        except OSError as exc:
            file_error = str(exc)

    env_prefix = INVALID_ENV_WITH_FILE_BANNER_TEXT if from_file else INVALID_ENV_CONFIG_BANNER_TEXT
    if file_error is not None:
        if env_issues:
            sources = join_oxford(["config.toml", *(var for var, _detail in env_issues)])
            return f"Configuration is invalid ({sources}); using built-in defaults."
        return INVALID_CONFIG_BANNER_TEXT
    if env_issues:
        names = join_oxford([var for var, _detail in env_issues])
        detail = env_issues[0][1].strip()
        if detail and "config.toml" not in detail.casefold():
            return f"{env_prefix} Check {names}: {detail}"
        return f"{env_prefix} Check {names}."
    if env_load_error is not None:
        cleaned = env_load_error.strip()
        if cleaned and "config.toml" not in cleaned.casefold():
            return f"{env_prefix} {cleaned}"
        return env_prefix
    if not show_when_none:
        return None
    cleaned_preflight = preflight_error.strip()
    if (
        cleaned_preflight
        and "AFV_" in cleaned_preflight
        and "config.toml" not in cleaned_preflight.casefold()
    ):
        return f"{env_prefix} {cleaned_preflight}"
    if from_file:
        return "Configuration warning (AFV_* environment or settings); using saved file settings."
    return INVALID_UNKNOWN_CONFIG_BANNER_TEXT


def _diagnose_config_fallback(
    *,
    show_when_none: bool,
    env: Mapping[str, str] | None = None,
    preflight_error: str = "",
) -> _FallbackConfigDiagnosis:
    """Delegate fallback loading to :func:`build_fallback_config` and classify the banner."""
    active_env = os.environ if env is None else env
    path_error: str | None = None
    try:
        target_path: Path | None = default_config_path(active_env)
    except OSError as exc:
        target_path = None
        path_error = str(exc)

    effective, from_file, load_error = build_fallback_config(
        target_path,
        env=active_env,
        load_config_fn=load_config,
    )
    file_error = path_error if path_error is not None else load_error
    banner_text = _classify_invalid_config_banner(
        target_path=target_path,
        from_file=from_file and path_error is None,
        file_error=file_error,
        show_when_none=show_when_none,
        active_env=active_env,
        preflight_error=preflight_error,
    )
    return _FallbackConfigDiagnosis(
        effective_config=effective,
        from_file=from_file and path_error is None,
        banner_text=banner_text,
    )


def _invalid_config_banner_text(
    *,
    show_when_none: bool,
    env: Mapping[str, str] | None = None,
    preflight_error: str = "",
) -> str | None:
    """Classify why ``controller.config`` is ``None`` without falsely blaming ``config.toml``."""
    return _diagnose_config_fallback(
        show_when_none=show_when_none,
        env=env,
        preflight_error=preflight_error,
    ).banner_text


def _config_defaults_key(config: AppConfig) -> _DefaultsKey:
    """Return a snapshot of the ``AppConfig`` fields driving ``InputsScreen`` defaults."""
    return (
        config.audio.quality,
        config.audio.copy_mp3,
        config.output.output_dir,
        config.output.flat,
        config.output.on_conflict,
        config.execution.jobs,
    )


def _extract_dropped_paths(event: QDropEvent) -> list[Path]:
    """Extract local filesystem paths from a Qt drop event."""
    mime = event.mimeData()
    if mime is None or not mime.hasUrls():
        return []
    paths: list[Path] = []
    for url in mime.urls():
        if url.isLocalFile():
            local = url.toLocalFile()
            if local:
                paths.append(Path(local))
    return paths


def _has_local_urls(event: QDragEnterEvent | QDragMoveEvent) -> bool:
    """Return True when the drag event carries at least one local file or folder URL."""
    mime = event.mimeData()
    if mime is None or not mime.hasUrls():
        return False
    return any(url.isLocalFile() and bool(url.toLocalFile()) for url in mime.urls())


def _is_directory_like(path: Path) -> bool:
    """Return True if ``path`` exists as a directory, or looks like one synthetically."""
    if path.exists():
        return path.is_dir()
    return not bool(path.suffix)


def _compute_default_output_preview(
    paths: Sequence[Path],
    config: AppConfig | None,
    *,
    is_dir_flags: Sequence[bool] | None = None,
) -> tuple[str, bool]:
    """Return ``(preview_text, has_multiple_roots)`` without scanning folder contents."""
    if config is not None and config.output.output_dir:
        return (f"Default (from config): {config.output.output_dir}", False)
    if not paths:
        return (f"Default: <input folder>/{DEFAULT_OUTPUT_FOLDER_NAME}", False)

    anchors: set[Path] = set()
    for idx, path in enumerate(paths):
        expanded = path.expanduser()
        is_dir = (
            is_dir_flags[idx]
            if is_dir_flags is not None and idx < len(is_dir_flags)
            else _is_directory_like(expanded)
        )
        if is_dir:
            anchors.add(expanded)
        else:
            anchors.add(expanded.parent)

    if len(anchors) == 1:
        anchor = next(iter(anchors))
        return (f"Default: {anchor / DEFAULT_OUTPUT_FOLDER_NAME}", False)
    return (
        "Inputs span multiple folders — choose an explicit output folder.",
        True,
    )


class _DropTargetFrame(QFrame):
    """Frame accepting file and folder drag-and-drop with hover highlight feedback."""

    def __init__(
        self,
        on_paths_dropped: Callable[[list[Path]], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_paths_dropped = on_paths_dropped
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAcceptDrops(True)
        self.setProperty("afvSurface", "drop-zone")
        self.setProperty("afvDragActive", False)

    def set_drag_active(self, active: bool) -> None:
        """Toggle the dashed drop-zone hover ring via centralized dynamic properties."""
        set_widget_role(self, "afvDragActive", bool(active))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if _has_local_urls(event):
            self.set_drag_active(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if _has_local_urls(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        self.set_drag_active(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        self.set_drag_active(False)
        paths = _extract_dropped_paths(event)
        if paths:
            self._on_paths_dropped(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class _DropListWidget(QListWidget):
    """Input list widget that forwards dropped local URLs directly to ``InputsScreen``."""

    def __init__(
        self,
        on_paths_dropped: Callable[[list[Path]], None],
        on_drag_active: Callable[[bool], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_paths_dropped = on_paths_dropped
        self._on_drag_active = on_drag_active
        self.setAcceptDrops(True)

    def sizeHint(self) -> QSize:
        """Keep a compact default list height inside scroll areas instead of Qt's 192 px box."""
        base = super().sizeHint()
        visible_rows = max(2, min(4, self.count()))
        target_height = 20 + visible_rows * 27 if self.count() > 0 else 76
        return QSize(base.width(), max(76, min(140, target_height)))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if _has_local_urls(event):
            if self._on_drag_active is not None:
                self._on_drag_active(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if _has_local_urls(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        if self._on_drag_active is not None:
            self._on_drag_active(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        if self._on_drag_active is not None:
            self._on_drag_active(False)
        paths = _extract_dropped_paths(event)
        if paths:
            self._on_paths_dropped(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class InputsScreen(QWidget):
    """Inputs screen listening to ``input_error``, ``setup_error``, and ``settings_saved``."""

    def __init__(
        self,
        controller: RunController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._paths: list[Path] = []
        self._path_is_dir: list[bool] = []
        self._default_quality: str = "320k"
        self._output_highlighted = False
        self._last_state: object = controller.state
        self._defaults_initialized: bool = controller.state not in (
            ControllerState.PREFLIGHT,
            ControllerState.SETUP,
        )
        self._had_valid_config: bool = False
        self._effective_fallback_config: AppConfig | None = None
        self._loaded_defaults_key: _DefaultsKey | None = None
        self._last_offline_default: bool | None = None
        self._preflight_setup_error: str = ""
        self._cleared_single_track_notice = False
        self._folder_icon: QIcon = load_icon("folder")
        self._file_icon: QIcon = load_icon("file_add")

        self.setAcceptDrops(True)
        set_accessible(
            self,
            "Inputs screen",
            "Select video files or folders, choose an output folder, and set per-run options.",
        )

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setObjectName("inputs_scroll_area")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        set_accessible(scroll, "Inputs content", "Scrollable container for input controls.")

        body = QWidget(scroll)
        body.setObjectName("inputs_scroll_body")
        body_row = QHBoxLayout(body)
        body_row.setContentsMargins(24, 10, 24, 12)
        body_row.setSpacing(0)

        column = QWidget(body)
        column.setObjectName("inputs_content_column")
        column.setMaximumWidth(960)
        column.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        body_row.addStretch(0)
        body_row.addWidget(column, stretch=1)
        body_row.addStretch(0)

        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        shortcut_label = _open_shortcut_text()

        # Header row with Settings button.
        header_row = QHBoxLayout()
        header_row.setSpacing(12)
        title_box = QVBoxLayout()
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(2)
        self.title_label = make_heading_label("Select Videos to Extract", parent=column)
        self.title_label.setObjectName("inputs_title_label")
        subtitle = make_muted_label(
            "Options changed here apply only to this run. Edit saved defaults in Settings.",
            parent=column,
        )
        subtitle.setObjectName("inputs_subtitle_label")
        title_box.addWidget(self.title_label)
        title_box.addWidget(subtitle)
        header_row.addLayout(title_box, stretch=1)

        self.settings_button = QPushButton("Se&ttings…", column)
        self.settings_button.setObjectName("inputs_settings_button")
        apply_secondary_button_role(self.settings_button, icon_name="settings")
        self.settings_button.setToolTip(
            "Open Settings to edit saved defaults, inspect cache statistics, or view logs."
        )
        set_accessible(
            self.settings_button,
            "Settings",
            "Open the Settings dialog for defaults, overrides, and cache maintenance.",
        )
        header_row.addWidget(self.settings_button, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header_row)

        # Persistent warning banner shown when configuration failed to load.
        self.config_warning_banner = IconStatusBanner(column)
        self.config_warning_banner.setObjectName("inputs_config_warning_banner")
        layout.addWidget(self.config_warning_banner)

        # Inline error banner (hidden until input_error or local validation failure).
        self.error_banner = IconStatusBanner(column)
        self.error_banner.setObjectName("inputs_error_banner")
        layout.addWidget(self.error_banner)

        # Brief neutral notice shown when a Setup detour switches Online to Offline.
        self.mode_notice_banner = IconStatusBanner(column)
        self.mode_notice_banner.setObjectName("inputs_mode_notice_banner")
        layout.addWidget(self.mode_notice_banner)

        # Input files and drop zone section (elevated SectionCard, single dashed drop zone).
        inputs_card = SectionCard(
            "Input Files and Folders",
            margins=(16, 9, 16, 9),
            spacing=6,
            parent=column,
        )
        inputs_card.setObjectName("inputs_files_card")

        self.recursive_checkbox = QCheckBox("Include su&bfolders", inputs_card)
        self.recursive_checkbox.setObjectName("inputs_recursive_checkbox")
        self.recursive_checkbox.setChecked(False)
        self.recursive_checkbox.setToolTip(
            "Scan inside nested subfolders when adding directories to the batch."
        )
        set_accessible(
            self.recursive_checkbox,
            "Include subfolders",
            "Scan added folders recursively for supported video files.",
        )
        inputs_card.add_header_accessory(self.recursive_checkbox)

        self.drop_zone = _DropTargetFrame(self.add_paths, inputs_card)
        self.drop_zone.setObjectName("inputs_drop_zone")
        self.drop_zone.setFrameShape(QFrame.Shape.NoFrame)
        set_accessible(
            self.drop_zone,
            "Input drop zone",
            "Drop video files or folders here to add them to the extraction queue.",
        )
        drop_layout = QVBoxLayout(self.drop_zone)
        drop_layout.setContentsMargins(12, 10, 12, 10)
        drop_layout.setSpacing(8)

        hint_row = QHBoxLayout()
        hint_row.setContentsMargins(0, 0, 0, 0)
        hint_row.setSpacing(8)
        self._drop_icon = ThemedSvgLabel("drop_zone", size=20, parent=self.drop_zone)
        self._drop_icon.setObjectName("inputs_drop_icon")
        self.drop_hint_label = make_muted_label(
            f"Drop video files or folders here, or use Add files ({shortcut_label}) or Add folder.",
            scale=0.95,
            parent=self.drop_zone,
        )
        self.drop_hint_label.setObjectName("inputs_drop_hint_label")
        hint_row.addWidget(self._drop_icon, alignment=Qt.AlignmentFlag.AlignVCenter)
        hint_row.addWidget(self.drop_hint_label, stretch=1)
        drop_layout.addLayout(hint_row)

        self.inputs_list = _DropListWidget(
            self.add_paths,
            on_drag_active=self.drop_zone.set_drag_active,
            parent=self.drop_zone,
        )
        self.inputs_list.setObjectName("inputs_list")
        self.inputs_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.inputs_list.setMinimumHeight(76)
        self.inputs_list.setMaximumHeight(140)
        self.inputs_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.inputs_list.setAlternatingRowColors(True)
        self.inputs_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.inputs_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.inputs_list.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.inputs_list.setToolTip(
            "Queued video files and folders. Press Delete or Backspace to remove selected items."
        )
        set_accessible(
            self.inputs_list,
            "Queued input files and folders",
            "List of selected video files and folders to scan and plan.",
        )
        drop_layout.addWidget(self.inputs_list)
        inputs_card.body_layout.addWidget(self.drop_zone)

        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.setSpacing(8)
        self.add_files_button = QPushButton("Add &files…", inputs_card)
        self.add_files_button.setObjectName("inputs_add_files_button")
        self.add_files_button.setIcon(self._file_icon)
        apply_secondary_button_role(self.add_files_button)
        self.add_files_button.setToolTip(f"Add one or more video files ({shortcut_label}).")
        set_accessible(
            self.add_files_button,
            "Add files",
            f"Choose video files to add to the batch queue (shortcut {shortcut_label}).",
        )

        self.add_folder_button = QPushButton("Add f&older…", inputs_card)
        self.add_folder_button.setObjectName("inputs_add_folder_button")
        self.add_folder_button.setIcon(self._folder_icon)
        apply_secondary_button_role(self.add_folder_button)
        self.add_folder_button.setToolTip("Add a folder of video files to scan.")
        set_accessible(
            self.add_folder_button,
            "Add folder",
            "Choose a folder of video files to scan.",
        )

        self.remove_button = QPushButton("&Remove", inputs_card)
        self.remove_button.setObjectName("inputs_remove_button")
        apply_secondary_button_role(self.remove_button, icon_name="trash")
        self.remove_button.setEnabled(False)
        self.remove_button.setToolTip(
            "Remove the selected files or folders from the queue (Delete or Backspace)."
        )
        set_accessible(
            self.remove_button,
            "Remove selected inputs",
            "Remove the highlighted files or folders from the input queue.",
        )

        self.clear_button = QPushButton("C&lear", inputs_card)
        self.clear_button.setObjectName("inputs_clear_button")
        apply_secondary_button_role(self.clear_button)
        self.clear_button.setEnabled(False)
        self.clear_button.setToolTip("Remove every file and folder from the input queue.")
        set_accessible(
            self.clear_button,
            "Clear all inputs",
            "Remove every file and folder from the input queue.",
        )

        toolbar_row.addWidget(self.add_files_button)
        toolbar_row.addWidget(self.add_folder_button)
        toolbar_row.addStretch(1)
        toolbar_row.addWidget(self.remove_button)
        toolbar_row.addWidget(self.clear_button)
        inputs_card.body_layout.addLayout(toolbar_row)
        layout.addWidget(inputs_card)

        # Output folder picker card.
        output_card = SectionCard(
            "Output Folder",
            margins=(16, 10, 16, 10),
            spacing=5,
            parent=column,
        )
        output_card.setObjectName("inputs_output_card")

        self.output_frame = QFrame(output_card)
        self.output_frame.setObjectName("inputs_output_frame")
        self.output_frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.output_frame.setFrameShape(QFrame.Shape.NoFrame)
        self.output_frame.setProperty("afvHighlight", "none")
        frame_layout = QHBoxLayout(self.output_frame)
        frame_layout.setContentsMargins(6, 5, 6, 5)
        frame_layout.setSpacing(8)

        output_label = QLabel("Out&put folder:", self.output_frame)
        self.output_dir_edit = QLineEdit(self.output_frame)
        self.output_dir_edit.setObjectName("inputs_output_dir_edit")
        self.output_dir_edit.setPlaceholderText("Leave blank to use default folder")
        self.output_dir_edit.setClearButtonEnabled(True)
        self.output_dir_edit.setToolTip(
            "Optional destination directory for extracted MP3 files. "
            "Leave blank to use the computed default shown below."
        )
        output_label.setBuddy(self.output_dir_edit)
        set_accessible(
            self.output_dir_edit,
            "Output folder",
            "Optional destination directory for extracted MP3 files. Leave blank for default.",
        )

        self.output_browse_button = QPushButton("Bro&wse…", self.output_frame)
        self.output_browse_button.setObjectName("inputs_output_browse_button")
        self.output_browse_button.setIcon(self._folder_icon)
        apply_secondary_button_role(self.output_browse_button)
        self.output_browse_button.setToolTip("Choose a destination folder for extracted MP3 files.")
        set_accessible(
            self.output_browse_button,
            "Browse output folder",
            "Choose a destination folder for extracted MP3 files.",
        )

        self.output_reset_button = QPushButton("Use &default", self.output_frame)
        self.output_reset_button.setObjectName("inputs_output_reset_button")
        apply_secondary_button_role(self.output_reset_button)
        self.output_reset_button.setEnabled(False)
        self.output_reset_button.setToolTip(
            "Clear the custom folder and return to the computed default destination."
        )
        set_accessible(
            self.output_reset_button,
            "Use default output folder",
            "Clear the explicit output folder and return to the computed default.",
        )

        frame_layout.addWidget(output_label)
        frame_layout.addWidget(self.output_dir_edit, stretch=1)
        frame_layout.addWidget(self.output_browse_button)
        frame_layout.addWidget(self.output_reset_button)
        output_card.body_layout.addWidget(self.output_frame)

        self.output_default_status = StatusLine(
            "status_info",
            f"Default: <input folder>/{DEFAULT_OUTPUT_FOLDER_NAME}",
            selectable=True,
            parent=output_card,
        )
        self.output_default_status.setObjectName("inputs_output_default_status")
        output_card.body_layout.addWidget(self.output_default_status)
        layout.addWidget(output_card)

        # Run options card (per-run only, never written back to QSettings or config.toml).
        options_card = SectionCard(
            "Run Options",
            subtitle="Applies only to this run; saved defaults live in Settings",
            margins=(16, 11, 16, 11),
            spacing=8,
            parent=column,
        )
        options_card.setObjectName("inputs_options_card")

        # Subsection 1: Metadata lookup and matching behavior.
        lookup_row = QHBoxLayout()
        lookup_row.setContentsMargins(0, 0, 0, 0)
        lookup_row.setSpacing(14)
        mode_label = QLabel("Lookup mode:", options_card)
        self.mode_group = QButtonGroup(self)
        self.online_radio = QRadioButton("O&nline", options_card)
        self.online_radio.setObjectName("inputs_online_radio")
        self.online_radio.setToolTip(
            "Query MusicBrainz and iTunes for track metadata and embed cover artwork."
        )
        self.offline_radio = QRadioButton("Offl&ine", options_card)
        self.offline_radio.setObjectName("inputs_offline_radio")
        self.offline_radio.setToolTip(
            "Skip all online requests and rely only on container tags and filenames."
        )
        self.mode_group.addButton(self.online_radio)
        self.mode_group.addButton(self.offline_radio)
        set_accessible(
            self.online_radio,
            "Online lookup mode",
            "Query MusicBrainz and iTunes for metadata and cover art during this run.",
        )
        set_accessible(
            self.offline_radio,
            "Offline lookup mode",
            "Skip all online lookups and use embedded container tags and filenames only.",
        )
        lookup_row.addWidget(mode_label)
        lookup_row.addWidget(self.online_radio)
        lookup_row.addWidget(self.offline_radio)
        lookup_row.addStretch(1)
        options_card.body_layout.addLayout(lookup_row)

        matching_checks = QGridLayout()
        matching_checks.setContentsMargins(0, 0, 0, 0)
        matching_checks.setHorizontalSpacing(18)
        matching_checks.setVerticalSpacing(6)
        self.always_prompt_checkbox = QCheckBox("Confirm every &match", options_card)
        self.always_prompt_checkbox.setObjectName("inputs_always_prompt_checkbox")
        self.always_prompt_checkbox.setChecked(False)
        self.always_prompt_checkbox.setToolTip(
            "Ask for confirmation before accepting each online match instead of only "
            "uncertain matches."
        )
        set_accessible(
            self.always_prompt_checkbox,
            "Confirm every match",
            "Always show the match dialog even for high-confidence online matches.",
        )

        self.keep_native_checkbox = QCheckBox("Keep nati&ve names", options_card)
        self.keep_native_checkbox.setObjectName("inputs_keep_native_checkbox")
        self.keep_native_checkbox.setChecked(False)
        self.keep_native_checkbox.setToolTip(
            "Skip the English alias prompt and retain native non-Latin artist and album names."
        )
        set_accessible(
            self.keep_native_checkbox,
            "Keep native names",
            "Keep native non-Latin artist and album names without prompting for an English alias.",
        )
        matching_checks.addWidget(self.always_prompt_checkbox, 0, 0)
        matching_checks.addWidget(self.keep_native_checkbox, 0, 1)
        matching_checks.setColumnStretch(0, 1)
        matching_checks.setColumnStretch(1, 1)
        options_card.body_layout.addLayout(matching_checks)

        options_card.body_layout.addWidget(make_divider(options_card))

        # Subsection 2: Compact top-labeled audio, output, and stream fields (fits <= 540 px).
        params_row1 = QGridLayout()
        params_row1.setContentsMargins(0, 0, 0, 0)
        params_row1.setHorizontalSpacing(14)
        params_row1.setVerticalSpacing(4)

        quality_label = make_muted_label(
            "&Quality",
            scale=0.9,
            word_wrap=False,
            parent=options_card,
        )
        self.quality_combo = QComboBox(options_card)
        self.quality_combo.setObjectName("inputs_quality_combo")
        self.quality_combo.setEditable(True)
        self.quality_combo.addItems(QUALITY_PRESETS)
        self.quality_combo.setToolTip(
            "MP3 bitrate such as 320k or 192k, or LAME variable bitrate preset (v0 through v9)."
        )
        quality_label.setBuddy(self.quality_combo)
        set_accessible(
            self.quality_combo,
            "Audio quality",
            "MP3 bitrate such as 320k, 192k, or LAME VBR preset v0 through v9.",
        )

        conflict_label = make_muted_label(
            "On &conflict",
            scale=0.9,
            word_wrap=False,
            parent=options_card,
        )
        self.conflict_combo = QComboBox(options_card)
        self.conflict_combo.setObjectName("inputs_conflict_combo")
        self.conflict_combo.addItem("Skip existing", ConflictPolicy.SKIP.value)
        self.conflict_combo.addItem("Overwrite", ConflictPolicy.OVERWRITE.value)
        self.conflict_combo.addItem("Rename", ConflictPolicy.RENAME.value)
        self.conflict_combo.setToolTip(
            "Choose how to handle destination MP3 files that already exist: skip, overwrite, "
            "or rename."
        )
        conflict_label.setBuddy(self.conflict_combo)
        set_accessible(
            self.conflict_combo,
            "On conflict policy",
            "Action when a destination file already exists: skip, overwrite, or rename.",
        )

        jobs_label = make_muted_label(
            "&Jobs",
            scale=0.9,
            word_wrap=False,
            parent=options_card,
        )
        self.jobs_spin = QSpinBox(options_card)
        self.jobs_spin.setObjectName("inputs_jobs_spin")
        self.jobs_spin.setRange(1, 256)
        self.jobs_spin.setToolTip("Maximum number of FFmpeg audio extractions to run in parallel.")
        jobs_label.setBuddy(self.jobs_spin)
        set_accessible(
            self.jobs_spin,
            "Parallel jobs",
            "Number of parallel FFmpeg extraction workers.",
        )

        params_row1.addWidget(quality_label, 0, 0)
        params_row1.addWidget(self.quality_combo, 1, 0)
        params_row1.addWidget(conflict_label, 0, 1)
        params_row1.addWidget(self.conflict_combo, 1, 1)
        params_row1.addWidget(jobs_label, 0, 2)
        params_row1.addWidget(self.jobs_spin, 1, 2)
        params_row1.setColumnStretch(0, 2)
        params_row1.setColumnStretch(1, 3)
        params_row1.setColumnStretch(2, 1)
        options_card.body_layout.addLayout(params_row1)

        params_row2 = QGridLayout()
        params_row2.setContentsMargins(0, 0, 0, 0)
        params_row2.setHorizontalSpacing(14)
        params_row2.setVerticalSpacing(4)

        lang_label = make_muted_label(
            "Preferred audio language",
            scale=0.9,
            word_wrap=False,
            parent=options_card,
        )
        self.lang_edit = QLineEdit(options_card)
        self.lang_edit.setObjectName("inputs_lang_edit")
        self.lang_edit.setPlaceholderText("e.g., eng or jpn")
        self.lang_edit.setClearButtonEnabled(True)
        lang_label.setBuddy(self.lang_edit)
        self.lang_edit.setToolTip(
            "Optional ISO 639-2 language code to prefer when a video has multiple audio streams."
        )
        set_accessible(
            self.lang_edit,
            "Preferred audio language",
            "Optional preferred audio track language code, such as eng or jpn.",
        )

        stream_label = make_muted_label(
            "Audio str&eam index",
            scale=0.9,
            word_wrap=False,
            parent=options_card,
        )
        self.stream_spin = QSpinBox(options_card)
        self.stream_spin.setObjectName("inputs_stream_spin")
        self.stream_spin.setRange(-1, 63)
        self.stream_spin.setValue(-1)
        self.stream_spin.setSpecialValueText("Auto")
        stream_label.setBuddy(self.stream_spin)
        self.stream_spin.setToolTip(
            "Explicit zero-based audio stream index, or Auto to pick the best audio stream."
        )
        set_accessible(
            self.stream_spin,
            "Audio stream index",
            "Explicit zero-based audio stream index, or Auto to pick automatically.",
        )

        params_row2.addWidget(lang_label, 0, 0)
        params_row2.addWidget(self.lang_edit, 1, 0)
        params_row2.addWidget(stream_label, 0, 1)
        params_row2.addWidget(self.stream_spin, 1, 1)
        params_row2.setColumnStretch(0, 2)
        params_row2.setColumnStretch(1, 1)
        options_card.body_layout.addLayout(params_row2)

        options_card.body_layout.addWidget(make_divider(options_card))

        # Subsection 3: Balanced 2-row grid of boolean run flags (no 640 px overflow).
        self.reencode_checkbox = QCheckBox("Re-encode MP&3", options_card)
        self.reencode_checkbox.setObjectName("inputs_reencode_checkbox")
        self.reencode_checkbox.setToolTip(
            "Force re-encoding of embedded MP3 audio instead of copying the existing stream "
            "unchanged."
        )
        set_accessible(
            self.reencode_checkbox,
            "Re-encode MP3 streams",
            "Force re-encoding instead of stream-copying embedded MP3 audio.",
        )
        self.normalize_checkbox = QCheckBox("Normali&ze loudness", options_card)
        self.normalize_checkbox.setObjectName("inputs_normalize_checkbox")
        self.normalize_checkbox.setToolTip(
            "Apply two-pass loudness normalization during audio extraction."
        )
        set_accessible(
            self.normalize_checkbox,
            "Normalize loudness",
            "Apply two-pass FFmpeg loudnorm normalization during extraction.",
        )
        self.artwork_checkbox = QCheckBox("Embed cover &art", options_card)
        self.artwork_checkbox.setObjectName("inputs_artwork_checkbox")
        self.artwork_checkbox.setChecked(True)
        self.artwork_checkbox.setToolTip(
            "Fetch and embed front-cover artwork inside exported MP3 files."
        )
        set_accessible(
            self.artwork_checkbox,
            "Embed cover art",
            "Fetch and embed front-cover artwork into exported MP3 files.",
        )
        self.flat_checkbox = QCheckBox("Flat la&yout", options_card)
        self.flat_checkbox.setObjectName("inputs_flat_checkbox")
        self.flat_checkbox.setToolTip(
            "Write MP3 files directly into the output folder without Artist/Album subfolders."
        )
        set_accessible(
            self.flat_checkbox,
            "Flat output layout",
            "Write MP3 files directly into the output root without Artist/Album folders.",
        )
        self.fail_fast_checkbox = QCheckBox("Stop at first fail&ure", options_card)
        self.fail_fast_checkbox.setObjectName("inputs_fail_fast_checkbox")
        self.fail_fast_checkbox.setToolTip(
            "Stop launching new track extractions as soon as any track fails."
        )
        set_accessible(
            self.fail_fast_checkbox,
            "Stop at first failure",
            "Stop starting remaining extractions after the first track failure.",
        )

        flags_grid = QGridLayout()
        flags_grid.setContentsMargins(0, 0, 0, 0)
        flags_grid.setHorizontalSpacing(14)
        flags_grid.setVerticalSpacing(8)
        flags_grid.addWidget(self.reencode_checkbox, 0, 0)
        flags_grid.addWidget(self.normalize_checkbox, 0, 1)
        flags_grid.addWidget(self.artwork_checkbox, 0, 2)
        flags_grid.addWidget(self.flat_checkbox, 1, 0)
        flags_grid.addWidget(self.fail_fast_checkbox, 1, 1, 1, 2)
        flags_grid.setColumnStretch(0, 1)
        flags_grid.setColumnStretch(1, 1)
        flags_grid.setColumnStretch(2, 1)
        options_card.body_layout.addLayout(flags_grid)
        layout.addWidget(options_card)

        # Collapsible "Pin tags" card styled to match elevated SectionCards.
        pin_container = QFrame(column)
        pin_container.setObjectName("inputs_pin_tags_card")
        pin_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        pin_container.setFrameShape(QFrame.Shape.StyledPanel)
        pin_container.setFrameShadow(QFrame.Shadow.Plain)
        pin_container.setProperty("afvSurface", "card")
        pin_outer = QVBoxLayout(pin_container)
        pin_outer.setContentsMargins(16, 10, 16, 10)
        pin_outer.setSpacing(8)

        pin_header_row = QHBoxLayout()
        pin_header_row.setContentsMargins(0, 0, 0, 0)
        pin_header_row.setSpacing(6)

        tag_icon = ThemedSvgLabel("tag", size=16, parent=pin_container)
        tag_icon.setObjectName("inputs_pin_tags_icon")
        pin_header_row.addWidget(tag_icon, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.pin_tags_toggle = QToolButton(pin_container)
        self.pin_tags_toggle.setObjectName("inputs_pin_tags_toggle")
        self.pin_tags_toggle.setText("Pin ta&gs (optional overrides)")
        self.pin_tags_toggle.setCheckable(True)
        self.pin_tags_toggle.setChecked(False)
        self.pin_tags_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.pin_tags_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.pin_tags_toggle.setAutoRaise(True)
        self.pin_tags_toggle.setProperty("afvRole", "disclosure")
        self.pin_tags_toggle.setToolTip(
            "Expand or collapse manual tag override fields applied to tracks in this run."
        )
        set_accessible(
            self.pin_tags_toggle,
            "Pin tags section toggle",
            "Expand or collapse manual tag override fields.",
        )
        pin_header_row.addWidget(self.pin_tags_toggle)

        self.pin_tags_hint_label = make_muted_label("", scale=0.88, parent=pin_container)
        self.pin_tags_hint_label.setObjectName("inputs_pin_tags_hint_label")
        self.pin_tags_hint_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.pin_tags_hint_label.setVisible(False)
        pin_header_row.addWidget(self.pin_tags_hint_label, stretch=1)
        pin_outer.addLayout(pin_header_row)

        self.pin_tags_content = QWidget(pin_container)
        self.pin_tags_content.setObjectName("inputs_pin_tags_content")
        pin_content_layout = QVBoxLayout(self.pin_tags_content)
        pin_content_layout.setContentsMargins(0, 0, 0, 4)
        pin_content_layout.setSpacing(8)
        pin_content_layout.addWidget(make_divider(self.pin_tags_content))

        pin_form = QFormLayout()
        pin_form.setContentsMargins(4, 2, 4, 0)
        pin_form.setHorizontalSpacing(12)
        pin_form.setVerticalSpacing(8)
        pin_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        pin_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        artist_label = QLabel("Artist:", self.pin_tags_content)
        self.artist_edit = QLineEdit(self.pin_tags_content)
        self.artist_edit.setObjectName("inputs_artist_edit")
        self.artist_edit.setPlaceholderText("Optional artist override")
        self.artist_edit.setClearButtonEnabled(True)
        self.artist_edit.setToolTip("Override the artist tag for every track in this run.")
        artist_label.setBuddy(self.artist_edit)
        set_accessible(
            self.artist_edit,
            "Pin artist tag",
            "Override the artist tag for every track in this run.",
        )
        pin_form.addRow(artist_label, self.artist_edit)

        album_artist_label = QLabel("Album artist:", self.pin_tags_content)
        self.album_artist_edit = QLineEdit(self.pin_tags_content)
        self.album_artist_edit.setObjectName("inputs_album_artist_edit")
        self.album_artist_edit.setPlaceholderText("Optional album artist override")
        self.album_artist_edit.setClearButtonEnabled(True)
        self.album_artist_edit.setToolTip(
            "Override the album artist tag for every track in this run."
        )
        album_artist_label.setBuddy(self.album_artist_edit)
        set_accessible(
            self.album_artist_edit,
            "Pin album artist tag",
            "Override the album artist tag for every track in this run.",
        )
        pin_form.addRow(album_artist_label, self.album_artist_edit)

        album_label = QLabel("Album:", self.pin_tags_content)
        self.album_edit = QLineEdit(self.pin_tags_content)
        self.album_edit.setObjectName("inputs_album_edit")
        self.album_edit.setPlaceholderText("Optional album override")
        self.album_edit.setClearButtonEnabled(True)
        self.album_edit.setToolTip("Override the album tag for every track in this run.")
        album_label.setBuddy(self.album_edit)
        set_accessible(
            self.album_edit,
            "Pin album tag",
            "Override the album tag for every track in this run.",
        )
        pin_form.addRow(album_label, self.album_edit)

        title_label = QLabel("Title:", self.pin_tags_content)
        self.title_edit = QLineEdit(self.pin_tags_content)
        self.title_edit.setObjectName("inputs_title_edit")
        self.title_edit.setPlaceholderText("Single input only")
        self.title_edit.setClearButtonEnabled(True)
        self.title_edit.setToolTip(
            "Override the track title tag when queuing a single file or single-file folder."
        )
        title_label.setBuddy(self.title_edit)
        set_accessible(
            self.title_edit,
            "Pin title tag",
            "Override the track title tag. Enabled when a single file or folder is selected.",
        )
        pin_form.addRow(title_label, self.title_edit)

        track_label = QLabel("Track:", self.pin_tags_content)
        self.track_spin = QSpinBox(self.pin_tags_content)
        self.track_spin.setObjectName("inputs_track_spin")
        self.track_spin.setRange(0, 999)
        self.track_spin.setValue(0)
        self.track_spin.setSpecialValueText("None")
        self.track_spin.setMaximumWidth(120)
        self.track_spin.setToolTip(
            "Override the track number tag when queuing a single file or single-file folder."
        )
        track_label.setBuddy(self.track_spin)
        set_accessible(
            self.track_spin,
            "Pin track number",
            "Override the track number tag. Enabled when a single file or folder is selected.",
        )
        pin_form.addRow(track_label, self.track_spin)

        self.single_file_hint_label = make_muted_label(
            "Title and track number apply only to a single-file input.",
            scale=0.9,
            parent=self.pin_tags_content,
        )
        self.single_file_hint_label.setObjectName("inputs_single_file_hint_label")
        self.single_file_hint_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        pin_form.addRow("", self.single_file_hint_label)

        pin_content_layout.addLayout(pin_form)
        self.pin_tags_content.setVisible(False)
        pin_outer.addWidget(self.pin_tags_content)
        layout.addWidget(pin_container)

        layout.addStretch(1)
        scroll.setWidget(body)
        outer_layout.addWidget(scroll, stretch=1)

        # Sticky bottom action bar kept outside the scroll area so Scan and plan is always visible.
        footer = ActionBarFooter(self, margins=(24, 10, 24, 12), spacing=12)
        footer.setObjectName("inputs_footer_bar")
        self._footer_summary_label = MiddleElideLabel("0 inputs queued · Online · 320k", footer)
        self._footer_summary_label.setObjectName("inputs_footer_summary_label")
        self._footer_summary_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._footer_summary_label.setProperty("afvText", "muted")
        set_accessible(
            self._footer_summary_label,
            "Queued run summary",
            "Live summary of queued input count, lookup mode, and audio quality.",
        )
        footer.content_layout.addWidget(self._footer_summary_label, stretch=1)

        self.scan_button = QPushButton("&Scan and plan", footer)
        self.scan_button.setObjectName("inputs_scan_button")
        apply_primary_button_role(self.scan_button, icon_name="play")
        self.scan_button.setDefault(True)
        self.scan_button.setEnabled(False)
        self.scan_button.setToolTip(
            "Scan the queued files and folders and build an extraction plan."
        )
        set_accessible(
            self.scan_button,
            "Scan and plan",
            "Discover audio tracks and build an extraction plan for the queued inputs.",
        )
        footer.content_layout.addWidget(self.scan_button)
        outer_layout.addWidget(footer)

        # Keyboard shortcuts: Ctrl/Cmd+O opens the file chooser; Delete/Backspace remove rows.
        self.open_files_shortcut = QShortcut(QKeySequence.StandardKey.Open, self)
        self.open_files_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.open_files_shortcut.activated.connect(self._on_add_files)

        self._delete_shortcut = QShortcut(QKeySequence.StandardKey.Delete, self.inputs_list)
        self._delete_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._delete_shortcut.activated.connect(self.remove_selected)

        self._backspace_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Backspace),
            self.inputs_list,
        )
        self._backspace_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._backspace_shortcut.activated.connect(self.remove_selected)

        # Load initial defaults from controller.config (or diagnosed fallback once when ready).
        if self._controller.config is not None:
            self.load_defaults(
                self._controller.config,
                offline_default=self._controller.offline_default,
            )
            self._apply_config_warning_message(None)
        elif self._defaults_initialized:
            diagnosis = self._resolve_fallback_diagnosis(show_when_none=False)
            self._effective_fallback_config = diagnosis.effective_config
            self._apply_config_defaults(
                diagnosis.effective_config,
                is_valid_config=False,
                reset_all=True,
            )
            use_offline = bool(self._controller.offline_default)
            self._set_lookup_mode_offline(use_offline)
            self._last_offline_default = use_offline
            self._apply_config_warning_message(diagnosis.banner_text)
        else:
            self.load_defaults(
                None,
                offline_default=self._controller.offline_default,
            )
            self._last_offline_default = None

        # Wire UI controls and controller signals via bound methods only.
        self.add_files_button.clicked.connect(self._on_add_files)
        self.add_folder_button.clicked.connect(self._on_add_folder)
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button.clicked.connect(self.clear_inputs)
        self.inputs_list.itemSelectionChanged.connect(self._update_action_states)
        self.output_dir_edit.textChanged.connect(self._on_output_dir_changed)
        self.output_browse_button.clicked.connect(self._on_browse_output)
        self.output_reset_button.clicked.connect(self._on_reset_output)
        self.online_radio.toggled.connect(self._on_mode_toggled)
        self.recursive_checkbox.toggled.connect(self._clear_mode_notice)
        self.always_prompt_checkbox.toggled.connect(self._clear_mode_notice)
        self.keep_native_checkbox.toggled.connect(self._clear_mode_notice)
        self.quality_combo.currentTextChanged.connect(self._on_quality_changed)
        self.conflict_combo.currentIndexChanged.connect(self._clear_mode_notice)
        self.jobs_spin.valueChanged.connect(self._clear_mode_notice)
        self.lang_edit.textChanged.connect(self._clear_mode_notice)
        self.stream_spin.valueChanged.connect(self._clear_mode_notice)
        self.reencode_checkbox.toggled.connect(self._clear_mode_notice)
        self.normalize_checkbox.toggled.connect(self._clear_mode_notice)
        self.artwork_checkbox.toggled.connect(self._clear_mode_notice)
        self.flat_checkbox.toggled.connect(self._clear_mode_notice)
        self.fail_fast_checkbox.toggled.connect(self._clear_mode_notice)
        self.artist_edit.textChanged.connect(self._clear_mode_notice)
        self.album_artist_edit.textChanged.connect(self._clear_mode_notice)
        self.album_edit.textChanged.connect(self._clear_mode_notice)
        self.title_edit.textChanged.connect(self._clear_mode_notice)
        self.track_spin.valueChanged.connect(self._clear_mode_notice)
        self.pin_tags_toggle.toggled.connect(self._on_pin_tags_toggled)
        self.scan_button.clicked.connect(self._on_scan_and_plan)

        self._controller.input_error.connect(self._on_input_error)
        self._controller.setup_error.connect(self._on_setup_error)
        self._controller.settings_saved.connect(self._on_settings_saved)
        self._controller.state_changed.connect(self._on_state_changed)

        self._update_action_states()
        self._update_output_preview()

        # Logical tab order across interactive widgets.
        tab_chain: tuple[QWidget, ...] = (
            self.settings_button,
            self.recursive_checkbox,
            self.inputs_list,
            self.add_files_button,
            self.add_folder_button,
            self.remove_button,
            self.clear_button,
            self.output_dir_edit,
            self.output_browse_button,
            self.output_reset_button,
            self.online_radio,
            self.offline_radio,
            self.always_prompt_checkbox,
            self.keep_native_checkbox,
            self.quality_combo,
            self.conflict_combo,
            self.jobs_spin,
            self.lang_edit,
            self.stream_spin,
            self.reencode_checkbox,
            self.normalize_checkbox,
            self.artwork_checkbox,
            self.flat_checkbox,
            self.fail_fast_checkbox,
            self.pin_tags_toggle,
            self.artist_edit,
            self.album_artist_edit,
            self.album_edit,
            self.title_edit,
            self.track_spin,
            self.scan_button,
        )
        for prev_widget, next_widget in itertools.pairwise(tab_chain):
            self.setTabOrder(prev_widget, next_widget)

    @property
    def output_highlighted(self) -> bool:
        """Return True while the output folder picker is highlighted for ``multiple_roots``."""
        return self._output_highlighted

    def input_paths(self) -> tuple[Path, ...]:
        """Return queued input paths in display order."""
        return tuple(self._paths)

    @Slot()
    def _clear_mode_notice(self) -> None:
        """Clear and hide the brief mode-switch notice without affecting other banners."""
        self.mode_notice_banner.clear_message()

    @Slot(str)
    def _on_quality_changed(self, _text: str) -> None:
        """Clear the mode notice and refresh the sticky footer summary text."""
        self._clear_mode_notice()
        self._update_footer_summary()

    def _update_footer_summary(self) -> None:
        """Refresh the live run summary displayed beside ``Scan and plan``."""
        count = len(self._paths)
        noun = "input" if count == 1 else "inputs"
        mode = "Offline" if self.offline_radio.isChecked() else "Online"
        quality = self.quality_combo.currentText().strip() or self._default_quality
        self._footer_summary_label.setText(f"{count} {noun} queued · {mode} · {quality}")

    def _set_lookup_mode_offline(self, offline: bool) -> None:
        """Select the Offline or Online radio button and refresh dependent checkboxes."""
        if offline:
            self.offline_radio.setChecked(True)
        else:
            self.online_radio.setChecked(True)
        self._on_mode_toggled(self.online_radio.isChecked())

    def _apply_config_defaults(
        self,
        config: AppConfig | None,
        *,
        is_valid_config: bool | None = None,
        reset_all: bool = False,
    ) -> None:
        """Populate non-mode per-run option widgets from ``config`` without changing the mode.

        When ``reset_all`` is ``False`` and a prior defaults snapshot exists, only
        controls whose underlying default changed **and** whose current value still
        equals the previous default are updated so explicit user edits always
        survive both Settings saves and external ``config.toml`` reloads.
        """
        cfg = config if config is not None else AppConfig()
        new_key = _config_defaults_key(cfg)
        prev_key = None if reset_all else self._loaded_defaults_key

        self._default_quality = cfg.audio.quality
        if prev_key is None or (
            new_key[0] != prev_key[0] and self.quality_combo.currentText().strip() == prev_key[0]
        ):
            self.quality_combo.setCurrentText(cfg.audio.quality)
        if prev_key is None or (
            new_key[3] != prev_key[3] and self.flat_checkbox.isChecked() == prev_key[3]
        ):
            self.flat_checkbox.setChecked(cfg.output.flat)
        if prev_key is None or (
            new_key[4] != prev_key[4]
            and str(self.conflict_combo.currentData()) == prev_key[4].value
        ):
            idx = self.conflict_combo.findData(cfg.output.on_conflict.value)
            if idx >= 0:
                self.conflict_combo.setCurrentIndex(idx)
        target_jobs = max(1, cfg.execution.jobs if cfg.execution.jobs else default_jobs())
        if target_jobs > self.jobs_spin.maximum():
            self.jobs_spin.setMaximum(target_jobs)
        if prev_key is None:
            self.jobs_spin.setValue(target_jobs)
        elif new_key[5] != prev_key[5]:
            prev_jobs = max(1, prev_key[5] if prev_key[5] else default_jobs())
            if self.jobs_spin.value() == prev_jobs:
                self.jobs_spin.setValue(target_jobs)

        valid_flag = (config is not None) if is_valid_config is None else is_valid_config
        self._had_valid_config = valid_flag
        if valid_flag:
            self._effective_fallback_config = None
            self._preflight_setup_error = ""
            self.config_warning_banner.clear_message()
        self._loaded_defaults_key = new_key if (config is not None or valid_flag) else None
        self._update_output_preview(config=cfg)
        self._update_footer_summary()

    def load_defaults(
        self,
        config: AppConfig | None,
        *,
        offline_default: bool | None = None,
    ) -> None:
        """Populate per-run option widgets from ``config`` and ``offline_default``.

        This never modifies ``QSettings`` or ``config.toml``.
        """
        self._apply_config_defaults(config, reset_all=True)
        use_offline = bool(
            self._controller.offline_default if offline_default is None else offline_default
        )
        self._set_lookup_mode_offline(use_offline)
        self._last_offline_default = use_offline

    def _apply_config_warning_message(self, message: str | None) -> None:
        """Display or hide ``config_warning_banner`` for ``message``."""
        if message is not None:
            self.config_warning_banner.set_message(
                "status_warn",
                message,
                role_label="Configuration warning",
            )
        else:
            self.config_warning_banner.clear_message()

    def _resolve_fallback_diagnosis(self, *, show_when_none: bool) -> _FallbackConfigDiagnosis:
        """Read fallback defaults via ``controller.fallback_config_state`` and classify banner."""
        state = self._controller.fallback_config_state
        if state is not None:
            effective = state.config
            banner_text = _classify_invalid_config_banner(
                target_path=effective.config_path,
                from_file=state.from_file,
                file_error=state.file_error,
                show_when_none=show_when_none,
                active_env=os.environ,
                preflight_error=self._preflight_setup_error,
            )
            return _FallbackConfigDiagnosis(
                effective_config=effective,
                from_file=state.from_file,
                banner_text=banner_text,
            )
        return _diagnose_config_fallback(
            show_when_none=show_when_none,
            preflight_error=self._preflight_setup_error,
        )

    def _update_config_warning_banner(
        self,
        config: AppConfig | None,
        *,
        show_when_none: bool = False,
    ) -> None:
        """Show a warning banner when entering ``INPUTS`` with an invalid configuration."""
        if config is not None:
            self._effective_fallback_config = None
            self._preflight_setup_error = ""
            self.config_warning_banner.clear_message()
            return
        diagnosis = self._resolve_fallback_diagnosis(show_when_none=show_when_none)
        self._effective_fallback_config = diagnosis.effective_config
        self._apply_config_warning_message(diagnosis.banner_text)

    def add_paths(self, paths: Iterable[Path]) -> None:
        """Add local file or folder paths to the queue, avoiding duplicates."""
        existing = {str(p): True for p in self._paths}
        added = False
        for raw in paths:
            path = Path(raw)
            key = str(path)
            if key in existing:
                continue
            existing[key] = True
            self._paths.append(path)
            is_dir = _is_directory_like(path.expanduser())
            self._path_is_dir.append(is_dir)
            item = QListWidgetItem(str(path), self.inputs_list)
            item.setToolTip(str(path))
            item.setData(int(Qt.ItemDataRole.UserRole), path)
            item.setIcon(self._folder_icon if is_dir else self._file_icon)
            added = True
        if added:
            self.error_banner.clear_message()
            self._clear_mode_notice()
            self._set_output_highlighted(False)
            self._update_action_states()
            self._update_output_preview()

    @Slot()
    def remove_selected(self) -> None:
        """Remove currently selected rows from the input queue."""
        selected_rows = sorted(
            {self.inputs_list.row(item) for item in self.inputs_list.selectedItems()},
            reverse=True,
        )
        for row in selected_rows:
            self.inputs_list.takeItem(row)
            del self._paths[row]
            del self._path_is_dir[row]
        if selected_rows:
            self._clear_mode_notice()
        self._set_output_highlighted(False)
        self._update_action_states()
        self._update_output_preview()

    @Slot()
    def clear_inputs(self) -> None:
        """Remove all paths from the input queue."""
        self.inputs_list.clear()
        self._paths.clear()
        self._path_is_dir.clear()
        self._cleared_single_track_notice = False
        self.error_banner.clear_message()
        self._clear_mode_notice()
        self._set_output_highlighted(False)
        self._update_action_states()
        self._update_output_preview()

    def is_single_file_input(self) -> bool:
        """Return True when exactly one non-directory file is queued."""
        if len(self._paths) != 1:
            return False
        return not self._path_is_dir[0]

    def allows_single_track_overrides(self) -> bool:
        """Return True when Title and Track overrides may be edited.

        A single queued directory is permitted here because the CLI allows
        ``--title`` and ``--track`` when that directory resolves to a single
        video file (enforced by ``validate_manual_overrides`` during planning).
        """
        return len(self._paths) == 1

    def build_run_options(self) -> RunOptions:
        """Construct a frozen ``RunOptions`` snapshot from the current UI controls."""
        output_text = self.output_dir_edit.text().strip()
        output_dir = Path(output_text) if output_text else None

        quality_text = self.quality_combo.currentText().strip()
        quality: str | None = None
        if quality_text and quality_text != self._default_quality:
            parse_quality(quality_text)
            quality = quality_text

        conflict_data = self.conflict_combo.currentData()
        try:
            conflict_policy = ConflictPolicy(str(conflict_data))
        except ValueError:
            conflict_policy = ConflictPolicy.SKIP

        lang_text = self.lang_edit.text().strip()
        lang = lang_text or None

        stream_val = self.stream_spin.value()
        stream = None if stream_val < 0 else stream_val

        offline = self.offline_radio.isChecked()
        allow_single_track = self.allows_single_track_overrides()

        artist = self.artist_edit.text().strip() or None
        album_artist = self.album_artist_edit.text().strip() or None
        album = self.album_edit.text().strip() or None
        title = (self.title_edit.text().strip() or None) if allow_single_track else None
        raw_track = self.track_spin.value()
        track = raw_track if (allow_single_track and raw_track >= 1) else None

        overrides = ManualOverrides(
            artist=artist,
            album_artist=album_artist,
            album=album,
            title=title,
            track=track,
        )

        return RunOptions(
            inputs=tuple(self._paths),
            recursive=self.recursive_checkbox.isChecked(),
            output_dir=output_dir,
            offline=offline,
            always_prompt=(False if offline else self.always_prompt_checkbox.isChecked()),
            keep_native_missing=(False if offline else self.keep_native_checkbox.isChecked()),
            quality=quality,
            reencode=self.reencode_checkbox.isChecked(),
            normalize=self.normalize_checkbox.isChecked(),
            no_artwork=not self.artwork_checkbox.isChecked(),
            flat=self.flat_checkbox.isChecked(),
            on_conflict=conflict_policy,
            jobs=self.jobs_spin.value(),
            fail_fast=self.fail_fast_checkbox.isChecked(),
            lang=lang,
            stream=stream,
            overrides=overrides,
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if _has_local_urls(event):
            self.drop_zone.set_drag_active(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if _has_local_urls(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        self.drop_zone.set_drag_active(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        self.drop_zone.set_drag_active(False)
        paths = _extract_dropped_paths(event)
        if paths:
            self.add_paths(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _set_output_highlighted(self, highlighted: bool) -> None:
        self._output_highlighted = highlighted
        set_widget_role(
            self.output_frame,
            "afvHighlight",
            "warn" if highlighted else "none",
        )
        if highlighted:
            self.output_dir_edit.setFocus()

    def _update_output_preview(self, *, config: object = _UNSET) -> None:
        if config is _UNSET:
            active_config = (
                self._controller.config
                if self._controller.config is not None
                else self._effective_fallback_config
            )
        else:
            active_config = config
        explicit = self.output_dir_edit.text().strip()
        self.output_reset_button.setEnabled(bool(explicit))
        preview, multi_root = _compute_default_output_preview(
            self._paths,
            active_config if isinstance(active_config, AppConfig) else None,
            is_dir_flags=self._path_is_dir,
        )
        if explicit:
            self.output_dir_edit.setPlaceholderText("Leave blank to use default folder")
            status_text = f"Output folder: {explicit}"
            self.output_default_status.set_status("status_ok", status_text)
            self.output_default_status.setToolTip(explicit)
        elif multi_root:
            self.output_dir_edit.setPlaceholderText("Choose an output folder (required)")
            self.output_default_status.set_status("status_warn", preview)
            self.output_default_status.setToolTip(preview)
        else:
            self.output_dir_edit.setPlaceholderText("Leave blank to use default folder")
            self.output_default_status.set_status("status_info", preview)
            self.output_default_status.setToolTip(preview)

    @Slot()
    def _update_action_states(self) -> None:
        has_items = len(self._paths) > 0
        has_selection = bool(self.inputs_list.selectedItems())
        shortcut_label = _open_shortcut_text()
        self.remove_button.setEnabled(has_selection)
        self.clear_button.setEnabled(has_items)
        self.scan_button.setEnabled(has_items)
        self.inputs_list.updateGeometry()
        self.drop_hint_label.setText(
            f"Drop video files or folders here, or use Add files ({shortcut_label}) or Add folder."
            if not has_items
            else f"{len(self._paths)} input item{'s' if len(self._paths) != 1 else ''} queued. "
            "Drop more files or folders to add them."
        )
        count = len(self._paths)
        allow_single_track = count == 1
        if count == 0:
            self._cleared_single_track_notice = False
            self.title_edit.setEnabled(False)
            self.track_spin.setEnabled(False)
            self.single_file_hint_label.setText(
                "Queue a single file or folder to pin title and track number overrides."
            )
            self.single_file_hint_label.setVisible(True)
            self.pin_tags_hint_label.setText("")
            self.pin_tags_hint_label.setVisible(False)
        elif allow_single_track:
            self._cleared_single_track_notice = False
            self.title_edit.setEnabled(True)
            self.track_spin.setEnabled(True)
            if self._path_is_dir[0]:
                self.single_file_hint_label.setText(
                    "Title and track number apply only if this folder resolves to a single "
                    "video file."
                )
                self.single_file_hint_label.setVisible(True)
                self.pin_tags_hint_label.setText("Single folder queued")
                self.pin_tags_hint_label.setVisible(True)
            else:
                self.single_file_hint_label.setVisible(False)
                self.pin_tags_hint_label.setText("")
                self.pin_tags_hint_label.setVisible(False)
        else:
            had_values = bool(self.title_edit.text().strip()) or self.track_spin.value() >= 1
            if had_values:
                self.title_edit.clear()
                self.track_spin.setValue(0)
                self._cleared_single_track_notice = True
            self.title_edit.setEnabled(False)
            self.track_spin.setEnabled(False)
            if self._cleared_single_track_notice:
                self.pin_tags_hint_label.setText("Title and track cleared (multiple inputs)")
                self.single_file_hint_label.setText(
                    "Title and track number apply only to a single-file input; previous values "
                    "were cleared."
                )
            else:
                self.pin_tags_hint_label.setText("Title and track disabled (multiple inputs)")
                self.single_file_hint_label.setText(
                    "Title and track number apply only to a single-file input."
                )
            self.single_file_hint_label.setVisible(True)
            self.pin_tags_hint_label.setVisible(True)
        self._update_footer_summary()

    @Slot(bool)
    def _on_mode_toggled(self, online_checked: bool) -> None:
        self.always_prompt_checkbox.setEnabled(online_checked)
        self.keep_native_checkbox.setEnabled(online_checked)
        self._clear_mode_notice()
        self._update_footer_summary()

    @Slot(bool)
    def _on_pin_tags_toggled(self, expanded: bool) -> None:
        self.pin_tags_content.setVisible(expanded)
        self.pin_tags_toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )

    @Slot(str)
    def _on_output_dir_changed(self, text: str) -> None:
        if text.strip():
            self._set_output_highlighted(False)
        self._clear_mode_notice()
        self._update_output_preview()

    @Slot()
    def _on_add_files(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS))
        file_filter = f"Video files ({patterns});;All files (*)"
        settings = open_ui_settings(self)
        start_dir = str(settings.value(_LAST_INPUTS_FOLDER_KEY, "") or "")
        selected, _ = QFileDialog.getOpenFileNames(
            self,
            "Select video files",
            start_dir,
            file_filter,
        )
        if not selected:
            return
        picked = [Path(p) for p in selected]
        settings.setValue(_LAST_INPUTS_FOLDER_KEY, str(picked[0].parent))
        self.add_paths(picked)

    @Slot()
    def _on_add_folder(self) -> None:
        settings = open_ui_settings(self)
        start_dir = str(settings.value(_LAST_INPUTS_FOLDER_KEY, "") or "")
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select folder containing videos",
            start_dir,
        )
        if not selected:
            return
        folder = Path(selected)
        settings.setValue(_LAST_INPUTS_FOLDER_KEY, str(folder))
        self.add_paths([folder])

    @Slot()
    def _on_browse_output(self) -> None:
        settings = open_ui_settings(self)
        current = self.output_dir_edit.text().strip()
        start_dir = current or str(settings.value(_LAST_OUTPUT_FOLDER_KEY, "") or "")
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            start_dir,
        )
        if not selected:
            return
        settings.setValue(_LAST_OUTPUT_FOLDER_KEY, selected)
        self.output_dir_edit.setText(selected)
        self.error_banner.clear_message()
        self._set_output_highlighted(False)

    @Slot()
    def _on_reset_output(self) -> None:
        self.output_dir_edit.clear()
        self._update_output_preview()

    @Slot()
    def _on_scan_and_plan(self) -> None:
        if not self._paths:
            return
        try:
            options = self.build_run_options()
        except ConfigError as exc:
            self._on_input_error("validation", str(exc))
            return
        self.error_banner.clear_message()
        self._clear_mode_notice()
        self._set_output_highlighted(False)
        self._controller.start_plan(options)

    @Slot(str, str)
    def _on_input_error(self, kind: str, message: str) -> None:
        clean_kind = kind.strip().lower()
        text = message.strip()
        if clean_kind == "no_inputs":
            if not text:
                text = "No supported video files found."
            elif "no supported video files found" not in text.casefold():
                text = f"No supported video files found. {text}"
        elif clean_kind == "multiple_roots":
            self._set_output_highlighted(True)
            if not text:
                text = (
                    "Inputs span multiple directories. Choose an explicit output folder to "
                    "continue."
                )
            self.output_default_status.set_status("status_error", text)
            self.output_default_status.setToolTip(text)
        elif not text:
            text = "Check your selected inputs and options before continuing."

        if clean_kind != "multiple_roots":
            self._set_output_highlighted(False)

        self._clear_mode_notice()
        if clean_kind == "config_changed":
            self.error_banner.set_message(
                "status_info",
                text,
                role_label="Configuration updated",
            )
            return
        self.error_banner.set_message("status_error", text, role_label="Input error")

    @Slot(str)
    def _on_setup_error(self, message: str) -> None:
        if self._controller.state == ControllerState.PREFLIGHT:
            self._preflight_setup_error = message.strip()

    @Slot(object)
    def _on_settings_saved(self, config: AppConfig | None) -> None:
        was_initialized = self._defaults_initialized
        if config is not None:
            new_key = _config_defaults_key(config)
            should_reload = (
                not self._defaults_initialized
                or not self._had_valid_config
                or new_key != self._loaded_defaults_key
            )
            if should_reload:
                self._apply_config_defaults(
                    config,
                    is_valid_config=True,
                    reset_all=not self._defaults_initialized,
                )
            else:
                self._effective_fallback_config = None
                self._preflight_setup_error = ""
                self._update_output_preview(config=config)
            self._defaults_initialized = True
            self._apply_config_warning_message(None)
        else:
            diagnosis = self._resolve_fallback_diagnosis(show_when_none=True)
            self._effective_fallback_config = diagnosis.effective_config
            new_key = _config_defaults_key(diagnosis.effective_config)
            if not self._defaults_initialized:
                self._apply_config_defaults(
                    diagnosis.effective_config,
                    is_valid_config=False,
                    reset_all=True,
                )
                self._defaults_initialized = True
            elif new_key != self._loaded_defaults_key:
                self._apply_config_defaults(
                    diagnosis.effective_config,
                    is_valid_config=False,
                    reset_all=False,
                )
            else:
                self._had_valid_config = False
                self._update_output_preview(config=diagnosis.effective_config)
            self._apply_config_warning_message(diagnosis.banner_text)

        offline_default = bool(self._controller.offline_default)
        if (
            not was_initialized
            or self._last_offline_default is None
            or offline_default != self._last_offline_default
        ):
            self._set_lookup_mode_offline(offline_default)
            self._last_offline_default = offline_default

        self._clear_mode_notice()

    @Slot(object)
    def _on_state_changed(self, state: object) -> None:
        prev = self._last_state
        self._last_state = state
        if state == ControllerState.PREFLIGHT:
            self._preflight_setup_error = ""
            self._clear_mode_notice()
            return
        if state != ControllerState.INPUTS:
            self._clear_mode_notice()
            return
        config = self._controller.config
        if prev in (None, ControllerState.PREFLIGHT, ControllerState.SETUP):
            self.error_banner.clear_message()
            self._set_output_highlighted(False)
            was_initialized = self._defaults_initialized
            was_online = self.online_radio.isChecked()
            switched_to_offline = False
            if config is not None:
                new_key = _config_defaults_key(config)
                should_reload = (
                    not self._defaults_initialized
                    or not self._had_valid_config
                    or new_key != self._loaded_defaults_key
                )
                if should_reload:
                    self._apply_config_defaults(
                        config,
                        is_valid_config=True,
                        reset_all=not was_initialized,
                    )
                    offline_default = bool(self._controller.offline_default)
                    if (
                        not was_initialized
                        or (prev == ControllerState.SETUP and offline_default)
                        or (
                            self._last_offline_default is None
                            or offline_default != self._last_offline_default
                        )
                    ):
                        self._set_lookup_mode_offline(offline_default)
                    self._last_offline_default = offline_default
                    self._defaults_initialized = True
                    switched_to_offline = (
                        prev == ControllerState.SETUP
                        and was_initialized
                        and was_online
                        and self.offline_radio.isChecked()
                        and self._controller.offline_default
                    )
                else:
                    if prev == ControllerState.SETUP and self._controller.offline_default:
                        self._set_lookup_mode_offline(True)
                        switched_to_offline = was_initialized and was_online
                    self._last_offline_default = bool(self._controller.offline_default)
                    self._update_output_preview()
                self._apply_config_warning_message(None)
            else:
                diagnosis = self._resolve_fallback_diagnosis(show_when_none=True)
                self._effective_fallback_config = diagnosis.effective_config
                new_key = _config_defaults_key(diagnosis.effective_config)
                if not was_initialized:
                    self._apply_config_defaults(
                        diagnosis.effective_config,
                        is_valid_config=False,
                        reset_all=True,
                    )
                    use_offline = bool(self._controller.offline_default)
                    self._set_lookup_mode_offline(use_offline)
                    self._last_offline_default = use_offline
                    self._defaults_initialized = True
                else:
                    if new_key != self._loaded_defaults_key:
                        self._apply_config_defaults(
                            diagnosis.effective_config,
                            is_valid_config=False,
                            reset_all=False,
                        )
                    else:
                        self._had_valid_config = False
                        self._update_output_preview(config=diagnosis.effective_config)
                    if prev == ControllerState.SETUP and self._controller.offline_default:
                        self._set_lookup_mode_offline(True)
                        switched_to_offline = was_initialized and was_online
                    self._last_offline_default = bool(self._controller.offline_default)
                self._apply_config_warning_message(diagnosis.banner_text)
            if switched_to_offline:
                self.mode_notice_banner.set_message(
                    "status_info",
                    OFFLINE_SWITCH_NOTICE_TEXT,
                    role_label="Lookup mode notice",
                )
            else:
                self._clear_mode_notice()
        elif config is not None:
            self._update_config_warning_banner(config)

    def changeEvent(self, event: QEvent) -> None:
        """Refresh themed SVG icons on live palette or style changes."""
        super().changeEvent(event)
        if not is_theme_updating() and event.type() in (
            QEvent.Type.PaletteChange,
            QEvent.Type.ApplicationPaletteChange,
            QEvent.Type.StyleChange,
        ):
            refresh_themed_widgets(self)
