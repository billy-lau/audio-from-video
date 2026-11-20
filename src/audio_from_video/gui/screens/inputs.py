"""Inputs screen for choosing media files, folders, output directory, and run options."""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import (
    QDragEnterEvent,
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
    QGroupBox,
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
    validate_mb_contact,
    validate_mb_user_agent,
)
from audio_from_video.discovery import DEFAULT_OUTPUT_FOLDER_NAME, VIDEO_EXTENSIONS
from audio_from_video.gui.controller import ControllerState
from audio_from_video.gui.resources import load_icon, render_svg_pixmap
from audio_from_video.gui.screens._common import (
    QUALITY_PRESETS,
    IconStatusBanner,
    StatusLine,
    make_bold_font,
    make_heading_label,
    open_ui_settings,
    set_accessible,
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
INVALID_UNKNOWN_CONFIG_BANNER_TEXT = (
    "Configuration is invalid (config.toml or AFV_* environment); using built-in defaults."
)
OFFLINE_SWITCH_NOTICE_TEXT = "Switched to Offline mode because start-offline is enabled."
_UNSET = object()

_DefaultsKey = tuple[str, bool, str | None, bool, ConflictPolicy, int]


def _join_oxford(items: Sequence[str]) -> str:
    """Join ``items`` into a prose list using the Oxford comma."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _find_invalid_env_overrides(active_env: Mapping[str, str]) -> list[tuple[str, str]]:
    """Return ``(env_var, error)`` pairs for invalid ``AFV_*`` environment overrides."""
    issues: list[tuple[str, str]] = []
    raw_contact = active_env.get("AFV_MB_CONTACT", "").strip()
    if raw_contact:
        try:
            validate_mb_contact(raw_contact)
        except ConfigError as exc:
            issues.append(("AFV_MB_CONTACT", str(exc)))
    raw_ua = active_env.get("AFV_MB_USER_AGENT", "").strip()
    if raw_ua:
        try:
            validate_mb_user_agent(raw_ua)
        except ConfigError as exc:
            issues.append(("AFV_MB_USER_AGENT", str(exc)))
    return issues


def _invalid_config_banner_text(
    *,
    show_when_none: bool,
    env: Mapping[str, str] | None = None,
    preflight_error: str = "",
) -> str | None:
    """Classify why ``controller.config`` is ``None`` without falsely blaming ``config.toml``."""
    active_env = os.environ if env is None else env
    file_error: str | None = None
    has_config_file = False
    try:
        target_path = default_config_path(active_env)
        has_config_file = target_path.is_file()
    except OSError as exc:
        target_path = None
        file_error = str(exc)

    if file_error is None and target_path is not None and has_config_file:
        try:
            load_config(config_path=target_path, env={})
        except (ConfigError, OSError) as exc:
            file_error = str(exc)

    env_issues = _find_invalid_env_overrides(active_env)
    env_load_error: str | None = None
    if file_error is None and not env_issues and target_path is not None:
        try:
            load_config(config_path=target_path, env=active_env)
        except (ConfigError, OSError) as exc:
            env_load_error = str(exc)

    if file_error is not None:
        if env_issues:
            sources = _join_oxford(["config.toml", *(var for var, _detail in env_issues)])
            return f"Configuration is invalid ({sources}); using built-in defaults."
        return INVALID_CONFIG_BANNER_TEXT

    if env_issues:
        names = _join_oxford([var for var, _detail in env_issues])
        detail = env_issues[0][1].strip()
        if detail and "config.toml" not in detail.casefold():
            return f"{INVALID_ENV_CONFIG_BANNER_TEXT} Check {names}: {detail}"
        return f"{INVALID_ENV_CONFIG_BANNER_TEXT} Check {names}."

    if env_load_error is not None:
        cleaned = env_load_error.strip()
        if cleaned and "config.toml" not in cleaned.casefold():
            return f"{INVALID_ENV_CONFIG_BANNER_TEXT} {cleaned}"
        return INVALID_ENV_CONFIG_BANNER_TEXT

    if not show_when_none:
        return None

    cleaned_preflight = preflight_error.strip()
    if (
        cleaned_preflight
        and "AFV_" in cleaned_preflight
        and "config.toml" not in cleaned_preflight.casefold()
    ):
        return f"{INVALID_ENV_CONFIG_BANNER_TEXT} {cleaned_preflight}"
    return INVALID_UNKNOWN_CONFIG_BANNER_TEXT


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
    """Frame accepting file and folder drag-and-drop without adding custom Qt signals."""

    def __init__(
        self,
        on_paths_dropped: Callable[[list[Path]], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_paths_dropped = on_paths_dropped
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if _has_local_urls(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if _has_local_urls(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
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
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_paths_dropped = on_paths_dropped
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if _has_local_urls(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if _has_local_urls(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
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

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        set_accessible(scroll, "Inputs content", "Scrollable container for input controls.")

        body = QWidget(scroll)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # Header row with Settings button.
        header_row = QHBoxLayout()
        header_row.setSpacing(12)
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        self.title_label = make_heading_label("Select Videos to Extract", parent=body)
        subtitle = QLabel(
            "Options changed here apply only to this run. Edit saved defaults in Settings.",
            body,
        )
        subtitle.setWordWrap(True)
        title_box.addWidget(self.title_label)
        title_box.addWidget(subtitle)
        header_row.addLayout(title_box, stretch=1)

        self.settings_button = QPushButton("Se&ttings…", body)
        self.settings_button.setIcon(load_icon("settings"))
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
        self.config_warning_banner = IconStatusBanner(body)
        layout.addWidget(self.config_warning_banner)

        # Inline error banner (hidden until input_error or local validation failure).
        self.error_banner = IconStatusBanner(body)
        layout.addWidget(self.error_banner)

        # Brief neutral notice shown when a Setup detour switches Online to Offline.
        self.mode_notice_banner = IconStatusBanner(body)
        layout.addWidget(self.mode_notice_banner)

        # Input files and drop zone section.
        inputs_group = QGroupBox("Input Files and Folders", body)
        inputs_layout = QVBoxLayout(inputs_group)
        inputs_layout.setSpacing(8)

        self.drop_zone = _DropTargetFrame(self.add_paths, inputs_group)
        self.drop_zone.setFrameShape(QFrame.Shape.StyledPanel)
        self.drop_zone.setFrameShadow(QFrame.Shadow.Sunken)
        set_accessible(
            self.drop_zone,
            "Input drop zone",
            "Drop video files or folders here to add them to the extraction queue.",
        )
        drop_layout = QVBoxLayout(self.drop_zone)
        drop_layout.setContentsMargins(10, 8, 10, 8)
        drop_layout.setSpacing(6)

        hint_row = QHBoxLayout()
        hint_row.setSpacing(8)
        drop_icon = QLabel(self.drop_zone)
        drop_icon.setPixmap(render_svg_pixmap("drop_zone", size=20))
        drop_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.drop_hint_label = QLabel(
            "Drop video files or folders here, or use Add files (Ctrl/Cmd+O) or Add folder.",
            self.drop_zone,
        )
        self.drop_hint_label.setWordWrap(True)
        hint_row.addWidget(drop_icon)
        hint_row.addWidget(self.drop_hint_label, stretch=1)
        drop_layout.addLayout(hint_row)

        self.inputs_list = _DropListWidget(self.add_paths, self.drop_zone)
        self.inputs_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.inputs_list.setMinimumHeight(110)
        self.inputs_list.setAlternatingRowColors(True)
        self.inputs_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        set_accessible(
            self.inputs_list,
            "Queued input files and folders",
            "List of selected video files and folders to scan and plan.",
        )
        drop_layout.addWidget(self.inputs_list)
        inputs_layout.addWidget(self.drop_zone)

        toolbar_row = QHBoxLayout()
        toolbar_row.setSpacing(8)
        self.add_files_button = QPushButton("Add &files…", inputs_group)
        self.add_files_button.setIcon(self._file_icon)
        self.add_files_button.setToolTip("Add one or more video files (Ctrl/Cmd+O).")
        set_accessible(
            self.add_files_button,
            "Add files",
            "Choose video files to add to the batch queue (shortcut Ctrl/Cmd+O).",
        )

        self.add_folder_button = QPushButton("Add f&older…", inputs_group)
        self.add_folder_button.setIcon(self._folder_icon)
        set_accessible(
            self.add_folder_button,
            "Add folder",
            "Choose a folder of video files to scan.",
        )

        self.remove_button = QPushButton("&Remove", inputs_group)
        self.remove_button.setEnabled(False)
        set_accessible(
            self.remove_button,
            "Remove selected inputs",
            "Remove the highlighted files or folders from the input queue.",
        )

        self.clear_button = QPushButton("C&lear", inputs_group)
        self.clear_button.setEnabled(False)
        set_accessible(
            self.clear_button,
            "Clear all inputs",
            "Remove every file and folder from the input queue.",
        )

        self.recursive_checkbox = QCheckBox("Include su&bfolders", inputs_group)
        self.recursive_checkbox.setChecked(False)
        set_accessible(
            self.recursive_checkbox,
            "Include subfolders",
            "Scan added folders recursively for supported video files.",
        )

        toolbar_row.addWidget(self.add_files_button)
        toolbar_row.addWidget(self.add_folder_button)
        toolbar_row.addWidget(self.remove_button)
        toolbar_row.addWidget(self.clear_button)
        toolbar_row.addStretch(1)
        toolbar_row.addWidget(self.recursive_checkbox)
        inputs_layout.addLayout(toolbar_row)
        layout.addWidget(inputs_group)

        # Output folder picker.
        output_group = QGroupBox("Output Folder", body)
        output_layout = QVBoxLayout(output_group)
        output_layout.setSpacing(6)

        self.output_frame = QFrame(output_group)
        self.output_frame.setFrameShape(QFrame.Shape.NoFrame)
        self.output_frame.setLineWidth(1)
        frame_layout = QHBoxLayout(self.output_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(8)

        output_label = QLabel("Out&put folder:", self.output_frame)
        self.output_dir_edit = QLineEdit(self.output_frame)
        self.output_dir_edit.setClearButtonEnabled(True)
        output_label.setBuddy(self.output_dir_edit)
        set_accessible(
            self.output_dir_edit,
            "Output folder",
            "Optional destination directory for extracted MP3 files. Leave blank for default.",
        )

        self.output_browse_button = QPushButton("Bro&wse…", self.output_frame)
        self.output_browse_button.setIcon(self._folder_icon)
        set_accessible(
            self.output_browse_button,
            "Browse output folder",
            "Choose a destination folder for extracted MP3 files.",
        )

        self.output_reset_button = QPushButton("Use &default", self.output_frame)
        self.output_reset_button.setEnabled(False)
        set_accessible(
            self.output_reset_button,
            "Use default output folder",
            "Clear the explicit output folder and return to the computed default.",
        )

        frame_layout.addWidget(output_label)
        frame_layout.addWidget(self.output_dir_edit, stretch=1)
        frame_layout.addWidget(self.output_browse_button)
        frame_layout.addWidget(self.output_reset_button)
        output_layout.addWidget(self.output_frame)

        self.output_default_status = StatusLine(
            "status_info",
            f"Default: <input folder>/{DEFAULT_OUTPUT_FOLDER_NAME}",
            parent=output_group,
        )
        output_layout.addWidget(self.output_default_status)
        layout.addWidget(output_group)

        # Run options panel (per-run only, never written back to QSettings or config.toml).
        options_group = QGroupBox("Run Options (this run only)", body)
        options_grid = QGridLayout(options_group)
        options_grid.setHorizontalSpacing(16)
        options_grid.setVerticalSpacing(8)

        # Mode toggle (Online vs Offline).
        mode_row = QHBoxLayout()
        mode_row.setSpacing(12)
        mode_label = QLabel("Lookup mode:", options_group)
        self.mode_group = QButtonGroup(self)
        self.online_radio = QRadioButton("O&nline", options_group)
        self.offline_radio = QRadioButton("Offl&ine", options_group)
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
        mode_row.addWidget(self.online_radio)
        mode_row.addWidget(self.offline_radio)
        mode_row.addStretch(1)
        options_grid.addWidget(mode_label, 0, 0)
        options_grid.addLayout(mode_row, 0, 1, 1, 3)

        self.always_prompt_checkbox = QCheckBox("Confirm every &match", options_group)
        self.always_prompt_checkbox.setChecked(False)
        self.always_prompt_checkbox.setToolTip(
            "Prompt before accepting every online match instead of only uncertain matches."
        )
        set_accessible(
            self.always_prompt_checkbox,
            "Confirm every match",
            "Always show the match dialog even for high-confidence online matches.",
        )

        self.keep_native_checkbox = QCheckBox("Keep nati&ve names", options_group)
        self.keep_native_checkbox.setChecked(False)
        self.keep_native_checkbox.setToolTip(
            "Keep non-Latin artist or album names when no English alias is available."
        )
        set_accessible(
            self.keep_native_checkbox,
            "Keep native names",
            "Keep native non-Latin artist and album names without prompting for an English alias.",
        )
        options_grid.addWidget(self.always_prompt_checkbox, 1, 0, 1, 2)
        options_grid.addWidget(self.keep_native_checkbox, 1, 2, 1, 2)

        quality_label = QLabel("&Quality:", options_group)
        self.quality_combo = QComboBox(options_group)
        self.quality_combo.setEditable(True)
        self.quality_combo.addItems(QUALITY_PRESETS)
        quality_label.setBuddy(self.quality_combo)
        set_accessible(
            self.quality_combo,
            "Audio quality",
            "MP3 bitrate such as 320k, 192k, or LAME VBR preset v0 through v9.",
        )
        options_grid.addWidget(quality_label, 2, 0)
        options_grid.addWidget(self.quality_combo, 2, 1)

        conflict_label = QLabel("On &conflict:", options_group)
        self.conflict_combo = QComboBox(options_group)
        self.conflict_combo.addItem("Skip existing", ConflictPolicy.SKIP.value)
        self.conflict_combo.addItem("Overwrite", ConflictPolicy.OVERWRITE.value)
        self.conflict_combo.addItem("Rename", ConflictPolicy.RENAME.value)
        conflict_label.setBuddy(self.conflict_combo)
        set_accessible(
            self.conflict_combo,
            "On conflict policy",
            "Action when a destination file already exists: skip, overwrite, or rename.",
        )
        options_grid.addWidget(conflict_label, 2, 2)
        options_grid.addWidget(self.conflict_combo, 2, 3)

        jobs_label = QLabel("&Jobs:", options_group)
        self.jobs_spin = QSpinBox(options_group)
        self.jobs_spin.setRange(1, 256)
        jobs_label.setBuddy(self.jobs_spin)
        set_accessible(
            self.jobs_spin,
            "Parallel jobs",
            "Number of parallel FFmpeg extraction workers.",
        )
        options_grid.addWidget(jobs_label, 3, 0)
        options_grid.addWidget(self.jobs_spin, 3, 1)

        lang_label = QLabel("Preferred audio language:", options_group)
        self.lang_edit = QLineEdit(options_group)
        self.lang_edit.setPlaceholderText("e.g., eng or jpn")
        self.lang_edit.setClearButtonEnabled(True)
        lang_label.setBuddy(self.lang_edit)
        self.lang_edit.setToolTip(
            "Optional ISO 639-2 language code to prefer when choosing an audio stream."
        )
        set_accessible(
            self.lang_edit,
            "Preferred audio language",
            "Optional preferred audio track language code, such as eng or jpn.",
        )
        options_grid.addWidget(lang_label, 3, 2)
        options_grid.addWidget(self.lang_edit, 3, 3)

        stream_label = QLabel("Audio str&eam index:", options_group)
        self.stream_spin = QSpinBox(options_group)
        self.stream_spin.setRange(-1, 63)
        self.stream_spin.setValue(-1)
        self.stream_spin.setSpecialValueText("Auto")
        stream_label.setBuddy(self.stream_spin)
        self.stream_spin.setToolTip(
            "Explicit 0-based audio stream index (--stream), or Auto to select automatically."
        )
        set_accessible(
            self.stream_spin,
            "Audio stream index",
            "Explicit zero-based audio stream index, or Auto to pick automatically.",
        )
        options_grid.addWidget(stream_label, 4, 0)
        options_grid.addWidget(self.stream_spin, 4, 1)

        self.reencode_checkbox = QCheckBox("Re-encode MP&3", options_group)
        set_accessible(
            self.reencode_checkbox,
            "Re-encode MP3 streams",
            "Force re-encoding instead of stream-copying embedded MP3 audio.",
        )
        self.normalize_checkbox = QCheckBox("Normali&ze loudness", options_group)
        set_accessible(
            self.normalize_checkbox,
            "Normalize loudness",
            "Apply two-pass FFmpeg loudnorm normalization during extraction.",
        )
        self.artwork_checkbox = QCheckBox("Cover &art", options_group)
        self.artwork_checkbox.setChecked(True)
        set_accessible(
            self.artwork_checkbox,
            "Embed cover art",
            "Fetch and embed front-cover artwork into exported MP3 files.",
        )
        self.flat_checkbox = QCheckBox("Flat la&yout", options_group)
        set_accessible(
            self.flat_checkbox,
            "Flat output layout",
            "Write MP3 files directly into the output root without Artist/Album folders.",
        )
        self.fail_fast_checkbox = QCheckBox("Stop at first fail&ure", options_group)
        set_accessible(
            self.fail_fast_checkbox,
            "Stop at first failure",
            "Stop starting remaining extractions after the first track failure (--fail-fast).",
        )

        flags_row1 = QHBoxLayout()
        flags_row1.setSpacing(14)
        flags_row1.addWidget(self.reencode_checkbox)
        flags_row1.addWidget(self.normalize_checkbox)
        flags_row1.addWidget(self.artwork_checkbox)
        flags_row1.addWidget(self.flat_checkbox)
        flags_row1.addWidget(self.fail_fast_checkbox)
        flags_row1.addStretch(1)
        options_grid.addLayout(flags_row1, 5, 0, 1, 4)
        layout.addWidget(options_group)

        # Collapsible "Pin tags" section.
        pin_container = QFrame(body)
        pin_container.setFrameShape(QFrame.Shape.StyledPanel)
        pin_outer = QVBoxLayout(pin_container)
        pin_outer.setContentsMargins(10, 6, 10, 8)
        pin_outer.setSpacing(6)

        self.pin_tags_toggle = QToolButton(pin_container)
        self.pin_tags_toggle.setText("Pin ta&gs (optional overrides)")
        self.pin_tags_toggle.setCheckable(True)
        self.pin_tags_toggle.setChecked(False)
        self.pin_tags_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.pin_tags_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.pin_tags_toggle.setAutoRaise(True)
        set_accessible(
            self.pin_tags_toggle,
            "Pin tags section toggle",
            "Expand or collapse manual tag override fields.",
        )
        pin_header_row = QHBoxLayout()
        pin_header_row.setContentsMargins(0, 0, 0, 0)
        pin_header_row.setSpacing(10)
        pin_header_row.addWidget(self.pin_tags_toggle)
        self.pin_tags_hint_label = QLabel("", pin_container)
        self.pin_tags_hint_label.setTextFormat(Qt.TextFormat.PlainText)
        self.pin_tags_hint_label.setWordWrap(True)
        self.pin_tags_hint_label.setVisible(False)
        pin_header_row.addWidget(self.pin_tags_hint_label, stretch=1)
        pin_outer.addLayout(pin_header_row)

        self.pin_tags_content = QWidget(pin_container)
        pin_form = QFormLayout(self.pin_tags_content)
        pin_form.setContentsMargins(6, 4, 6, 4)
        pin_form.setSpacing(6)

        self.artist_edit = QLineEdit(self.pin_tags_content)
        self.artist_edit.setPlaceholderText("Optional artist override")
        set_accessible(
            self.artist_edit,
            "Pin artist tag",
            "Override the artist tag for every track in this run.",
        )
        pin_form.addRow("Artist:", self.artist_edit)

        self.album_artist_edit = QLineEdit(self.pin_tags_content)
        self.album_artist_edit.setPlaceholderText("Optional album artist override")
        set_accessible(
            self.album_artist_edit,
            "Pin album artist tag",
            "Override the album artist tag for every track in this run.",
        )
        pin_form.addRow("Album artist:", self.album_artist_edit)

        self.album_edit = QLineEdit(self.pin_tags_content)
        self.album_edit.setPlaceholderText("Optional album override")
        set_accessible(
            self.album_edit,
            "Pin album tag",
            "Override the album tag for every track in this run.",
        )
        pin_form.addRow("Album:", self.album_edit)

        self.title_edit = QLineEdit(self.pin_tags_content)
        self.title_edit.setPlaceholderText("Single input only")
        set_accessible(
            self.title_edit,
            "Pin title tag",
            "Override the track title tag. Enabled when a single file or folder is selected.",
        )
        pin_form.addRow("Title:", self.title_edit)

        self.track_spin = QSpinBox(self.pin_tags_content)
        self.track_spin.setRange(0, 999)
        self.track_spin.setValue(0)
        self.track_spin.setSpecialValueText("None")
        set_accessible(
            self.track_spin,
            "Pin track number",
            "Override the track number tag. Enabled when a single file or folder is selected.",
        )
        pin_form.addRow("Track:", self.track_spin)

        self.single_file_hint_label = QLabel(
            "Title and track number apply only to a single-file input.",
            self.pin_tags_content,
        )
        self.single_file_hint_label.setWordWrap(True)
        pin_form.addRow("", self.single_file_hint_label)

        self.pin_tags_content.setVisible(False)
        pin_outer.addWidget(self.pin_tags_content)
        layout.addWidget(pin_container)

        # Primary footer action.
        footer_row = QHBoxLayout()
        footer_row.addStretch(1)
        self.scan_button = QPushButton("&Scan and plan", body)
        self.scan_button.setFont(make_bold_font(self.scan_button))
        self.scan_button.setDefault(True)
        self.scan_button.setEnabled(False)
        set_accessible(
            self.scan_button,
            "Scan and plan",
            "Discover audio tracks and build an extraction plan for the queued inputs.",
        )
        footer_row.addWidget(self.scan_button)
        layout.addLayout(footer_row)
        layout.addStretch(1)

        scroll.setWidget(body)
        outer_layout.addWidget(scroll)

        # Keyboard shortcut: Ctrl/Cmd+O opens the file chooser.
        self.open_files_shortcut = QShortcut(QKeySequence.StandardKey.Open, self)
        self.open_files_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.open_files_shortcut.activated.connect(self._on_add_files)

        # Load initial defaults from controller.config and controller.offline_default.
        self.load_defaults(
            self._controller.config, offline_default=self._controller.offline_default
        )
        if not self._defaults_initialized:
            self._last_offline_default = None
        if self._defaults_initialized:
            self._update_config_warning_banner(self._controller.config, show_when_none=False)

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
        self.quality_combo.currentTextChanged.connect(self._clear_mode_notice)
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
            self.inputs_list,
            self.add_files_button,
            self.add_folder_button,
            self.remove_button,
            self.clear_button,
            self.recursive_checkbox,
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

    def _set_lookup_mode_offline(self, offline: bool) -> None:
        """Select the Offline or Online radio button and refresh dependent checkboxes."""
        if offline:
            self.offline_radio.setChecked(True)
        else:
            self.online_radio.setChecked(True)
        self._on_mode_toggled(self.online_radio.isChecked())

    def _apply_config_defaults(self, config: AppConfig | None) -> None:
        """Populate non-mode per-run option widgets from ``config`` without changing the mode."""
        cfg = config if config is not None else AppConfig()
        self._default_quality = cfg.audio.quality
        self.quality_combo.setCurrentText(cfg.audio.quality)
        self.flat_checkbox.setChecked(cfg.output.flat)
        idx = self.conflict_combo.findData(cfg.output.on_conflict.value)
        if idx >= 0:
            self.conflict_combo.setCurrentIndex(idx)
        target_jobs = max(1, cfg.execution.jobs if cfg.execution.jobs else default_jobs())
        if target_jobs > self.jobs_spin.maximum():
            self.jobs_spin.setMaximum(target_jobs)
        self.jobs_spin.setValue(target_jobs)
        self._had_valid_config = config is not None
        self._loaded_defaults_key = _config_defaults_key(config) if config is not None else None
        if config is not None:
            self._preflight_setup_error = ""
            self.config_warning_banner.clear_message()
        self._update_output_preview(config=config)

    def load_defaults(
        self,
        config: AppConfig | None,
        *,
        offline_default: bool | None = None,
    ) -> None:
        """Populate per-run option widgets from ``config`` and ``offline_default``.

        This never modifies ``QSettings`` or ``config.toml``.
        """
        self._apply_config_defaults(config)
        use_offline = bool(
            self._controller.offline_default if offline_default is None else offline_default
        )
        self._set_lookup_mode_offline(use_offline)
        self._last_offline_default = use_offline

    def _update_config_warning_banner(
        self,
        config: AppConfig | None,
        *,
        show_when_none: bool = False,
    ) -> None:
        """Show a warning banner when entering ``INPUTS`` with an invalid configuration."""
        if config is not None:
            self._preflight_setup_error = ""
            self.config_warning_banner.clear_message()
            return
        message = _invalid_config_banner_text(
            show_when_none=show_when_none,
            preflight_error=self._preflight_setup_error,
        )
        if message is not None:
            self.config_warning_banner.set_message(
                "status_warn",
                message,
                role_label="Configuration warning",
            )
        else:
            self.config_warning_banner.clear_message()

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
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if _has_local_urls(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = _extract_dropped_paths(event)
        if paths:
            self.add_paths(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _set_output_highlighted(self, highlighted: bool) -> None:
        self._output_highlighted = highlighted
        if highlighted:
            self.output_frame.setFrameShape(QFrame.Shape.Box)
            self.output_frame.setFrameShadow(QFrame.Shadow.Plain)
            self.output_frame.setLineWidth(2)
            self.output_dir_edit.setFocus()
        else:
            self.output_frame.setFrameShape(QFrame.Shape.NoFrame)
            self.output_frame.setLineWidth(1)

    def _update_output_preview(self, *, config: object = _UNSET) -> None:
        active_config = self._controller.config if config is _UNSET else config
        explicit = self.output_dir_edit.text().strip()
        self.output_reset_button.setEnabled(bool(explicit))
        preview, multi_root = _compute_default_output_preview(
            self._paths,
            active_config if isinstance(active_config, AppConfig) else None,
            is_dir_flags=self._path_is_dir,
        )
        self.output_dir_edit.setPlaceholderText(preview)
        if explicit:
            self.output_default_status.set_status("status_ok", f"Output folder: {explicit}")
        elif multi_root:
            self.output_default_status.set_status("status_warn", preview)
        else:
            self.output_default_status.set_status("status_info", preview)

    @Slot()
    def _update_action_states(self) -> None:
        has_items = len(self._paths) > 0
        has_selection = bool(self.inputs_list.selectedItems())
        self.remove_button.setEnabled(has_selection)
        self.clear_button.setEnabled(has_items)
        self.scan_button.setEnabled(has_items)
        self.drop_hint_label.setText(
            "Drop video files or folders here, or use Add files (Ctrl/Cmd+O) or Add folder."
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
                hint = (
                    "Title and track number apply only if this folder resolves to a single "
                    "video file."
                )
                self.single_file_hint_label.setText(hint)
                self.single_file_hint_label.setVisible(True)
                self.pin_tags_hint_label.setText(hint)
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
                hint = (
                    "Title and track number apply only to a single-file input; cleared previous "
                    "values because multiple inputs are queued."
                )
            else:
                hint = "Title and track number apply only to a single-file input."
            self.single_file_hint_label.setText(hint)
            self.single_file_hint_label.setVisible(True)
            self.pin_tags_hint_label.setText(hint)
            self.pin_tags_hint_label.setVisible(True)

    @Slot(bool)
    def _on_mode_toggled(self, online_checked: bool) -> None:
        self.always_prompt_checkbox.setEnabled(online_checked)
        self.keep_native_checkbox.setEnabled(online_checked)
        self._clear_mode_notice()

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
        elif not text:
            text = "Check your selected inputs and options before continuing."

        if clean_kind != "multiple_roots":
            self._set_output_highlighted(False)

        self._clear_mode_notice()
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
                self._apply_config_defaults(config)
            else:
                self._preflight_setup_error = ""
                self._update_output_preview(config=config)
            self._defaults_initialized = True
        elif not self._defaults_initialized:
            self._apply_config_defaults(None)
            self._defaults_initialized = True
        else:
            self._had_valid_config = False
            self._loaded_defaults_key = None
            self._update_output_preview(config=None)

        offline_default = bool(self._controller.offline_default)
        if (
            not was_initialized
            or self._last_offline_default is None
            or offline_default != self._last_offline_default
        ):
            self._set_lookup_mode_offline(offline_default)
            self._last_offline_default = offline_default

        self._clear_mode_notice()
        self._update_config_warning_banner(config, show_when_none=True)

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
                    self.load_defaults(
                        config,
                        offline_default=self._controller.offline_default,
                    )
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
                self._update_config_warning_banner(config)
            else:
                if not self._defaults_initialized:
                    self.load_defaults(
                        None,
                        offline_default=self._controller.offline_default,
                    )
                    self._defaults_initialized = True
                else:
                    self._had_valid_config = False
                    self._loaded_defaults_key = None
                    if prev == ControllerState.SETUP and self._controller.offline_default:
                        self._set_lookup_mode_offline(True)
                        switched_to_offline = was_initialized and was_online
                    self._last_offline_default = bool(self._controller.offline_default)
                    self._update_output_preview()
                self._update_config_warning_banner(None, show_when_none=True)
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
