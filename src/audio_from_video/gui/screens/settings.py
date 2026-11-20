"""Settings screen for editing config keys, cache maintenance, and GUI defaults."""

from __future__ import annotations

import itertools
import math
import os
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QSignalBlocker, Qt, Slot
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from audio_from_video.cache import NamespaceStats
from audio_from_video.config import (
    EDITABLE_SETTINGS,
    ENV_OVERRIDES,
    AppConfig,
    ConfigError,
    default_config_path,
    default_jobs,
    load_config,
    parse_quality,
    validate_jobs,
    validate_mb_contact,
    validate_mb_user_agent,
)
from audio_from_video.gui.resources import load_icon
from audio_from_video.gui.screens._common import (
    QUALITY_PRESETS,
    IconStatusBanner,
    StatusLine,
    format_byte_size,
    format_ttl_duration,
    log_directory,
    make_bold_font,
    make_heading_label,
    open_local_path,
    open_ui_settings,
    read_bool_setting,
    set_accessible,
)
from audio_from_video.models import ConflictPolicy

if TYPE_CHECKING:
    from audio_from_video.gui.controller import RunController

LOG_LEVEL_KEY = "log/level"
STARTUP_OFFLINE_KEY = "startup/offline"
LOG_LEVEL_CHOICES: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")
_DEFAULT_MAX_JOBS = 256

_KNOWN_SECTION_KEYS: Mapping[str, frozenset[str]] = {
    "binaries": frozenset({"ffmpeg", "ffprobe", "ffmpeg_path", "ffprobe_path"}),
    "audio": frozenset(
        {
            "quality",
            "copy_mp3",
            "loudnorm_mode",
            "loudnorm_i",
            "loudnorm_tp",
            "loudnorm_lra",
        }
    ),
    "output": frozenset({"output_dir", "flat", "on_conflict"}),
    "execution": frozenset({"jobs"}),
    "musicbrainz": frozenset({"contact", "user_agent"}),
    "matching": frozenset({"auto_accept_score", "min_margin", "ambiguous_floor"}),
}

_LOUDNORM_NUMERIC_RANGES: tuple[tuple[str, float, float], ...] = (
    ("loudnorm_i", -70.0, -5.0),
    ("loudnorm_tp", -9.0, 0.0),
    ("loudnorm_lra", 1.0, 50.0),
)
_MATCHING_KEYS: tuple[str, ...] = ("auto_accept_score", "min_margin", "ambiguous_floor")


def _join_oxford(items: Sequence[str]) -> str:
    """Join ``items`` as a human-readable list using the Oxford comma."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _is_finite_number(value: object) -> bool:
    """Return True when ``value`` is a finite ``int`` or ``float`` (excluding ``bool``)."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _path_exists(path: Path) -> bool:
    """Return True if ``path`` exists without raising on OS errors."""
    try:
        return path.expanduser().exists()
    except OSError:
        return False


@dataclass(frozen=True, slots=True)
class _RawInvalidValue:
    """Marker preserving a raw TOML value whose type or choice is invalid."""

    raw: object


def _setting_values_equal(left: object, right: object) -> bool:
    """Compare normalized setting values strictly without conflating ``bool`` and ``int``."""
    if type(left) is not type(right):
        return False
    return left == right


def _normalize_optional_str(value: object) -> str | None:
    """Return a stripped non-empty string or ``None``."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _is_cache_related_error(title: str, details: str) -> bool:
    """Return True when a controller error describes a cache or busy failure."""
    haystack = f"{title} {details}".casefold()
    return "cache" in haystack or "busy" in haystack


def _is_cache_busy_error(title: str, details: str) -> bool:
    """Return True when a controller error reports a transient cache-busy notice."""
    title_cf = title.casefold()
    details_cf = details.casefold()
    return (
        "cache busy" in title_cf
        or ("cache" in title_cf and "busy" in details_cf)
        or ("cache" in details_cf and "preflight" in details_cf)
    )


class SettingsScreen(QWidget):
    """Settings screen widget.

    Listens to ``cache_stats_ready``, ``cache_cleared``, and ``settings_saved``.
    """

    def __init__(
        self,
        controller: RunController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._overridden_keys: dict[str, str] = {}
        self._pre_override_texts: dict[str, str] = {}
        self._baseline_values: dict[str, object] = {}
        self._editable_config_issues: dict[str, str] = {}
        self._uneditable_config_issues: list[tuple[str, str]] = []
        self._raw_load_error_text: str | None = None
        self._loaded_from_raw_toml = False
        self._raw_validation_warning: str | None = None
        self._config_load_blocked = False
        self._cache_clear_pending = False
        self._cache_stats_pending = False
        self._cache_notice_is_busy = False
        self._pending_ui_preferences: tuple[str, bool] | None = None
        self._ui_rollback_snapshot: tuple[bool, object, bool, object] | None = None
        self._loaded_ui_preferences: tuple[str, bool] = ("INFO", False)
        self._config_path: Path = (
            controller.config.config_path
            if controller.config is not None
            else default_config_path()
        )

        set_accessible(
            self,
            "Settings screen",
            "Edit configuration keys, inspect environment overrides, manage the cache, "
            "and configure startup and logging preferences.",
        )

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        set_accessible(scroll, "Settings content", "Scrollable container for settings controls.")

        body = QWidget(scroll)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        self.title_label = make_heading_label("Settings", parent=body)
        layout.addWidget(self.title_label)

        # Inline status/error banner (for save confirmation and ConfigError text).
        self.status_banner = IconStatusBanner(body)
        layout.addWidget(self.status_banner)

        # Config file path row.
        config_group = QGroupBox("Configuration File", body)
        config_row = QHBoxLayout(config_group)
        config_row.setSpacing(8)
        self.config_path_edit = QLineEdit(str(self._config_path), config_group)
        self.config_path_edit.setReadOnly(True)
        set_accessible(
            self.config_path_edit,
            "Configuration file path",
            "Path to the local config.toml file.",
        )
        self.open_config_button = QPushButton("&Open config file", config_group)
        self.open_config_button.setIcon(load_icon("folder"))
        set_accessible(
            self.open_config_button,
            "Open config file",
            "Open the config.toml file or its containing folder.",
        )
        config_row.addWidget(self.config_path_edit, stretch=1)
        config_row.addWidget(self.open_config_button)
        layout.addWidget(config_group)

        # Editable configuration keys (all keys in config.EDITABLE_SETTINGS).
        self.env_override_labels: dict[str, StatusLine] = {}
        self.invalid_field_labels: dict[str, StatusLine] = {}

        # -- Binaries --
        binaries_group = QGroupBox("Binaries", body)
        binaries_form = QFormLayout(binaries_group)
        binaries_form.setSpacing(6)

        ffmpeg_wrap, self.ffmpeg_edit, self.browse_ffmpeg_button = self._make_path_row(
            parent=binaries_group,
            placeholder="Auto-detect from PATH",
            acc_name="FFmpeg binary path",
            acc_desc="Explicit path to the ffmpeg executable (blank to use PATH).",
            browse_text="Browse…",
            browse_name="Browse for ffmpeg binary",
        )
        self.env_override_labels["binaries.ffmpeg_path"] = StatusLine(
            "status_info",
            "",
            parent=binaries_group,
        )
        self.env_override_labels["binaries.ffmpeg_path"].setVisible(False)
        self.invalid_field_labels["binaries.ffmpeg_path"] = StatusLine(
            "status_warn",
            "",
            parent=binaries_group,
        )
        self.invalid_field_labels["binaries.ffmpeg_path"].setVisible(False)
        ffmpeg_col = QVBoxLayout()
        ffmpeg_col.setSpacing(2)
        ffmpeg_col.addWidget(ffmpeg_wrap)
        ffmpeg_col.addWidget(self.env_override_labels["binaries.ffmpeg_path"])
        ffmpeg_col.addWidget(self.invalid_field_labels["binaries.ffmpeg_path"])
        binaries_form.addRow("ffmpeg:", ffmpeg_col)

        ffprobe_wrap, self.ffprobe_edit, self.browse_ffprobe_button = self._make_path_row(
            parent=binaries_group,
            placeholder="Auto-detect from PATH",
            acc_name="FFprobe binary path",
            acc_desc="Explicit path to the ffprobe executable (blank to use PATH).",
            browse_text="Browse…",
            browse_name="Browse for ffprobe binary",
        )
        self.env_override_labels["binaries.ffprobe_path"] = StatusLine(
            "status_info",
            "",
            parent=binaries_group,
        )
        self.env_override_labels["binaries.ffprobe_path"].setVisible(False)
        self.invalid_field_labels["binaries.ffprobe_path"] = StatusLine(
            "status_warn",
            "",
            parent=binaries_group,
        )
        self.invalid_field_labels["binaries.ffprobe_path"].setVisible(False)
        ffprobe_col = QVBoxLayout()
        ffprobe_col.setSpacing(2)
        ffprobe_col.addWidget(ffprobe_wrap)
        ffprobe_col.addWidget(self.env_override_labels["binaries.ffprobe_path"])
        ffprobe_col.addWidget(self.invalid_field_labels["binaries.ffprobe_path"])
        binaries_form.addRow("ffprobe:", ffprobe_col)
        layout.addWidget(binaries_group)

        # -- Audio, Output, and Execution --
        defaults_group = QGroupBox("Audio, Output, and Execution Defaults", body)
        defaults_form = QFormLayout(defaults_group)
        defaults_form.setSpacing(6)

        self.quality_combo = QComboBox(defaults_group)
        self.quality_combo.setEditable(True)
        self.quality_combo.addItems(QUALITY_PRESETS)
        set_accessible(
            self.quality_combo,
            "Default audio quality",
            "Default MP3 bitrate (such as 320k or 192k) or LAME VBR preset (v0 through v9).",
        )
        self.invalid_field_labels["audio.quality"] = StatusLine(
            "status_warn",
            "",
            parent=defaults_group,
        )
        self.invalid_field_labels["audio.quality"].setVisible(False)
        quality_col = QVBoxLayout()
        quality_col.setSpacing(2)
        quality_col.addWidget(self.quality_combo)
        quality_col.addWidget(self.invalid_field_labels["audio.quality"])
        defaults_form.addRow("Default &quality:", quality_col)

        self.copy_mp3_checkbox = QCheckBox(
            "Copy existing MP3 audio without re-encoding",
            defaults_group,
        )
        set_accessible(
            self.copy_mp3_checkbox,
            "Copy existing MP3 streams",
            "Stream-copy MP3 sources by default instead of re-encoding.",
        )
        self.invalid_field_labels["audio.copy_mp3"] = StatusLine(
            "status_warn",
            "",
            parent=defaults_group,
        )
        self.invalid_field_labels["audio.copy_mp3"].setVisible(False)
        copy_col = QVBoxLayout()
        copy_col.setSpacing(2)
        copy_col.addWidget(self.copy_mp3_checkbox)
        copy_col.addWidget(self.invalid_field_labels["audio.copy_mp3"])
        defaults_form.addRow("", copy_col)

        out_wrap, self.output_dir_edit, self.browse_output_button = self._make_path_row(
            parent=defaults_group,
            placeholder="Default: <input folder>/extracted_audio",
            acc_name="Default output directory",
            acc_desc="Optional default root directory for extracted MP3 files.",
            browse_text="Browse…",
            browse_name="Browse default output folder",
        )
        self.env_override_labels["output.output_dir"] = StatusLine(
            "status_info",
            "",
            parent=defaults_group,
        )
        self.env_override_labels["output.output_dir"].setVisible(False)
        self.invalid_field_labels["output.output_dir"] = StatusLine(
            "status_warn",
            "",
            parent=defaults_group,
        )
        self.invalid_field_labels["output.output_dir"].setVisible(False)
        out_col = QVBoxLayout()
        out_col.setSpacing(2)
        out_col.addWidget(out_wrap)
        out_col.addWidget(self.env_override_labels["output.output_dir"])
        out_col.addWidget(self.invalid_field_labels["output.output_dir"])
        defaults_form.addRow("Output folder:", out_col)

        self.flat_checkbox = QCheckBox(
            "Write files directly into the output folder (flat layout)",
            defaults_group,
        )
        set_accessible(
            self.flat_checkbox,
            "Default flat output layout",
            "Write extracted files directly into the output folder without Artist/Album folders.",
        )
        self.invalid_field_labels["output.flat"] = StatusLine(
            "status_warn",
            "",
            parent=defaults_group,
        )
        self.invalid_field_labels["output.flat"].setVisible(False)
        flat_col = QVBoxLayout()
        flat_col.setSpacing(2)
        flat_col.addWidget(self.flat_checkbox)
        flat_col.addWidget(self.invalid_field_labels["output.flat"])
        defaults_form.addRow("", flat_col)

        self.conflict_combo = QComboBox(defaults_group)
        self._populate_conflict_choices()
        set_accessible(
            self.conflict_combo,
            "Default conflict policy",
            "Default behavior when a destination file already exists: skip, overwrite, or rename.",
        )
        self.invalid_field_labels["output.on_conflict"] = StatusLine(
            "status_warn",
            "",
            parent=defaults_group,
        )
        self.invalid_field_labels["output.on_conflict"].setVisible(False)
        conflict_col = QVBoxLayout()
        conflict_col.setSpacing(2)
        conflict_col.addWidget(self.conflict_combo)
        conflict_col.addWidget(self.invalid_field_labels["output.on_conflict"])
        defaults_form.addRow("On conflict:", conflict_col)

        self.jobs_spin = QSpinBox(defaults_group)
        self.jobs_spin.setRange(1, _DEFAULT_MAX_JOBS)
        set_accessible(
            self.jobs_spin,
            "Default parallel jobs",
            "Default number of parallel FFmpeg workers.",
        )
        self.invalid_field_labels["execution.jobs"] = StatusLine(
            "status_warn",
            "",
            parent=defaults_group,
        )
        self.invalid_field_labels["execution.jobs"].setVisible(False)
        jobs_col = QVBoxLayout()
        jobs_col.setSpacing(2)
        jobs_col.addWidget(self.jobs_spin)
        jobs_col.addWidget(self.invalid_field_labels["execution.jobs"])
        defaults_form.addRow("Parallel &jobs:", jobs_col)
        layout.addWidget(defaults_group)

        # -- MusicBrainz --
        mb_group = QGroupBox("MusicBrainz", body)
        mb_form = QFormLayout(mb_group)
        mb_form.setSpacing(6)

        self.contact_edit = QLineEdit(mb_group)
        self.contact_edit.setPlaceholderText(
            "Reachable email relay or public https:// repository URL"
        )
        self.contact_edit.setClearButtonEnabled(True)
        set_accessible(
            self.contact_edit,
            "MusicBrainz contact",
            "Reachable email relay or public repository URL stored in config.toml.",
        )
        self.env_override_labels["musicbrainz.contact"] = StatusLine(
            "status_info",
            "",
            parent=mb_group,
        )
        self.env_override_labels["musicbrainz.contact"].setVisible(False)
        self.invalid_field_labels["musicbrainz.contact"] = StatusLine(
            "status_warn",
            "",
            parent=mb_group,
        )
        self.invalid_field_labels["musicbrainz.contact"].setVisible(False)
        contact_col = QVBoxLayout()
        contact_col.setSpacing(2)
        contact_col.addWidget(self.contact_edit)
        contact_col.addWidget(self.env_override_labels["musicbrainz.contact"])
        contact_col.addWidget(self.invalid_field_labels["musicbrainz.contact"])
        mb_form.addRow("Contact:", contact_col)

        self.user_agent_edit = QLineEdit(mb_group)
        self.user_agent_edit.setReadOnly(True)
        self.user_agent_edit.setPlaceholderText("Derived automatically from contact")
        set_accessible(
            self.user_agent_edit,
            "User-Agent override",
            "Read-only display of any configured User-Agent override.",
        )
        self.env_override_labels["musicbrainz.user_agent_override"] = StatusLine(
            "status_info",
            "",
            parent=mb_group,
        )
        self.env_override_labels["musicbrainz.user_agent_override"].setVisible(False)
        self.invalid_field_labels["musicbrainz.user_agent_override"] = StatusLine(
            "status_warn",
            "",
            parent=mb_group,
        )
        self.invalid_field_labels["musicbrainz.user_agent_override"].setVisible(False)
        self._user_agent_row_widget = QWidget(mb_group)
        ua_col = QVBoxLayout(self._user_agent_row_widget)
        ua_col.setContentsMargins(0, 0, 0, 0)
        ua_col.setSpacing(2)
        ua_col.addWidget(self.user_agent_edit)
        ua_col.addWidget(self.env_override_labels["musicbrainz.user_agent_override"])
        ua_col.addWidget(self.invalid_field_labels["musicbrainz.user_agent_override"])
        mb_form.addRow("User-Agent:", self._user_agent_row_widget)
        layout.addWidget(mb_group)

        # -- GUI Startup & Logging (QSettings) --
        gui_group = QGroupBox("Desktop Application (stored locally per user)", body)
        gui_form = QFormLayout(gui_group)
        gui_form.setSpacing(6)

        log_row = QHBoxLayout()
        log_row.setSpacing(8)
        self.log_level_combo = QComboBox(gui_group)
        self.log_level_combo.addItems(LOG_LEVEL_CHOICES)
        set_accessible(
            self.log_level_combo,
            "Log level",
            "Logging verbosity saved in application preferences: DEBUG, INFO, WARNING, or ERROR.",
        )
        self.open_log_folder_button = QPushButton("Open lo&g folder", gui_group)
        self.open_log_folder_button.setIcon(load_icon("folder"))
        set_accessible(
            self.open_log_folder_button,
            "Open log folder",
            "Open the application log folder in the system file manager.",
        )
        log_row.addWidget(self.log_level_combo)
        log_row.addWidget(self.open_log_folder_button)
        log_row.addStretch(1)
        gui_form.addRow("Log level:", log_row)

        self.startup_offline_checkbox = QCheckBox("Start in o&ffline mode", gui_group)
        set_accessible(
            self.startup_offline_checkbox,
            "Start in offline mode",
            "Skip first-run MusicBrainz setup and default new batches to offline mode.",
        )
        gui_form.addRow("", self.startup_offline_checkbox)
        layout.addWidget(gui_group)

        # -- Cache Maintenance --
        cache_group = QGroupBox("Metadata and Artwork Cache", body)
        cache_layout = QVBoxLayout(cache_group)
        cache_layout.setSpacing(8)

        self.cache_status = StatusLine(
            "status_pending",
            "Loading cache stats…",
            parent=cache_group,
        )
        cache_layout.addWidget(self.cache_status)

        self.cache_table = QTableWidget(0, 5, cache_group)
        self.cache_table.setHorizontalHeaderLabels(
            ["Namespace", "Entries", "Expired", "Size", "TTL"]
        )
        self.cache_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.cache_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.cache_table.setAlternatingRowColors(True)
        self.cache_table.verticalHeader().setVisible(False)
        self.cache_table.setMaximumHeight(180)
        cache_header = self.cache_table.horizontalHeader()
        cache_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 5):
            cache_header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        set_accessible(
            self.cache_table,
            "Cache statistics table",
            "Per-namespace row count, expired row count, byte size, and TTL.",
        )
        cache_layout.addWidget(self.cache_table)

        cache_buttons = QHBoxLayout()
        cache_buttons.setSpacing(8)
        self.refresh_cache_button = QPushButton("&Refresh stats", cache_group)
        set_accessible(
            self.refresh_cache_button,
            "Refresh cache statistics",
            "Read updated metadata and artwork cache statistics on a worker thread.",
        )
        self.clear_expired_button = QPushButton("Clear &expired", cache_group)
        set_accessible(
            self.clear_expired_button,
            "Clear expired cache entries",
            "Delete only expired entries from the local SQLite cache.",
        )
        self.clear_cache_button = QPushButton("Clear ca&che", cache_group)
        set_accessible(
            self.clear_cache_button,
            "Clear entire cache",
            "Delete all cached MusicBrainz, iTunes, and artwork entries.",
        )
        cache_buttons.addWidget(self.refresh_cache_button)
        cache_buttons.addWidget(self.clear_expired_button)
        cache_buttons.addWidget(self.clear_cache_button)
        cache_buttons.addStretch(1)
        cache_layout.addLayout(cache_buttons)

        self.cache_notice_label = QLabel("", cache_group)
        self.cache_notice_label.setTextFormat(Qt.TextFormat.PlainText)
        self.cache_notice_label.setWordWrap(True)
        cache_layout.addWidget(self.cache_notice_label)
        layout.addWidget(cache_group)

        # Map attribute path -> primary widget for inspection and parity checks.
        self.setting_widgets: dict[str, QWidget] = {
            "binaries.ffmpeg_path": self.ffmpeg_edit,
            "binaries.ffprobe_path": self.ffprobe_edit,
            "audio.quality": self.quality_combo,
            "audio.copy_mp3": self.copy_mp3_checkbox,
            "output.output_dir": self.output_dir_edit,
            "output.flat": self.flat_checkbox,
            "output.on_conflict": self.conflict_combo,
            "execution.jobs": self.jobs_spin,
            "musicbrainz.contact": self.contact_edit,
        }

        # Footer action bar.
        footer = QHBoxLayout()
        footer.setSpacing(10)
        footer.addStretch(1)
        self.close_button = QPushButton("C&lose", body)
        set_accessible(
            self.close_button,
            "Close settings",
            "Close the Settings window without saving unapplied changes.",
        )
        self.save_button = QPushButton("&Save", body)
        self.save_button.setFont(make_bold_font(self.save_button))
        self.save_button.setDefault(True)
        set_accessible(
            self.save_button,
            "Save settings",
            "Validate and save configuration settings and application preferences.",
        )
        footer.addWidget(self.close_button)
        footer.addWidget(self.save_button)
        layout.addLayout(footer)

        scroll.setWidget(body)
        outer_layout.addWidget(scroll)

        # Populate widgets from current config, QSettings, and ENV_OVERRIDES.
        self.load_from_config(self._controller.config)
        self.load_ui_preferences()
        self.refresh_env_overrides()

        # Wire signals using bound methods only.
        self.open_config_button.clicked.connect(self._on_open_config)
        self.browse_ffmpeg_button.clicked.connect(self._on_browse_ffmpeg)
        self.browse_ffprobe_button.clicked.connect(self._on_browse_ffprobe)
        self.browse_output_button.clicked.connect(self._on_browse_output)
        self.open_log_folder_button.clicked.connect(self._on_open_log_folder)
        self.refresh_cache_button.clicked.connect(self.refresh_cache_stats)
        self.clear_expired_button.clicked.connect(self._on_clear_expired)
        self.clear_cache_button.clicked.connect(self._on_clear_cache)
        self.save_button.clicked.connect(self._on_save_clicked)

        self._controller.cache_stats_ready.connect(self._on_cache_stats_ready)
        self._controller.cache_cleared.connect(self._on_cache_cleared)
        self._controller.settings_saved.connect(self._on_settings_saved)
        self._controller.error.connect(self._on_controller_error)

        # Logical tab order.
        tab_chain: tuple[QWidget, ...] = (
            self.config_path_edit,
            self.open_config_button,
            self.ffmpeg_edit,
            self.browse_ffmpeg_button,
            self.ffprobe_edit,
            self.browse_ffprobe_button,
            self.quality_combo,
            self.copy_mp3_checkbox,
            self.output_dir_edit,
            self.browse_output_button,
            self.flat_checkbox,
            self.conflict_combo,
            self.jobs_spin,
            self.contact_edit,
            self.user_agent_edit,
            self.log_level_combo,
            self.open_log_folder_button,
            self.startup_offline_checkbox,
            self.cache_table,
            self.refresh_cache_button,
            self.clear_expired_button,
            self.clear_cache_button,
            self.save_button,
            self.close_button,
        )
        for prev_widget, next_widget in itertools.pairwise(tab_chain):
            self.setTabOrder(prev_widget, next_widget)

    @property
    def loaded_from_raw_toml(self) -> bool:
        """Return True when fields were populated directly from raw ``config.toml``."""
        return self._loaded_from_raw_toml

    @property
    def is_config_load_blocked(self) -> bool:
        """Return True when an unreadable or malformed ``config.toml`` disables saving."""
        return self._config_load_blocked

    @property
    def baseline_values(self) -> dict[str, object]:
        """Return a copy of the snapshot captured at load time for diffing edits."""
        return dict(self._baseline_values)

    @property
    def invalid_editable_keys(self) -> frozenset[str]:
        """Return editable setting keys currently flagged as invalid in raw ``config.toml``."""
        return frozenset(self._editable_config_issues)

    @property
    def uneditable_config_issues(self) -> tuple[tuple[str, str], ...]:
        """Return ``(key_label, detail)`` entries that cannot be edited in Settings."""
        return tuple(self._uneditable_config_issues)

    def _clear_invalid_field_labels(self) -> None:
        self._editable_config_issues.clear()
        self._uneditable_config_issues.clear()
        self._raw_load_error_text = None
        for label in self.invalid_field_labels.values():
            label.set_status("status_warn", "")
            label.setVisible(False)

    def _set_invalid_field_label(self, key: str, message: str) -> None:
        label = self.invalid_field_labels.get(key)
        if label is not None:
            label.set_status("status_warn", message)
            label.setVisible(True)

    def _compose_raw_toml_warning(self, filename: str) -> str | None:
        """Build an actionable warning banner for invalid editable and uneditable keys."""
        editable_labels: list[str] = []
        manual_labels: list[str] = []
        has_needs_edit_field = False

        for attr_path in self._editable_config_issues:
            toml_label = EDITABLE_SETTINGS.get(attr_path, attr_path)
            if attr_path in self._overridden_keys:
                env_var = self._overridden_keys[attr_path]
                manual_labels.append(f"{toml_label} (locked by {env_var})")
                continue
            editable_labels.append(toml_label)
            base = self._baseline_values.get(attr_path)
            if not isinstance(base, _RawInvalidValue) or attr_path == "output.on_conflict":
                has_needs_edit_field = True

        for key_label, _detail in self._uneditable_config_issues:
            if key_label and key_label not in manual_labels:
                manual_labels.append(key_label)

        detail_text = self._raw_load_error_text
        if not detail_text and self._uneditable_config_issues:
            detail_text = self._uneditable_config_issues[0][1]

        if not editable_labels and not manual_labels:
            if detail_text:
                return (
                    f"Existing {filename} has an invalid setting: {detail_text} "
                    "Click Open config file to edit it manually."
                )
            return None

        parts: list[str] = []
        if manual_labels and not editable_labels:
            noun = "setting" if len(manual_labels) == 1 else "settings"
            joined = _join_oxford(manual_labels)
            parts.append(
                f"Existing {filename} has an invalid {noun} that cannot be edited in Settings: "
                f"{joined}."
            )
            if detail_text:
                parts.append(f"Details: {detail_text}")
            parts.append("Click Open config file to edit it manually.")
            return " ".join(parts)

        if editable_labels and manual_labels:
            joined_manual = _join_oxford(manual_labels)
            joined_editable = _join_oxford(editable_labels)
            parts.append(f"Existing {filename} has invalid settings.")
            if detail_text:
                parts.append(str(detail_text))
            parts.append(f"Editable in Settings: {joined_editable}.")
            parts.append(
                f"Cannot be edited in Settings: {joined_manual} "
                "(click Open config file to edit manually)."
            )
            return " ".join(parts)

        joined_editable = _join_oxford(editable_labels)
        field_noun = "field" if len(editable_labels) == 1 else "fields"
        header = (
            f"Existing {filename} has an invalid setting: {detail_text}"
            if detail_text
            else f"Existing {filename} has an invalid setting."
        )
        action = (
            f"Update the highlighted {field_noun} ({joined_editable}) and click Save "
            "to keep your other settings."
            if has_needs_edit_field
            else f"Review the highlighted {field_noun} ({joined_editable}) and click Save "
            "to replace the invalid value while keeping your other settings."
        )
        return f"{header} {action}"

    def _populate_conflict_choices(self) -> None:
        self.conflict_combo.clear()
        self.conflict_combo.addItem("Skip existing", ConflictPolicy.SKIP.value)
        self.conflict_combo.addItem("Overwrite", ConflictPolicy.OVERWRITE.value)
        self.conflict_combo.addItem("Rename", ConflictPolicy.RENAME.value)

    def _set_jobs_spin_value(self, jobs: int) -> None:
        maximum = max(_DEFAULT_MAX_JOBS, jobs)
        if self.jobs_spin.maximum() != maximum:
            self.jobs_spin.setMaximum(maximum)
        self.jobs_spin.setValue(max(1, jobs))

    @staticmethod
    def _make_path_row(
        *,
        parent: QWidget,
        placeholder: str,
        acc_name: str,
        acc_desc: str,
        browse_text: str,
        browse_name: str,
    ) -> tuple[QWidget, QLineEdit, QPushButton]:
        container = QWidget(parent)
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        edit = QLineEdit(container)
        edit.setPlaceholderText(placeholder)
        edit.setClearButtonEnabled(True)
        set_accessible(edit, acc_name, acc_desc)
        btn = QPushButton(browse_text, container)
        btn.setIcon(load_icon("folder"))
        set_accessible(btn, browse_name, f"Choose a path for {acc_name}.")
        row.addWidget(edit, stretch=1)
        row.addWidget(btn)
        return container, edit, btn

    def _apply_defaults_to_widgets(self, cfg: AppConfig) -> None:
        self._populate_conflict_choices()
        self.ffmpeg_edit.setText(cfg.binaries.ffmpeg_path or "")
        self.ffprobe_edit.setText(cfg.binaries.ffprobe_path or "")
        self.quality_combo.setCurrentText(cfg.audio.quality)
        self.copy_mp3_checkbox.setChecked(cfg.audio.copy_mp3)
        self.output_dir_edit.setText(cfg.output.output_dir or "")
        self.flat_checkbox.setChecked(cfg.output.flat)
        idx = self.conflict_combo.findData(cfg.output.on_conflict.value)
        if idx >= 0:
            self.conflict_combo.setCurrentIndex(idx)
        jobs = cfg.execution.jobs if cfg.execution.jobs >= 1 else default_jobs()
        self._set_jobs_spin_value(jobs)
        self.contact_edit.setText(cfg.musicbrainz.contact or "")
        self.user_agent_edit.setText(cfg.musicbrainz.user_agent_override or "")
        self._baseline_values = {
            "binaries.ffmpeg_path": _normalize_optional_str(cfg.binaries.ffmpeg_path),
            "binaries.ffprobe_path": _normalize_optional_str(cfg.binaries.ffprobe_path),
            "audio.quality": cfg.audio.quality.strip(),
            "audio.copy_mp3": bool(cfg.audio.copy_mp3),
            "output.output_dir": _normalize_optional_str(cfg.output.output_dir),
            "output.flat": bool(cfg.output.flat),
            "output.on_conflict": cfg.output.on_conflict,
            "execution.jobs": int(jobs),
            "musicbrainz.contact": _normalize_optional_str(cfg.musicbrainz.contact),
        }

    def _load_from_raw_toml(self, target_path: Path) -> None:
        """Populate widgets from raw ``config.toml`` when ``controller.config`` is ``None``."""
        defaults = AppConfig(config_path=target_path)
        self._apply_defaults_to_widgets(defaults)
        self._clear_invalid_field_labels()
        self._loaded_from_raw_toml = False
        self._raw_validation_warning = None
        self._config_load_blocked = False
        self.save_button.setEnabled(True)

        try:
            raw_text = target_path.read_text(encoding="utf-8")
            parsed = tomllib.loads(raw_text)
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            self._config_load_blocked = True
            self.save_button.setEnabled(False)
            self.status_banner.set_message(
                "status_error",
                f"Cannot save settings until {target_path.name} is fixed ({exc}). "
                "Click Open config file to edit the file manually.",
                role_label="Configuration parse error",
            )
            return

        self._loaded_from_raw_toml = True
        baseline = dict(self._baseline_values)
        editable_issues: dict[str, str] = {}
        uneditable_issues: list[tuple[str, str]] = []

        # Flag unrecognized top-level keys/sections and non-table known sections.
        for top_key, top_val in parsed.items():
            if top_key not in _KNOWN_SECTION_KEYS:
                label = f"[{top_key}]" if isinstance(top_val, dict) else str(top_key)
                uneditable_issues.append(
                    (label, f"Unrecognized config section or key '{top_key}'.")
                )
            elif not isinstance(top_val, dict):
                uneditable_issues.append(
                    (f"[{top_key}]", f"Config section '[{top_key}]' must be a TOML table.")
                )
            else:
                known_keys = _KNOWN_SECTION_KEYS[top_key]
                for sub_key in top_val:
                    if sub_key not in known_keys:
                        uneditable_issues.append(
                            (
                                f"[{top_key}] {sub_key}",
                                f"Unrecognized config key '{top_key}.{sub_key}'.",
                            )
                        )

        binaries_tbl = parsed.get("binaries")
        if isinstance(binaries_tbl, dict):
            raw_ffmpeg = binaries_tbl.get("ffmpeg")
            if raw_ffmpeg is None or (isinstance(raw_ffmpeg, str) and not raw_ffmpeg.strip()):
                raw_ffmpeg = binaries_tbl.get("ffmpeg_path")
            if isinstance(raw_ffmpeg, str):
                cleaned_ffmpeg = raw_ffmpeg.strip() or None
                self.ffmpeg_edit.setText(cleaned_ffmpeg or "")
                baseline["binaries.ffmpeg_path"] = cleaned_ffmpeg
            elif raw_ffmpeg is not None:
                self.ffmpeg_edit.setText("")
                baseline["binaries.ffmpeg_path"] = _RawInvalidValue(raw_ffmpeg)
                editable_issues["binaries.ffmpeg_path"] = (
                    f"Invalid value {raw_ffmpeg!r} in {target_path.name}; "
                    "Save will replace it (blank clears it)."
                )

            raw_ffprobe = binaries_tbl.get("ffprobe")
            if raw_ffprobe is None or (isinstance(raw_ffprobe, str) and not raw_ffprobe.strip()):
                raw_ffprobe = binaries_tbl.get("ffprobe_path")
            if isinstance(raw_ffprobe, str):
                cleaned_ffprobe = raw_ffprobe.strip() or None
                self.ffprobe_edit.setText(cleaned_ffprobe or "")
                baseline["binaries.ffprobe_path"] = cleaned_ffprobe
            elif raw_ffprobe is not None:
                self.ffprobe_edit.setText("")
                baseline["binaries.ffprobe_path"] = _RawInvalidValue(raw_ffprobe)
                editable_issues["binaries.ffprobe_path"] = (
                    f"Invalid value {raw_ffprobe!r} in {target_path.name}; "
                    "Save will replace it (blank clears it)."
                )

        audio_tbl = parsed.get("audio")
        if isinstance(audio_tbl, dict):
            if "quality" in audio_tbl:
                raw_quality = audio_tbl["quality"]
                if isinstance(raw_quality, str):
                    cleaned_quality = raw_quality.strip()
                    self.quality_combo.setCurrentText(cleaned_quality)
                    baseline["audio.quality"] = cleaned_quality
                    if not cleaned_quality:
                        editable_issues["audio.quality"] = (
                            "Quality cannot be empty; pick a valid CBR bitrate or v0 through v9."
                        )
                    else:
                        try:
                            parse_quality(cleaned_quality)
                        except ConfigError as exc:
                            editable_issues["audio.quality"] = str(exc)
                else:
                    baseline["audio.quality"] = _RawInvalidValue(raw_quality)
                    editable_issues["audio.quality"] = (
                        f"Invalid value {raw_quality!r} in {target_path.name}; "
                        f"Save will replace it with {defaults.audio.quality!r} "
                        "(or your chosen value)."
                    )
            if "copy_mp3" in audio_tbl:
                raw_copy = audio_tbl["copy_mp3"]
                if isinstance(raw_copy, bool):
                    self.copy_mp3_checkbox.setChecked(raw_copy)
                    baseline["audio.copy_mp3"] = raw_copy
                else:
                    baseline["audio.copy_mp3"] = _RawInvalidValue(raw_copy)
                    editable_issues["audio.copy_mp3"] = (
                        f"Invalid value {raw_copy!r} in {target_path.name}; "
                        "Save will replace it with the shown setting."
                    )
            if "loudnorm_mode" in audio_tbl:
                raw_mode = audio_tbl["loudnorm_mode"]
                if not isinstance(raw_mode, str) or raw_mode.strip() not in ("linear", "dynamic"):
                    uneditable_issues.append(
                        (
                            "[audio] loudnorm_mode",
                            "Config key 'audio.loudnorm_mode' must be 'linear' or 'dynamic'.",
                        )
                    )
            for ln_key, lo, hi in _LOUDNORM_NUMERIC_RANGES:
                if ln_key in audio_tbl:
                    ln_val = audio_tbl[ln_key]
                    if not _is_finite_number(ln_val):
                        uneditable_issues.append(
                            (f"[audio] {ln_key}", f"Config key 'audio.{ln_key}' must be a number.")
                        )
                    elif not (lo <= float(ln_val) <= hi):
                        uneditable_issues.append(
                            (
                                f"[audio] {ln_key}",
                                f"Config key 'audio.{ln_key}' must be between {lo} and {hi}.",
                            )
                        )

        output_tbl = parsed.get("output")
        if isinstance(output_tbl, dict):
            if "output_dir" in output_tbl:
                raw_out = output_tbl["output_dir"]
                if isinstance(raw_out, str):
                    cleaned_out = raw_out.strip() or None
                    self.output_dir_edit.setText(cleaned_out or "")
                    baseline["output.output_dir"] = cleaned_out
                elif raw_out is not None:
                    self.output_dir_edit.setText("")
                    baseline["output.output_dir"] = _RawInvalidValue(raw_out)
                    editable_issues["output.output_dir"] = (
                        f"Invalid value {raw_out!r} in {target_path.name}; "
                        "Save will replace it (blank clears it)."
                    )
            if "flat" in output_tbl:
                raw_flat = output_tbl["flat"]
                if isinstance(raw_flat, bool):
                    self.flat_checkbox.setChecked(raw_flat)
                    baseline["output.flat"] = raw_flat
                else:
                    baseline["output.flat"] = _RawInvalidValue(raw_flat)
                    editable_issues["output.flat"] = (
                        f"Invalid value {raw_flat!r} in {target_path.name}; "
                        "Save will replace it with the shown setting."
                    )
            if "on_conflict" in output_tbl:
                raw_conflict = output_tbl["on_conflict"]
                if isinstance(raw_conflict, str):
                    cleaned_conflict = raw_conflict.strip()
                    try:
                        policy = ConflictPolicy(cleaned_conflict)
                    except ValueError:
                        marker = _RawInvalidValue(raw_conflict)
                        self.conflict_combo.addItem(f"{cleaned_conflict!r} (invalid)", marker)
                        self.conflict_combo.setCurrentIndex(self.conflict_combo.count() - 1)
                        baseline["output.on_conflict"] = marker
                        editable_issues["output.on_conflict"] = (
                            f"Invalid value {raw_conflict!r} in {target_path.name}; "
                            "choose skip, overwrite, or rename."
                        )
                    else:
                        idx = self.conflict_combo.findData(policy.value)
                        if idx >= 0:
                            self.conflict_combo.setCurrentIndex(idx)
                        baseline["output.on_conflict"] = policy
                else:
                    marker = _RawInvalidValue(raw_conflict)
                    self.conflict_combo.addItem(f"{raw_conflict!r} (invalid)", marker)
                    self.conflict_combo.setCurrentIndex(self.conflict_combo.count() - 1)
                    baseline["output.on_conflict"] = marker
                    editable_issues["output.on_conflict"] = (
                        f"Invalid value {raw_conflict!r} in {target_path.name}; "
                        "choose skip, overwrite, or rename."
                    )

        exec_tbl = parsed.get("execution")
        if isinstance(exec_tbl, dict) and "jobs" in exec_tbl:
            raw_jobs = exec_tbl["jobs"]
            if type(raw_jobs) is int and raw_jobs >= 1:
                self._set_jobs_spin_value(raw_jobs)
                baseline["execution.jobs"] = raw_jobs
            else:
                baseline["execution.jobs"] = _RawInvalidValue(raw_jobs)
                editable_issues["execution.jobs"] = (
                    f"Invalid value {raw_jobs!r} in {target_path.name}; "
                    f"Save will replace it with {self.jobs_spin.value()} "
                    "(or your chosen count)."
                )

        mb_tbl = parsed.get("musicbrainz")
        if isinstance(mb_tbl, dict):
            if "contact" in mb_tbl:
                raw_contact = mb_tbl["contact"]
                if isinstance(raw_contact, str):
                    cleaned_contact = raw_contact.strip() or None
                    self.contact_edit.setText(cleaned_contact or "")
                    baseline["musicbrainz.contact"] = cleaned_contact
                    if cleaned_contact is not None:
                        try:
                            validate_mb_contact(cleaned_contact)
                        except ConfigError as exc:
                            editable_issues["musicbrainz.contact"] = str(exc)
                elif raw_contact is not None:
                    self.contact_edit.setText("")
                    baseline["musicbrainz.contact"] = _RawInvalidValue(raw_contact)
                    editable_issues["musicbrainz.contact"] = (
                        f"Invalid value {raw_contact!r} in {target_path.name}; "
                        "Save will replace it (blank clears it)."
                    )
            if "user_agent" in mb_tbl:
                raw_ua = mb_tbl["user_agent"]
                if isinstance(raw_ua, str):
                    cleaned_ua = raw_ua.strip()
                    self.user_agent_edit.setText(cleaned_ua)
                    if cleaned_ua:
                        try:
                            validate_mb_user_agent(cleaned_ua)
                        except ConfigError as exc:
                            uneditable_issues.append(("[musicbrainz] user_agent", str(exc)))
                            self._set_invalid_field_label(
                                "musicbrainz.user_agent_override",
                                "Invalid [musicbrainz] user_agent in "
                                f"{target_path.name}; click Open config file to edit it manually.",
                            )
                elif raw_ua is not None:
                    uneditable_issues.append(
                        (
                            "[musicbrainz] user_agent",
                            "Config key 'musicbrainz.user_agent' must be a string.",
                        )
                    )
                    self._set_invalid_field_label(
                        "musicbrainz.user_agent_override",
                        f"Invalid [musicbrainz] user_agent value ({raw_ua!r}) in "
                        f"{target_path.name}; click Open config file to edit it manually.",
                    )

        match_tbl = parsed.get("matching")
        if isinstance(match_tbl, dict):
            for attr in _MATCHING_KEYS:
                if attr in match_tbl:
                    val = match_tbl[attr]
                    if not _is_finite_number(val):
                        uneditable_issues.append(
                            (
                                f"[matching] {attr}",
                                f"Config key 'matching.{attr}' must be a number.",
                            )
                        )
                    elif not (0.0 <= float(val) <= 1.0):
                        uneditable_issues.append(
                            (
                                f"[matching] {attr}",
                                f"Config key 'matching.{attr}' must be between 0.0 and 1.0.",
                            )
                        )

        self._baseline_values = baseline
        self._editable_config_issues = editable_issues
        self._uneditable_config_issues = uneditable_issues

        for attr_path, message in editable_issues.items():
            self._set_invalid_field_label(attr_path, message)

        # Inspect load_config validation for the exact ConfigError text and any unexpected issues.
        try:
            load_config(config_path=target_path, env={})
        except ConfigError as exc:
            self._raw_load_error_text = str(exc)

        warning = self._compose_raw_toml_warning(target_path.name)
        if warning:
            self._raw_validation_warning = warning
            self.status_banner.set_message(
                "status_warn",
                warning,
                role_label="Configuration warning",
            )

    def load_from_config(
        self,
        config: AppConfig | None,
        *,
        config_path: Path | None = None,
    ) -> None:
        """Populate editable setting widgets and capture a baseline snapshot for diffing.

        When ``config`` is ``None`` and ``config.toml`` already exists on disk,
        reads the raw TOML file so valid settings (such as ``contact``,
        ``ffmpeg``, and ``output_dir``) are preserved while letting the user fix
        invalid values. Disables Save when the file cannot be read or parsed as
        valid TOML syntax.
        """
        self._pre_override_texts.clear()
        self._load_from_config_impl(config, config_path=config_path)

    def _load_from_config_impl(
        self,
        config: AppConfig | None,
        *,
        config_path: Path | None = None,
    ) -> None:
        if config is not None:
            had_config_notice = (
                self._config_load_blocked or self._raw_validation_warning is not None
            )
            self._clear_invalid_field_labels()
            self._config_path = config.config_path
            self.config_path_edit.setText(str(config.config_path))
            self._loaded_from_raw_toml = False
            self._raw_validation_warning = None
            self._config_load_blocked = False
            self.save_button.setEnabled(True)
            if had_config_notice:
                self.status_banner.clear_message()
            self._apply_defaults_to_widgets(config)
            return

        if config_path is not None:
            target_path = config_path
        elif os.environ.get("AFV_CONFIG_PATH", "").strip():
            target_path = default_config_path()
        else:
            target_path = self._config_path

        self._config_path = target_path
        self.config_path_edit.setText(str(target_path))

        if target_path.is_file():
            self._load_from_raw_toml(target_path)
            return

        had_config_notice = self._config_load_blocked or self._raw_validation_warning is not None
        self._clear_invalid_field_labels()
        self._loaded_from_raw_toml = False
        self._raw_validation_warning = None
        self._config_load_blocked = False
        self.save_button.setEnabled(True)
        if had_config_notice:
            self.status_banner.clear_message()
        self._apply_defaults_to_widgets(AppConfig(config_path=target_path))

    def _current_ui_preferences(self) -> tuple[str, bool]:
        level = self.log_level_combo.currentText().strip().upper()
        if level not in LOG_LEVEL_CHOICES:
            level = "INFO"
        return (level, self.startup_offline_checkbox.isChecked())

    def _read_saved_ui_preferences(self) -> tuple[str, bool]:
        settings = open_ui_settings(self)
        level_raw = str(settings.value(LOG_LEVEL_KEY, "INFO") or "INFO").strip().upper()
        if level_raw not in LOG_LEVEL_CHOICES:
            level_raw = "INFO"
        offline_val = read_bool_setting(settings, STARTUP_OFFLINE_KEY, default=False)
        return (level_raw, offline_val)

    def _write_ui_preferences(self, level: str, offline: bool) -> None:
        settings = open_ui_settings(self)
        settings.setValue(LOG_LEVEL_KEY, level)
        settings.setValue(STARTUP_OFFLINE_KEY, offline)
        settings.sync()

    def _capture_ui_rollback_snapshot(self) -> tuple[bool, object, bool, object]:
        settings = open_ui_settings(self)
        has_level = settings.contains(LOG_LEVEL_KEY)
        prev_level = settings.value(LOG_LEVEL_KEY) if has_level else None
        has_offline = settings.contains(STARTUP_OFFLINE_KEY)
        prev_offline = settings.value(STARTUP_OFFLINE_KEY) if has_offline else None
        return (has_level, prev_level, has_offline, prev_offline)

    def _restore_ui_rollback_snapshot(
        self,
        snapshot: tuple[bool, object, bool, object],
    ) -> None:
        has_level, prev_level, has_offline, prev_offline = snapshot
        settings = open_ui_settings(self)
        if has_level:
            settings.setValue(LOG_LEVEL_KEY, prev_level)
        else:
            settings.remove(LOG_LEVEL_KEY)
        if has_offline:
            settings.setValue(STARTUP_OFFLINE_KEY, prev_offline)
        else:
            settings.remove(STARTUP_OFFLINE_KEY)
        settings.sync()

    def load_ui_preferences(self) -> None:
        """Load ``log/level`` and ``startup/offline`` from the isolated ``QSettings`` store."""
        level_raw, offline_val = self._read_saved_ui_preferences()
        idx = self.log_level_combo.findText(level_raw)
        if idx >= 0:
            self.log_level_combo.setCurrentIndex(idx)
        self.startup_offline_checkbox.setChecked(offline_val)
        self._loaded_ui_preferences = (level_raw, offline_val)

    def save_ui_preferences(self) -> None:
        """Persist ``log/level`` and ``startup/offline`` to ``QSettings``."""
        level, offline = self._current_ui_preferences()
        self._write_ui_preferences(level, offline)
        self._loaded_ui_preferences = (level, offline)

    def refresh_env_overrides(self, env: Mapping[str, str] | None = None) -> None:
        """Detect active ``ENV_OVERRIDES`` and lock matching fields read-only."""
        active_env = os.environ if env is None else env
        previous_overrides = dict(self._overridden_keys)
        self._overridden_keys.clear()

        edit_map: dict[str, tuple[QLineEdit, QPushButton | None]] = {
            "binaries.ffmpeg_path": (self.ffmpeg_edit, self.browse_ffmpeg_button),
            "binaries.ffprobe_path": (self.ffprobe_edit, self.browse_ffprobe_button),
            "output.output_dir": (self.output_dir_edit, self.browse_output_button),
            "musicbrainz.contact": (self.contact_edit, None),
            "musicbrainz.user_agent_override": (self.user_agent_edit, None),
        }

        for attr_path, label in self.env_override_labels.items():
            label.setVisible(False)
            edit, button = edit_map[attr_path]
            if attr_path != "musicbrainz.user_agent_override":
                edit.setReadOnly(False)
            if button is not None:
                button.setEnabled(True)
            if attr_path in previous_overrides and attr_path in self._pre_override_texts:
                with QSignalBlocker(edit):
                    edit.setText(self._pre_override_texts.pop(attr_path))

        for env_var, attr_path in ENV_OVERRIDES.items():
            raw = active_env.get(env_var, "").strip()
            if not raw:
                continue
            self._overridden_keys[attr_path] = env_var
            edit, button = edit_map[attr_path]
            if attr_path not in self._pre_override_texts:
                self._pre_override_texts[attr_path] = edit.text()
            with QSignalBlocker(edit):
                edit.setText(raw)
            edit.setReadOnly(True)
            if button is not None:
                button.setEnabled(False)
            badge = self.env_override_labels.get(attr_path)
            if badge is not None:
                badge.set_status("status_info", f"Read-only: set by {env_var}")
                badge.setVisible(True)

        if self._loaded_from_raw_toml and not self._config_load_blocked:
            refreshed = self._compose_raw_toml_warning(self._config_path.name)
            if refreshed is not None and (
                refreshed != self._raw_validation_warning
                and (
                    self.status_banner.isHidden()
                    or self.status_banner.text == (self._raw_validation_warning or "")
                )
            ):
                self._raw_validation_warning = refreshed
                self.status_banner.set_message(
                    "status_warn",
                    refreshed,
                    role_label="Configuration warning",
                )
            elif refreshed is not None:
                self._raw_validation_warning = refreshed

    def overridden_settings(self) -> dict[str, str]:
        """Return mapping of ``AppConfig`` attribute paths currently overridden by env vars."""
        return dict(self._overridden_keys)

    def _field_changed(self, key: str, current: object) -> bool:
        base = self._baseline_values.get(key)
        if isinstance(base, _RawInvalidValue):
            return True
        return not _setting_values_equal(current, base)

    def collect_updates(self, *, include_all: bool = False) -> dict[str, object]:
        """Collect and validate changed settings to send to ``controller.save_settings``.

        By default (``include_all=False``), only keys whose values differ from
        the load-time baseline snapshot are returned so saving never pins
        unrelated defaults (such as machine-dependent ``execution.jobs``) or
        overwrites untouched keys inside ``config.toml``. Keys overridden by an
        environment variable in ``ENV_OVERRIDES`` are always omitted. Raises
        ``ConfigError`` when a collected value is invalid or when the existing
        configuration file has a TOML syntax error.
        """
        if self._config_load_blocked:
            raise ConfigError(
                f"Cannot save while {self._config_path.name} cannot be parsed. "
                "Click Open config file to repair the file syntax first."
            )

        updates: dict[str, object] = {}

        if "binaries.ffmpeg_path" not in self._overridden_keys:
            ffmpeg_val = self.ffmpeg_edit.text().strip() or None
            if include_all or self._field_changed("binaries.ffmpeg_path", ffmpeg_val):
                updates["binaries.ffmpeg_path"] = ffmpeg_val

        if "binaries.ffprobe_path" not in self._overridden_keys:
            ffprobe_val = self.ffprobe_edit.text().strip() or None
            if include_all or self._field_changed("binaries.ffprobe_path", ffprobe_val):
                updates["binaries.ffprobe_path"] = ffprobe_val

        quality = self.quality_combo.currentText().strip()
        if include_all or self._field_changed("audio.quality", quality):
            parse_quality(quality)
            updates["audio.quality"] = quality

        copy_mp3 = self.copy_mp3_checkbox.isChecked()
        if include_all or self._field_changed("audio.copy_mp3", copy_mp3):
            updates["audio.copy_mp3"] = copy_mp3

        if "output.output_dir" not in self._overridden_keys:
            out_val = self.output_dir_edit.text().strip() or None
            if include_all or self._field_changed("output.output_dir", out_val):
                updates["output.output_dir"] = out_val

        flat_val = self.flat_checkbox.isChecked()
        if include_all or self._field_changed("output.flat", flat_val):
            updates["output.flat"] = flat_val

        conflict_data = self.conflict_combo.currentData()
        if isinstance(conflict_data, _RawInvalidValue):
            if include_all:
                raise ConfigError(
                    "Config key 'output.on_conflict' must be 'skip', 'overwrite', or 'rename'."
                )
        else:
            try:
                policy = ConflictPolicy(str(conflict_data))
            except ValueError:
                policy = ConflictPolicy.SKIP
            if include_all or not _setting_values_equal(
                policy, self._baseline_values.get("output.on_conflict")
            ):
                updates["output.on_conflict"] = policy

        jobs_val = self.jobs_spin.value()
        if include_all or self._field_changed("execution.jobs", jobs_val):
            updates["execution.jobs"] = validate_jobs(jobs_val)

        if "musicbrainz.contact" not in self._overridden_keys:
            raw_contact = self.contact_edit.text().strip() or None
            if include_all or self._field_changed("musicbrainz.contact", raw_contact):
                if raw_contact is not None:
                    updates["musicbrainz.contact"] = validate_mb_contact(raw_contact)
                else:
                    updates["musicbrainz.contact"] = None

        # Verify every collected key belongs to EDITABLE_SETTINGS.
        for key in updates:
            if key not in EDITABLE_SETTINGS:
                raise ValueError(f"Unexpected settings key {key!r}.")

        return updates

    def show_error(self, message: str) -> None:
        """Display an inline validation or persistence error message."""
        self.status_banner.set_message(
            "status_error",
            message.strip() or "Failed to save settings.",
            role_label="Settings error",
        )

    @Slot()
    def refresh_cache_stats(self) -> None:
        """Put the cache stats banner into a loading state and ask the controller for stats."""
        self._cache_stats_pending = True
        self._cache_notice_is_busy = False
        self.cache_status.set_status("status_pending", "Loading cache stats…")
        if not self._cache_clear_pending:
            self.cache_notice_label.setText("")
        self._controller.request_cache_stats()

    def showEvent(self, event: QShowEvent) -> None:
        """Refresh environment variable override badges whenever Settings becomes visible."""
        super().showEvent(event)
        self.refresh_env_overrides()

    @Slot()
    def _on_open_config(self) -> None:
        if self._config_path.is_file():
            if not open_local_path(self._config_path):
                self.status_banner.set_message(
                    "status_info",
                    f"Could not open configuration file: {self._config_path}.",
                    role_label="Configuration notice",
                )
            return
        parent = self._config_path.parent
        if parent.is_dir():
            if not open_local_path(parent):
                self.status_banner.set_message(
                    "status_info",
                    f"Could not open configuration folder: {parent}.",
                    role_label="Configuration notice",
                )
            return
        self.status_banner.set_message(
            "status_info",
            f"Configuration file has not been created yet: {self._config_path}.",
            role_label="Configuration notice",
        )

    @Slot()
    def _on_browse_ffmpeg(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select ffmpeg executable",
            self.ffmpeg_edit.text().strip(),
            "All files (*)",
        )
        if selected:
            self.ffmpeg_edit.setText(selected)

    @Slot()
    def _on_browse_ffprobe(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select ffprobe executable",
            self.ffprobe_edit.text().strip(),
            "All files (*)",
        )
        if selected:
            self.ffprobe_edit.setText(selected)

    @Slot()
    def _on_browse_output(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select default output folder",
            self.output_dir_edit.text().strip(),
        )
        if selected:
            self.output_dir_edit.setText(selected)

    @Slot()
    def _on_open_log_folder(self) -> None:
        log_dir = log_directory()
        if not open_local_path(log_dir):
            text = (
                f"Log folder does not exist yet: {log_dir}."
                if not _path_exists(log_dir)
                else f"Could not open log folder: {log_dir}."
            )
            self.status_banner.set_message(
                "status_info",
                text,
                role_label="Log folder notice",
            )

    @Slot()
    def _on_clear_expired(self) -> None:
        self._cache_clear_pending = True
        self._cache_notice_is_busy = False
        self.clear_expired_button.setEnabled(False)
        self.clear_cache_button.setEnabled(False)
        self.cache_notice_label.setText("Clearing expired cache entries…")
        self._controller.clear_cache(expired_only=True)

    @Slot()
    def _on_clear_cache(self) -> None:
        self._cache_clear_pending = True
        self._cache_notice_is_busy = False
        self.clear_expired_button.setEnabled(False)
        self.clear_cache_button.setEnabled(False)
        self.cache_notice_label.setText("Clearing all cache entries…")
        self._controller.clear_cache(expired_only=False)

    @Slot()
    def _on_save_clicked(self) -> None:
        if self._config_load_blocked:
            return
        try:
            updates = self.collect_updates()
        except ConfigError as exc:
            self._pending_ui_preferences = None
            self._ui_rollback_snapshot = None
            self.show_error(str(exc))
            return

        self.status_banner.clear_message()
        desired_level, desired_offline = self._current_ui_preferences()

        if not updates:
            # No config.toml edits pending: write QSettings immediately so
            # RunController.save_settings({}) picks up offline_default and
            # log/level without modifying config.toml; keep a rollback snapshot
            # in case the controller rejects the save (e.g., busy or error).
            rollback = self._capture_ui_rollback_snapshot()
            self._pending_ui_preferences = None
            self._ui_rollback_snapshot = rollback
            self._write_ui_preferences(desired_level, desired_offline)
            try:
                self._controller.save_settings({})
            except (ConfigError, ValueError, OSError) as exc:
                self._restore_ui_rollback_snapshot(rollback)
                self._ui_rollback_snapshot = None
                self.show_error(str(exc))
            return

        # Hold UI preferences until the controller confirms config.toml succeeded
        # via settings_saved, so a failed file save never mutates QSettings.
        self._ui_rollback_snapshot = None
        self._pending_ui_preferences = (desired_level, desired_offline)
        try:
            self._controller.save_settings(updates)
        except (ConfigError, ValueError, OSError) as exc:
            self._pending_ui_preferences = None
            self.show_error(str(exc))
            return

    @Slot(object)
    def _on_cache_stats_ready(self, stats: list[NamespaceStats]) -> None:
        self._cache_stats_pending = False
        if not self._cache_clear_pending:
            self.clear_expired_button.setEnabled(True)
            self.clear_cache_button.setEnabled(True)
            if self._cache_notice_is_busy:
                self._cache_notice_is_busy = False
                self.cache_notice_label.setText("")
        self.cache_table.setRowCount(0)
        if not stats:
            self.cache_status.set_status("status_info", "Cache is empty (no cache file found).")
            return

        total_rows = 0
        total_expired = 0
        total_bytes = 0
        self.cache_table.setRowCount(len(stats))
        for row_idx, entry in enumerate(stats):
            total_rows += entry.rows
            total_expired += entry.expired
            total_bytes += entry.bytes
            self.cache_table.setItem(row_idx, 0, QTableWidgetItem(entry.namespace))
            self.cache_table.setItem(row_idx, 1, QTableWidgetItem(str(entry.rows)))
            self.cache_table.setItem(row_idx, 2, QTableWidgetItem(str(entry.expired)))
            self.cache_table.setItem(row_idx, 3, QTableWidgetItem(format_byte_size(entry.bytes)))
            self.cache_table.setItem(
                row_idx,
                4,
                QTableWidgetItem(format_ttl_duration(entry.ttl_seconds)),
            )

        self.cache_status.set_status(
            "status_ok",
            f"Total: {total_rows} entries ({total_expired} expired), "
            f"{format_byte_size(total_bytes)}.",
        )

    @Slot(int)
    def _on_cache_cleared(self, removed: int) -> None:
        self._cache_clear_pending = False
        self._cache_notice_is_busy = False
        self.clear_expired_button.setEnabled(True)
        self.clear_cache_button.setEnabled(True)
        count = max(0, int(removed))
        noun = "entry" if count == 1 else "entries"
        self.cache_notice_label.setText(f"Cleared {count} cache {noun}.")
        self.cache_status.set_status("status_info", f"Cleared {count} cache {noun}.")

    @Slot(object)
    def _on_settings_saved(self, config: AppConfig | None) -> None:
        self._ui_rollback_snapshot = None
        pending = self._pending_ui_preferences
        self._pending_ui_preferences = None
        if pending is not None:
            level, offline = pending
            self._write_ui_preferences(level, offline)
            self._loaded_ui_preferences = (level, offline)
        else:
            self._loaded_ui_preferences = self._current_ui_preferences()

        self.load_from_config(config)
        self.refresh_env_overrides()

        if config is None and self._raw_validation_warning:
            self.status_banner.set_message(
                "status_warn",
                f"Preferences saved. {self._raw_validation_warning}",
                role_label="Settings warning",
            )
        else:
            self.status_banner.set_message(
                "status_ok",
                "Settings saved.",
                role_label="Settings saved",
            )

    @Slot(str, str)
    def _on_controller_error(self, title: str, details: str) -> None:
        heading = title.strip()
        body = details.strip()
        text = f"{heading}: {body}" if heading and body else heading or body or "Operation failed."

        # Discard or roll back any unconfirmed UI preferences so a failed save
        # never mutates QSettings.
        self._pending_ui_preferences = None
        if self._ui_rollback_snapshot is not None:
            rollback = self._ui_rollback_snapshot
            self._ui_rollback_snapshot = None
            self._restore_ui_rollback_snapshot(rollback)

        was_cache_pending = self._cache_clear_pending
        was_stats_pending = self._cache_stats_pending
        self._cache_clear_pending = False
        self._cache_stats_pending = False
        self.clear_expired_button.setEnabled(True)
        self.clear_cache_button.setEnabled(True)

        if _is_cache_busy_error(heading, body):
            # Surface transient busy notes neutrally inside the cache section
            # instead of showing a red Settings error banner or clearing any
            # config.toml validation warning above.
            self._cache_notice_is_busy = True
            self.cache_notice_label.setText(text)
            self.cache_status.set_status("status_info", text)
            return

        self._cache_notice_is_busy = False
        if was_cache_pending:
            self.cache_notice_label.setText(text)
        if was_cache_pending or (was_stats_pending and _is_cache_related_error(heading, body)):
            self.cache_status.set_status("status_warn", text)

        self.show_error(text)
