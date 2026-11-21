"""First-run Setup screen for MusicBrainz contact, FFmpeg check, and offline opt-in."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from audio_from_video.binaries import format_install_instructions
from audio_from_video.config import ConfigError, validate_mb_contact
from audio_from_video.gui.controller import ControllerState
from audio_from_video.gui.resources import load_icon
from audio_from_video.gui.screens._common import (
    IconStatusBanner,
    SectionCard,
    StatusLine,
    ThemedSvgLabel,
    apply_primary_button_role,
    apply_secondary_button_role,
    make_divider,
    make_heading_label,
    make_mono_font,
    make_muted_label,
    set_accessible,
)
from audio_from_video.gui.theme import set_widget_role

if TYPE_CHECKING:
    from audio_from_video.gui.controller import BinariesStatus, RunController

_APPLE_HIDE_MY_EMAIL_URL = "https://support.apple.com/en-us/105078"
_SIMPLELOGIN_URL = "https://simplelogin.io/"
_DUCK_ADDRESS_URL = "https://duckduckgo.com/email/"

_WRITE_ERROR_HINTS: tuple[str, ...] = (
    "permission",
    "read-only",
    "readonly",
    "write",
    "saving",
    "denied",
    "disk",
    "errno",
    "oserror",
    "unable to save",
    "failed to save",
)


def _needs_env_fallback_hint(message: str) -> bool:
    """Return True when an error looks like a config-file write failure."""
    lowered = message.casefold()
    if "afv_mb_contact" in lowered:
        return False
    return any(hint in lowered for hint in _WRITE_ERROR_HINTS)


class SetupScreen(QWidget):
    """Setup screen widget (listens to ``binaries_status`` and ``setup_error``)."""

    def __init__(
        self,
        controller: RunController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._last_binaries_status: BinariesStatus | None = None

        set_accessible(
            self,
            "Setup screen",
            "Configure your MusicBrainz contact, verify FFmpeg, or choose offline mode.",
        )

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        set_accessible(scroll, "Setup content", "Scrollable container for setup controls.")

        body = QWidget(scroll)
        body.setObjectName("setup_scroll_body")
        body_row = QHBoxLayout(body)
        body_row.setContentsMargins(24, 14, 24, 14)
        body_row.setSpacing(0)

        column = QWidget(body)
        column.setObjectName("setup_content_column")
        column.setMaximumWidth(760)
        column.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        body_row.addStretch(0)
        body_row.addWidget(column, stretch=1)
        body_row.addStretch(0)

        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Top hero header bar with app emblem and Settings entry point.
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 2)
        header_row.setSpacing(12)

        emblem = ThemedSvgLabel("app_icon", size=34, parent=column)
        emblem.setObjectName("setup_app_icon")
        header_row.addWidget(emblem, alignment=Qt.AlignmentFlag.AlignTop)

        title_box = QVBoxLayout()
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(3)
        self.title_label = make_heading_label(
            "Welcome to Audio from Video",
            scale=1.45,
            parent=column,
        )
        self.title_label.setObjectName("setup_title_label")
        subtitle = make_muted_label(
            "Check your FFmpeg installation and set a MusicBrainz contact for online lookups, "
            "or run offline only.",
            parent=column,
        )
        subtitle.setObjectName("setup_subtitle_label")
        title_box.addWidget(self.title_label)
        title_box.addWidget(subtitle)
        header_row.addLayout(title_box, stretch=1)

        self.settings_button = QPushButton("Se&ttings…", column)
        self.settings_button.setObjectName("setup_settings_button")
        apply_secondary_button_role(self.settings_button, icon_name="settings")
        self.settings_button.setToolTip(
            "Open application settings for binaries, audio defaults, cache, and logging."
        )
        set_accessible(
            self.settings_button,
            "Settings",
            "Open the Settings dialog to edit defaults, environment overrides, and cache.",
        )
        header_row.addWidget(self.settings_button, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header_row)

        # Inline setup error banner (hidden until controller emits setup_error).
        self.error_banner = IconStatusBanner(column)
        self.error_banner.setObjectName("setup_error_banner")
        layout.addWidget(self.error_banner)

        # MusicBrainz contact configuration card.
        contact_card = SectionCard(
            "MusicBrainz Contact",
            subtitle="Optional · for online metadata and artwork lookups",
            margins=(18, 12, 18, 12),
            spacing=8,
            parent=column,
        )
        contact_card.setObjectName("setup_contact_card")

        self._contact_badge = QLabel("Optional", contact_card)
        self._contact_badge.setObjectName("setup_contact_badge")
        self._contact_badge.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._contact_badge.setTextFormat(Qt.TextFormat.PlainText)
        self._contact_badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._contact_badge.setProperty("afvSurface", "chip")
        self._contact_badge.setProperty("afvTone", "info")
        contact_card.add_header_accessory(self._contact_badge)

        contact_info = make_muted_label(
            "MusicBrainz requires a reachable email address or public repository URL to identify "
            "the application in the User-Agent request header. It is stored locally in config.toml "
            "and sent only to MusicBrainz and the Cover Art Archive.",
            scale=0.95,
            parent=contact_card,
        )
        contact_card.body_layout.addWidget(contact_info)

        self.relay_links_label = QLabel(
            f"Recommended anonymous email relays: "
            f'<a href="{_APPLE_HIDE_MY_EMAIL_URL}">Apple Hide My Email</a>, '
            f'<a href="{_SIMPLELOGIN_URL}">SimpleLogin</a>, and '
            f'<a href="{_DUCK_ADDRESS_URL}">Duck Address</a>.',
            contact_card,
        )
        self.relay_links_label.setObjectName("setup_relay_links_label")
        self.relay_links_label.setTextFormat(Qt.TextFormat.RichText)
        self.relay_links_label.setOpenExternalLinks(True)
        self.relay_links_label.setWordWrap(True)
        self.relay_links_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        set_accessible(
            self.relay_links_label,
            "Anonymous email relay links",
            "External links to Apple Hide My Email, SimpleLogin, and Duck Address.",
        )
        contact_card.body_layout.addWidget(self.relay_links_label)

        contact_field_col = QVBoxLayout()
        contact_field_col.setContentsMargins(0, 2, 0, 0)
        contact_field_col.setSpacing(6)

        contact_label = QLabel("&Contact (email relay or public URL):", contact_card)
        self.contact_edit = QLineEdit(contact_card)
        self.contact_edit.setObjectName("setup_contact_edit")
        self.contact_edit.setPlaceholderText("alias@relay-provider or https://...")
        self.contact_edit.setClearButtonEnabled(True)
        self.contact_edit.setToolTip(
            "Enter a reachable anonymous email relay address or a public repository URL "
            "for MusicBrainz lookups."
        )
        contact_label.setBuddy(self.contact_edit)
        set_accessible(
            self.contact_edit,
            "MusicBrainz contact",
            "Reachable email relay or public HTTP/HTTPS repository URL for MusicBrainz.",
        )
        contact_field_col.addWidget(contact_label)
        contact_field_col.addWidget(self.contact_edit)

        self.contact_status = StatusLine(
            "status_info",
            "Enter a reachable email relay or public repository URL to enable saving.",
            selectable=True,
            parent=contact_card,
        )
        self.contact_status.setObjectName("setup_contact_status")
        contact_field_col.addWidget(self.contact_status)
        contact_card.body_layout.addLayout(contact_field_col)

        contact_card.body_layout.addWidget(make_divider(contact_card))

        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 2, 0, 0)
        actions_row.setSpacing(10)
        self.save_contact_button = QPushButton("&Save contact", contact_card)
        self.save_contact_button.setObjectName("setup_save_contact_button")
        apply_primary_button_role(self.save_contact_button, icon_name="check_mark")
        self._save_icon_active = self.save_contact_button.icon()
        self._save_icon_disabled = load_icon("check_mark")
        self.save_contact_button.setEnabled(False)
        self.save_contact_button.setIcon(self._save_icon_disabled)
        self.save_contact_button.setDefault(True)
        self.save_contact_button.setToolTip(
            "Save the validated MusicBrainz contact to config.toml and continue."
        )
        set_accessible(
            self.save_contact_button,
            "Save contact",
            "Save the validated MusicBrainz contact to config.toml and continue.",
        )

        self.offline_button = QPushButton("Use &offline only", contact_card)
        self.offline_button.setObjectName("setup_offline_button")
        apply_secondary_button_role(self.offline_button)
        self.offline_button.setToolTip(
            "Skip online lookups and remember to start in offline mode. "
            "You can change this later in Settings."
        )
        set_accessible(
            self.offline_button,
            "Use offline only",
            "Continue without a MusicBrainz contact and start future launches in offline mode.",
        )
        actions_row.addWidget(self.save_contact_button)
        actions_row.addWidget(self.offline_button)
        actions_row.addStretch(1)
        contact_card.body_layout.addLayout(actions_row)
        layout.addWidget(contact_card)

        # FFmpeg, FFprobe, and libmp3lame availability check card.
        ffmpeg_card = SectionCard(
            "FFmpeg and FFprobe",
            subtitle="Required local tools for reading audio and encoding MP3 files",
            margins=(18, 12, 18, 12),
            spacing=8,
            parent=column,
        )
        ffmpeg_card.setObjectName("setup_ffmpeg_card")

        self._ffmpeg_badge = QLabel("Checking", ffmpeg_card)
        self._ffmpeg_badge.setObjectName("setup_ffmpeg_badge")
        self._ffmpeg_badge.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._ffmpeg_badge.setTextFormat(Qt.TextFormat.PlainText)
        self._ffmpeg_badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._ffmpeg_badge.setProperty("afvSurface", "chip")
        self._ffmpeg_badge.setProperty("afvTone", "neutral")
        ffmpeg_card.add_header_accessory(self._ffmpeg_badge)

        checklist_box = QFrame(ffmpeg_card)
        checklist_box.setObjectName("setup_binaries_checklist")
        checklist_box.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        checklist_box.setFrameShape(QFrame.Shape.StyledPanel)
        checklist_box.setFrameShadow(QFrame.Shadow.Plain)
        checklist_box.setProperty("afvSurface", "card-inset")
        checklist_layout = QVBoxLayout(checklist_box)
        checklist_layout.setContentsMargins(14, 8, 14, 8)
        checklist_layout.setSpacing(6)

        self.ffmpeg_status = StatusLine(
            "status_pending",
            "Checking FFmpeg…",
            selectable=True,
            parent=checklist_box,
        )
        self.ffmpeg_status.setObjectName("setup_ffmpeg_status")
        self.ffprobe_status = StatusLine(
            "status_pending",
            "Checking FFprobe…",
            selectable=True,
            parent=checklist_box,
        )
        self.ffprobe_status.setObjectName("setup_ffprobe_status")
        self.lame_status = StatusLine(
            "status_pending",
            "Checking libmp3lame encoder…",
            selectable=True,
            parent=checklist_box,
        )
        self.lame_status.setObjectName("setup_lame_status")
        self.lame_status.setToolTip("Required MP3 encoder (libmp3lame) bundled with FFmpeg.")

        checklist_layout.addWidget(self.ffmpeg_status)
        checklist_layout.addWidget(make_divider(checklist_box))
        checklist_layout.addWidget(self.ffprobe_status)
        checklist_layout.addWidget(make_divider(checklist_box))
        checklist_layout.addWidget(self.lame_status)
        ffmpeg_card.body_layout.addWidget(checklist_box)

        browse_row = QHBoxLayout()
        browse_row.setSpacing(8)
        self.browse_ffmpeg_button = QPushButton("&Browse for FFmpeg…", ffmpeg_card)
        self.browse_ffmpeg_button.setObjectName("setup_browse_ffmpeg_button")
        apply_secondary_button_role(self.browse_ffmpeg_button, icon_name="folder")
        self.browse_ffmpeg_button.setToolTip(
            "Select an explicit FFmpeg executable path and save it to config.toml."
        )
        set_accessible(
            self.browse_ffmpeg_button,
            "Browse for FFmpeg",
            "Select an explicit FFmpeg executable path and save it to config.toml.",
        )
        self.retry_button = QPushButton("&Retry check", ffmpeg_card)
        self.retry_button.setObjectName("setup_retry_button")
        apply_secondary_button_role(self.retry_button, icon_name="refresh")
        self.retry_button.setToolTip(
            "Reload config.toml and re-check FFmpeg, FFprobe, and libmp3lame availability."
        )
        set_accessible(
            self.retry_button,
            "Retry check",
            "Reload the configuration file and re-run the FFmpeg and FFprobe preflight check.",
        )
        browse_row.addWidget(self.browse_ffmpeg_button)
        browse_row.addWidget(self.retry_button)
        browse_row.addStretch(1)
        ffmpeg_card.body_layout.addLayout(browse_row)

        self._instructions_container = QWidget(ffmpeg_card)
        self._instructions_container.setObjectName("setup_instructions_container")
        inst_layout = QVBoxLayout(self._instructions_container)
        inst_layout.setContentsMargins(0, 2, 0, 0)
        inst_layout.setSpacing(5)

        instructions_label = make_muted_label(
            "Install instructions",
            scale=0.92,
            parent=self._instructions_container,
        )
        instructions_text = format_install_instructions()
        self.install_instructions = QPlainTextEdit(self._instructions_container)
        self.install_instructions.setObjectName("setup_install_instructions")
        self.install_instructions.setReadOnly(True)
        self.install_instructions.setProperty("afvReadonly", True)
        self.install_instructions.setFont(make_mono_font(self.install_instructions))
        self.install_instructions.setPlainText(instructions_text)
        self.install_instructions.setToolTip(
            "Platform commands for installing FFmpeg and FFprobe with MP3 encoding support."
        )
        line_count = max(2, len(instructions_text.splitlines()))
        fm = QFontMetrics(self.install_instructions.font())
        compact_height = max(62, min(92, line_count * max(15, fm.lineSpacing()) + 18))
        self.install_instructions.setMinimumHeight(compact_height)
        self.install_instructions.setMaximumHeight(compact_height)
        self.install_instructions.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        set_accessible(
            self.install_instructions,
            "FFmpeg install instructions",
            "Operating-system instructions for installing FFmpeg and FFprobe.",
        )
        inst_layout.addWidget(instructions_label)
        inst_layout.addWidget(self.install_instructions)
        ffmpeg_card.body_layout.addWidget(self._instructions_container)
        layout.addWidget(ffmpeg_card)

        layout.addStretch(1)
        scroll.setWidget(body)
        outer_layout.addWidget(scroll)

        # Pre-populate contact field when the controller already loaded one.
        cfg = self._controller.config
        if cfg is not None and cfg.musicbrainz.contact:
            self.contact_edit.setText(cfg.musicbrainz.contact)

        # Wire widget events and controller signals using bound methods only.
        self.contact_edit.textChanged.connect(self._on_contact_text_changed)
        self.contact_edit.returnPressed.connect(self._on_contact_return_pressed)
        self.save_contact_button.clicked.connect(self._on_save_contact)
        self.offline_button.clicked.connect(self._on_use_offline_only)
        self.browse_ffmpeg_button.clicked.connect(self._on_browse_ffmpeg)
        self.retry_button.clicked.connect(self._on_retry_check)

        self._controller.binaries_status.connect(self._on_binaries_status)
        self._controller.setup_error.connect(self._on_setup_error)
        self._controller.state_changed.connect(self._on_state_changed)

        # Validate any pre-populated text.
        self._on_contact_text_changed(self.contact_edit.text())

        # Logical keyboard tab order.
        self.setTabOrder(self.settings_button, self.relay_links_label)
        self.setTabOrder(self.relay_links_label, self.contact_edit)
        self.setTabOrder(self.contact_edit, self.save_contact_button)
        self.setTabOrder(self.save_contact_button, self.offline_button)
        self.setTabOrder(self.offline_button, self.browse_ffmpeg_button)
        self.setTabOrder(self.browse_ffmpeg_button, self.retry_button)
        self.setTabOrder(self.retry_button, self.install_instructions)

    def _set_chip(self, label: QLabel, text: str, tone: str) -> None:
        """Update the label and semantic tone of a card header status chip."""
        label.setText(text)
        set_widget_role(label, "afvTone", tone)
        label.adjustSize()
        label.updateGeometry()

    def clear_error(self) -> None:
        """Hide any displayed setup error banner."""
        self.error_banner.clear_message()

    def _set_save_enabled(self, enabled: bool) -> None:
        """Enable or disable ``save_contact_button`` and swap to a crisp contrast icon."""
        self.save_contact_button.setEnabled(enabled)
        self.save_contact_button.setIcon(
            self._save_icon_active if enabled else self._save_icon_disabled
        )

    @Slot(str)
    def _on_contact_text_changed(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            self._set_save_enabled(False)
            self.contact_status.set_status(
                "status_info",
                "Enter a reachable email relay or public repository URL to enable saving.",
            )
            self.contact_status.setToolTip("")
            self._set_chip(self._contact_badge, "Optional", "info")
            return
        try:
            validated = validate_mb_contact(cleaned)
        except ConfigError as exc:
            self._set_save_enabled(False)
            err_text = str(exc)
            self.contact_status.set_status("status_error", err_text)
            self.contact_status.setToolTip(err_text)
            self._set_chip(self._contact_badge, "Needs fix", "error")
            return
        self._set_save_enabled(True)
        valid_msg = f"Valid contact: {validated}"
        self.contact_status.set_status("status_ok", valid_msg)
        self.contact_status.setToolTip(valid_msg)
        self._set_chip(self._contact_badge, "Valid", "ok")

    @Slot()
    def _on_contact_return_pressed(self) -> None:
        if self.save_contact_button.isEnabled():
            self._on_save_contact()

    @Slot()
    def _on_save_contact(self) -> None:
        cleaned = self.contact_edit.text().strip()
        try:
            validated = validate_mb_contact(cleaned)
        except ConfigError as exc:
            self._set_save_enabled(False)
            err_text = str(exc)
            self.contact_status.set_status("status_error", err_text)
            self.contact_status.setToolTip(err_text)
            self._set_chip(self._contact_badge, "Needs fix", "error")
            return
        self.error_banner.clear_message()
        self._controller.save_contact(validated)

    @Slot()
    def _on_use_offline_only(self) -> None:
        self.error_banner.clear_message()
        self._controller.use_offline_only()

    @Slot()
    def _on_browse_ffmpeg(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select FFmpeg executable",
            "",
            "All files (*)",
        )
        if not selected:
            return
        self.error_banner.clear_message()
        self._controller.set_ffmpeg_path(Path(selected))

    def _reset_binaries_checking(self) -> None:
        """Reset the binary indicator lines back to their checking state."""
        self._last_binaries_status = None
        self.ffmpeg_status.set_status("status_pending", "Checking FFmpeg…")
        self.ffmpeg_status.setToolTip("")
        self.ffprobe_status.set_status("status_pending", "Checking FFprobe…")
        self.ffprobe_status.setToolTip("")
        self.lame_status.set_status("status_pending", "Checking libmp3lame encoder…")
        self.lame_status.setToolTip("Required MP3 encoder (libmp3lame) bundled with FFmpeg.")
        self._set_chip(self._ffmpeg_badge, "Checking", "neutral")
        self._instructions_container.setVisible(True)

    @Slot()
    def _on_retry_check(self) -> None:
        self.error_banner.clear_message()
        self._reset_binaries_checking()
        self._controller.start_preflight()

    @Slot(object)
    def _on_binaries_status(self, status: BinariesStatus) -> None:
        self._last_binaries_status = status
        ffmpeg = status.ffmpeg
        if ffmpeg.found and ffmpeg.path is not None:
            msg = f"FFmpeg found ({ffmpeg.source}): {ffmpeg.path}"
            self.ffmpeg_status.set_status("status_ok", msg)
            self.ffmpeg_status.setToolTip(str(ffmpeg.path))
        else:
            err = ffmpeg.error or "ffmpeg executable not found."
            msg = f"FFmpeg missing: {err}"
            self.ffmpeg_status.set_status("status_error", msg)
            self.ffmpeg_status.setToolTip(msg)

        ffprobe = status.ffprobe
        if ffprobe.found and ffprobe.path is not None:
            msg = f"FFprobe found ({ffprobe.source}): {ffprobe.path}"
            self.ffprobe_status.set_status("status_ok", msg)
            self.ffprobe_status.setToolTip(str(ffprobe.path))
        else:
            err = ffprobe.error or "ffprobe executable not found."
            msg = f"FFprobe missing: {err}"
            self.ffprobe_status.set_status("status_error", msg)
            self.ffprobe_status.setToolTip(msg)

        if not ffmpeg.found:
            msg = "libmp3lame encoder: unavailable until FFmpeg is installed."
            self.lame_status.set_status("status_warn", msg)
            self.lame_status.setToolTip(msg)
        elif status.has_libmp3lame:
            msg = "libmp3lame encoder: available."
            self.lame_status.set_status("status_ok", msg)
            self.lame_status.setToolTip("FFmpeg supports MP3 encoding via libmp3lame.")
        else:
            msg = "libmp3lame encoder: missing from this FFmpeg build (required to encode MP3)."
            self.lame_status.set_status("status_error", msg)
            self.lame_status.setToolTip(msg)

        all_ready = bool(ffmpeg.found and ffprobe.found and status.has_libmp3lame)
        if all_ready:
            self._set_chip(self._ffmpeg_badge, "Ready", "ok")
        else:
            self._set_chip(self._ffmpeg_badge, "Action needed", "error")
        # Keep install_instructions populated for programmatic inspection, but tuck it
        # away visually once FFmpeg, FFprobe, and libmp3lame are all verified ready.
        self._instructions_container.setVisible(not all_ready)

    @Slot(str)
    def _on_setup_error(self, message: str) -> None:
        text = message.strip() or "Setup error."
        if _needs_env_fallback_hint(text):
            text = (
                f"{text}\n\n"
                "Tip: if config.toml cannot be written, set the AFV_MB_CONTACT environment "
                "variable before launching."
            )
        self.error_banner.set_message("status_error", text, role_label="Setup error")
        if self._last_binaries_status is None:
            self.ffmpeg_status.set_status("status_warn", "Not checked: configuration error")
            self.ffmpeg_status.setToolTip("Fix the configuration error to check FFmpeg.")
            self.ffprobe_status.set_status("status_warn", "Not checked: configuration error")
            self.ffprobe_status.setToolTip("Fix the configuration error to check FFprobe.")
            self.lame_status.set_status("status_warn", "Not checked: configuration error")
            self.lame_status.setToolTip("Fix the configuration error to check libmp3lame.")
            self._set_chip(self._ffmpeg_badge, "Not checked", "warn")
            self._instructions_container.setVisible(True)

    @Slot(object)
    def _on_state_changed(self, state: object) -> None:
        if state == ControllerState.PREFLIGHT:
            self._reset_binaries_checking()
            return
        if state != ControllerState.SETUP:
            return
        if self.contact_edit.text().strip():
            return
        cfg = self._controller.config
        if cfg is not None and cfg.musicbrainz.contact:
            self.contact_edit.setText(cfg.musicbrainz.contact)
