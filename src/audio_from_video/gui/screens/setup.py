"""First-run Setup screen for MusicBrainz contact, FFmpeg check, and offline opt-in."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from audio_from_video.binaries import format_install_instructions
from audio_from_video.config import ConfigError, validate_mb_contact
from audio_from_video.gui.controller import ControllerState
from audio_from_video.gui.resources import load_icon
from audio_from_video.gui.screens._common import (
    IconStatusBanner,
    StatusLine,
    make_heading_label,
    set_accessible,
)

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
        layout = QVBoxLayout(body)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        # Top header bar with Settings entry point.
        header_row = QHBoxLayout()
        header_row.setSpacing(12)
        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        self.title_label = make_heading_label("Welcome to Audio from Video", parent=body)
        subtitle = QLabel(
            "Check your FFmpeg installation and set a MusicBrainz contact for online lookups, "
            "or run offline only.",
            body,
        )
        subtitle.setWordWrap(True)
        title_box.addWidget(self.title_label)
        title_box.addWidget(subtitle)
        header_row.addLayout(title_box, stretch=1)

        self.settings_button = QPushButton("Se&ttings…", body)
        self.settings_button.setIcon(load_icon("settings"))
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
        self.error_banner = IconStatusBanner(body)
        layout.addWidget(self.error_banner)

        # MusicBrainz contact configuration.
        contact_group = QGroupBox("MusicBrainz Contact (for online lookups)", body)
        contact_layout = QVBoxLayout(contact_group)
        contact_layout.setSpacing(8)

        contact_info = QLabel(
            "MusicBrainz requires a reachable email address or public repository URL in the "
            "User-Agent header. It is stored locally in config.toml and sent only to MusicBrainz "
            "and the Cover Art Archive.",
            contact_group,
        )
        contact_info.setWordWrap(True)
        contact_layout.addWidget(contact_info)

        self.relay_links_label = QLabel(
            f"Recommended anonymous email relays: "
            f'<a href="{_APPLE_HIDE_MY_EMAIL_URL}">Apple Hide My Email</a>, '
            f'<a href="{_SIMPLELOGIN_URL}">SimpleLogin</a>, and '
            f'<a href="{_DUCK_ADDRESS_URL}">Duck Address</a>.',
            contact_group,
        )
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
        contact_layout.addWidget(self.relay_links_label)

        contact_label = QLabel("&Contact (email relay or public URL):", contact_group)
        self.contact_edit = QLineEdit(contact_group)
        self.contact_edit.setPlaceholderText(
            "Enter a reachable email relay or public https:// repository URL"
        )
        self.contact_edit.setClearButtonEnabled(True)
        contact_label.setBuddy(self.contact_edit)
        set_accessible(
            self.contact_edit,
            "MusicBrainz contact",
            "Reachable email relay or public HTTP/HTTPS repository URL for MusicBrainz.",
        )
        contact_layout.addWidget(contact_label)
        contact_layout.addWidget(self.contact_edit)

        self.contact_status = StatusLine(
            "status_info",
            "Enter a reachable email relay or public repository URL to enable saving.",
            parent=contact_group,
        )
        contact_layout.addWidget(self.contact_status)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(10)
        self.save_contact_button = QPushButton("&Save contact", contact_group)
        self.save_contact_button.setEnabled(False)
        self.save_contact_button.setDefault(True)
        set_accessible(
            self.save_contact_button,
            "Save contact",
            "Save the validated MusicBrainz contact to config.toml and continue.",
        )

        self.offline_button = QPushButton("Use &offline only", contact_group)
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
        contact_layout.addLayout(actions_row)
        layout.addWidget(contact_group)

        # FFmpeg, FFprobe, and libmp3lame availability check.
        ffmpeg_group = QGroupBox("FFmpeg and FFprobe", body)
        ffmpeg_layout = QVBoxLayout(ffmpeg_group)
        ffmpeg_layout.setSpacing(8)

        self.ffmpeg_status = StatusLine("status_pending", "Checking FFmpeg…", parent=ffmpeg_group)
        self.ffprobe_status = StatusLine("status_pending", "Checking FFprobe…", parent=ffmpeg_group)
        self.lame_status = StatusLine(
            "status_pending",
            "Checking libmp3lame encoder…",
            parent=ffmpeg_group,
        )
        ffmpeg_layout.addWidget(self.ffmpeg_status)
        ffmpeg_layout.addWidget(self.ffprobe_status)
        ffmpeg_layout.addWidget(self.lame_status)

        browse_row = QHBoxLayout()
        browse_row.setSpacing(8)
        self.browse_ffmpeg_button = QPushButton("&Browse for ffmpeg…", ffmpeg_group)
        self.browse_ffmpeg_button.setIcon(load_icon("folder"))
        set_accessible(
            self.browse_ffmpeg_button,
            "Browse for ffmpeg",
            "Select an explicit ffmpeg executable path and save it to config.toml.",
        )
        self.retry_button = QPushButton("&Retry check", ffmpeg_group)
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
        ffmpeg_layout.addLayout(browse_row)

        instructions_label = QLabel("Install instructions:", ffmpeg_group)
        self.install_instructions = QPlainTextEdit(ffmpeg_group)
        self.install_instructions.setReadOnly(True)
        self.install_instructions.setPlainText(format_install_instructions())
        self.install_instructions.setMaximumHeight(140)
        set_accessible(
            self.install_instructions,
            "FFmpeg install instructions",
            "Operating-system instructions for installing FFmpeg and FFprobe.",
        )
        ffmpeg_layout.addWidget(instructions_label)
        ffmpeg_layout.addWidget(self.install_instructions)
        layout.addWidget(ffmpeg_group)

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

    def clear_error(self) -> None:
        """Hide any displayed setup error banner."""
        self.error_banner.clear_message()

    @Slot(str)
    def _on_contact_text_changed(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            self.save_contact_button.setEnabled(False)
            self.contact_status.set_status(
                "status_info",
                "Enter a reachable email relay or public repository URL to enable saving.",
            )
            return
        try:
            validated = validate_mb_contact(cleaned)
        except ConfigError as exc:
            self.save_contact_button.setEnabled(False)
            self.contact_status.set_status("status_error", str(exc))
            return
        self.save_contact_button.setEnabled(True)
        self.contact_status.set_status(
            "status_ok",
            f"Valid contact: {validated}",
        )

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
            self.save_contact_button.setEnabled(False)
            self.contact_status.set_status("status_error", str(exc))
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
            "Select ffmpeg executable",
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
        self.ffprobe_status.set_status("status_pending", "Checking FFprobe…")
        self.lame_status.set_status("status_pending", "Checking libmp3lame encoder…")

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
            self.ffmpeg_status.set_status(
                "status_ok",
                f"FFmpeg found ({ffmpeg.source}): {ffmpeg.path}",
            )
        else:
            err = ffmpeg.error or "ffmpeg executable not found."
            self.ffmpeg_status.set_status("status_error", f"FFmpeg missing: {err}")

        ffprobe = status.ffprobe
        if ffprobe.found and ffprobe.path is not None:
            self.ffprobe_status.set_status(
                "status_ok",
                f"FFprobe found ({ffprobe.source}): {ffprobe.path}",
            )
        else:
            err = ffprobe.error or "ffprobe executable not found."
            self.ffprobe_status.set_status("status_error", f"FFprobe missing: {err}")

        if not ffmpeg.found:
            self.lame_status.set_status(
                "status_warn",
                "libmp3lame encoder: unavailable until FFmpeg is installed.",
            )
        elif status.has_libmp3lame:
            self.lame_status.set_status(
                "status_ok",
                "libmp3lame encoder: available.",
            )
        else:
            self.lame_status.set_status(
                "status_error",
                "libmp3lame encoder: missing from this FFmpeg build (required to encode MP3).",
            )

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
            self.ffprobe_status.set_status("status_warn", "Not checked: configuration error")
            self.lame_status.set_status("status_warn", "Not checked: configuration error")

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
