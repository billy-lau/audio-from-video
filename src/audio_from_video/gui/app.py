"""Desktop GUI entry point (``afv-gui``), ``MainWindow``, logging, and exception hooks."""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import traceback
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QByteArray,
    QCoreApplication,
    QEvent,
    QObject,
    QSettings,
    Qt,
    QThread,
    Signal,
    Slot,
)
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from shiboken6 import isValid as qt_is_valid

from audio_from_video.cancel import OperationCancelled
from audio_from_video.gui.controller import (
    SETTINGS_APPLICATION,
    SETTINGS_ORGANIZATION,
    ControllerState,
    RunController,
)
from audio_from_video.gui.dialogs.host import DialogHost
from audio_from_video.gui.log_handler import QtLogHandler
from audio_from_video.gui.models.plan_table import PlanTableModel
from audio_from_video.gui.resources import load_icon
from audio_from_video.gui.screens import (
    InputsScreen,
    PlanningScreen,
    ReviewScreen,
    RunScreen,
    SettingsScreen,
    SetupScreen,
    SummaryScreen,
)
from audio_from_video.gui.screens._common import (
    SectionCard,
    StatusLine,
    ThemedSvgLabel,
    WorkflowHeader,
    apply_ghost_button_role,
    log_directory,
    make_bold_font,
    make_heading_label,
    make_mono_font,
    make_muted_label,
    open_local_path,
    set_accessible,
)
from audio_from_video.gui.theme import (
    ThemeController,
    install_theme,
    is_theme_updating,
    refresh_themed_widgets,
    repolish,
)
from audio_from_video.logging_setup import PACKAGE_LOGGER_NAME, configure_logging

if TYPE_CHECKING:
    from collections.abc import Callable

APP_DISPLAY_NAME = "Audio from Video"
GUI_LOG_FILENAME = "afv-gui.log"

WINDOW_GEOMETRY_KEY = "window/geometry"
WINDOW_STATE_KEY = "window/state"
REVIEW_SPLITTER_KEY = "review/splitter"
LOG_LEVEL_KEY = "log/level"
DEFAULT_LOG_LEVEL_NAME = "INFO"

_DEFAULT_WINDOW_WIDTH = 980
_DEFAULT_WINDOW_HEIGHT = 720
_MIN_WINDOW_WIDTH = 640
_MIN_WINDOW_HEIGHT = 480
_SETTINGS_DIALOG_WIDTH = 720
_SETTINGS_DIALOG_HEIGHT = 640
_SETTINGS_DIALOG_MIN_WIDTH = 480
_SETTINGS_DIALOG_MIN_HEIGHT = 360

_LOG_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

logger = logging.getLogger(__name__)


def open_qsettings(parent: QObject | None = None) -> QSettings:
    """Open the GUI ``QSettings`` store using the four-argument constructor.

    Using ``QSettings(defaultFormat(), UserScope, org, app)`` keeps native
    platform preferences in production while isolating tests that call
    ``QSettings.setDefaultFormat(IniFormat)`` and ``QSettings.setPath(...)``.
    """
    return QSettings(
        QSettings.defaultFormat(),
        QSettings.Scope.UserScope,
        SETTINGS_ORGANIZATION,
        SETTINGS_APPLICATION,
        parent,
    )


def default_log_dir() -> Path:
    """Return the user log directory for ``audio-from-video``."""
    return log_directory()


def default_log_file() -> Path:
    """Return the default UTF-8 GUI log file path (``afv-gui.log``)."""
    return default_log_dir() / GUI_LOG_FILENAME


def parse_log_level(raw: object, *, default: int = logging.INFO) -> int:
    """Parse a logging level name or integer value, defaulting to ``INFO``."""
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        cleaned = raw.strip().upper()
        if cleaned in _LOG_LEVEL_MAP:
            return _LOG_LEVEL_MAP[cleaned]
    return default


def read_configured_log_level() -> int:
    """Read the persisted ``log/level`` value from ``QSettings`` (default ``INFO``)."""
    settings = open_qsettings()
    raw = settings.value(LOG_LEVEL_KEY, DEFAULT_LOG_LEVEL_NAME)
    return parse_log_level(raw, default=logging.INFO)


def apply_log_level_from_settings(qt_log_handler: QtLogHandler | None = None) -> int:
    """Update the active package logger level to match ``QSettings['log/level']``."""
    level = read_configured_log_level()
    pkg_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    pkg_logger.setLevel(level)
    if qt_log_handler is not None:
        qt_log_handler.setLevel(level)
    return level


def configure_gui_logging(
    *,
    log_file: Path | None = None,
    qt_log_handler: QtLogHandler | None = None,
    level: int | None = None,
) -> tuple[logging.Logger, QtLogHandler, Path | None]:
    """Configure UTF-8 file logging and a ``QtLogHandler`` without console output.

    If ``log_file`` cannot be opened or its parent directory cannot be created,
    falls back safely to memory/signal-only logging without raising.
    """
    target = default_log_file() if log_file is None else Path(log_file)
    effective_level = read_configured_log_level() if level is None else level
    handler = qt_log_handler if qt_log_handler is not None else QtLogHandler()
    handler.setLevel(effective_level)

    try:
        pkg_logger = configure_logging(
            console_handler=False,
            log_file=target,
            extra_handlers=(handler,),
            level=effective_level,
        )
        return pkg_logger, handler, target
    except OSError as exc:
        pkg_logger = configure_logging(
            console_handler=False,
            log_file=None,
            extra_handlers=(handler,),
            level=effective_level,
        )
        pkg_logger.warning("Could not open GUI log file %s: %s", target, exc)
        return pkg_logger, handler, None


def teardown_gui_logging(
    qt_log_handler: QtLogHandler | None = None,
    *,
    close_file_handlers: bool = True,
) -> None:
    """Detach and close ``qt_log_handler`` and optionally close file handlers."""
    pkg_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    if qt_log_handler is not None:
        pkg_logger.removeHandler(qt_log_handler)
        with contextlib.suppress(Exception):
            qt_log_handler.close()

    for handler in list(pkg_logger.handlers):
        with contextlib.suppress(Exception):
            handler.flush()
        if close_file_handlers and isinstance(handler, logging.FileHandler):
            pkg_logger.removeHandler(handler)
            with contextlib.suppress(Exception):
                handler.close()


def _flush_package_file_handlers(pkg_logger: logging.Logger) -> set[Path]:
    """Flush attached ``FileHandler`` targets and return their resolved paths."""
    written_paths: set[Path] = set()
    for handler in list(pkg_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            with contextlib.suppress(OSError, ValueError):
                handler.flush()
                written_paths.add(Path(handler.baseFilename).resolve())
    return written_paths


def _append_fallback_log_entry(log_file: Path, heading: str, formatted_tb: str) -> None:
    """Append an unhandled traceback directly to ``log_file`` when no handler wrote it."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp} [CRITICAL] {__name__}: {heading}\n{formatted_tb.rstrip()}\n"
    with contextlib.suppress(OSError):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as stream:
            stream.write(entry)


def _format_traceback(
    exc_type: type[BaseException],
    exc_value: BaseException | None,
    exc_tb: TracebackType | None,
) -> str:
    """Format an exception traceback safely as plain text."""
    if exc_value is None:
        return exc_type.__name__
    try:
        return "".join(traceback.format_exception(exc_type, exc_value, exc_tb)).strip()
    except Exception:
        return f"{exc_type.__name__}: {exc_value!r}"


def _as_byte_array(value: object) -> QByteArray | None:
    """Convert a ``QSettings`` binary value into a non-empty ``QByteArray``."""
    if isinstance(value, QByteArray):
        return value if not value.isEmpty() else None
    if isinstance(value, (bytes, bytearray)) and value:
        return QByteArray(bytes(value))
    return None


class _ExceptionSignalRelay(QObject):
    """Internal ``QObject`` emitting unhandled-exception payloads across threads."""

    exception_raised = Signal(str, str)  # title, details


class _ExceptionUiReceiver(QObject):
    """UI-thread ``QObject`` forwarding unhandled exceptions to ``MainWindow``."""

    def __init__(
        self,
        window: MainWindow | None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._window = window
        self._active = True
        self.received: list[tuple[str, str, QThread]] = []

    def deactivate(self) -> None:
        """Ignore any already-queued exception notifications after hook removal."""
        self._active = False
        self._window = None

    @Slot(str, str)
    def handle_exception(self, title: str, details: str) -> None:
        """Display ``title`` and ``details`` on the UI thread without blocking."""
        if not self._active:
            return
        self.received.append((title, details, QThread.currentThread()))
        window = self._window
        if window is not None and qt_is_valid(window):
            window.show_unhandled_exception(title, details)


class InstalledExceptionHooks:
    """Manage installation and deterministic restoration of global exception hooks."""

    def __init__(
        self,
        *,
        window: MainWindow | None = None,
        log_file: Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        self._log_file = Path(log_file) if log_file is not None else None
        self._in_hook = threading.local()
        self._state_lock = threading.Lock()
        self._uninstalled = False
        self._previous_sys_hook: Callable[
            [type[BaseException], BaseException, TracebackType | None], object
        ] = sys.excepthook
        self._previous_threading_hook: Callable[[threading.ExceptHookArgs], object] = (
            threading.excepthook
        )

        # Keep relay and receiver bound to the UI thread when Qt is running.
        qobject_parent = parent if parent is not None else window
        self._receiver: _ExceptionUiReceiver | None = None
        self._relay: _ExceptionSignalRelay | None = None
        if QCoreApplication.instance() is not None:
            self._receiver = _ExceptionUiReceiver(window, qobject_parent)
            self._relay = _ExceptionSignalRelay(self._receiver)
            self._relay.exception_raised.connect(
                self._receiver.handle_exception,
                Qt.ConnectionType.QueuedConnection,
            )

        sys.excepthook = self._handle_sys_exception
        threading.excepthook = self._handle_threading_exception

    @property
    def active(self) -> bool:
        """Return True while these hooks are currently installed."""
        return not self._uninstalled

    @property
    def delivered_events(self) -> list[tuple[str, str, QThread]]:
        """Return ``(title, details, delivery_thread)`` tuples received on the UI thread."""
        if self._receiver is None:
            return []
        return list(self._receiver.received)

    def _record_and_dispatch(
        self,
        *,
        title: str,
        heading: str,
        exc_type: type[BaseException],
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Write the full traceback to the log file and dispatch a queued UI signal."""
        formatted = _format_traceback(exc_type, exc_value, exc_tb)
        pkg_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
        with contextlib.suppress(Exception):
            pkg_logger.critical("%s\n%s", heading, formatted)

        written_paths = _flush_package_file_handlers(pkg_logger)
        if self._log_file is not None:
            try:
                resolved = self._log_file.resolve()
            except OSError:
                resolved = self._log_file
            if resolved not in written_paths:
                _append_fallback_log_entry(self._log_file, heading, formatted)

        with self._state_lock:
            if self._uninstalled:
                return
            relay = self._relay
        if relay is not None and QCoreApplication.instance() is not None and qt_is_valid(relay):
            with contextlib.suppress(RuntimeError):
                relay.exception_raised.emit(title, formatted)

    def _handle_sys_exception(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType | None,
    ) -> None:
        if isinstance(exc_type, type):
            if issubclass(exc_type, OperationCancelled):
                logger.debug("Ignoring OperationCancelled in sys.excepthook.")
                return
            if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
                self._previous_sys_hook(exc_type, exc_value, exc_tb)
                return

        if getattr(self._in_hook, "active", False):
            return
        self._in_hook.active = True
        try:
            summary = f"{exc_type.__name__}: {exc_value}" if str(exc_value) else exc_type.__name__
            self._record_and_dispatch(
                title="Unexpected error",
                heading=f"Unhandled exception ({summary}):",
                exc_type=exc_type,
                exc_value=exc_value,
                exc_tb=exc_tb,
            )
        finally:
            self._in_hook.active = False

    def _handle_threading_exception(self, args: threading.ExceptHookArgs) -> None:
        exc_type = args.exc_type
        if exc_type is None:
            return
        if isinstance(exc_type, type):
            if issubclass(exc_type, OperationCancelled):
                logger.debug("Ignoring OperationCancelled in threading.excepthook.")
                return
            if issubclass(exc_type, SystemExit):
                return
            if issubclass(exc_type, KeyboardInterrupt):
                self._previous_threading_hook(args)
                return

        if getattr(self._in_hook, "active", False):
            return
        self._in_hook.active = True
        try:
            thread_name = args.thread.name if args.thread is not None else "unknown"
            self._record_and_dispatch(
                title=f"Unexpected error in thread {thread_name}",
                heading=f"Unhandled exception in thread {thread_name!r}:",
                exc_type=exc_type,
                exc_value=args.exc_value,
                exc_tb=args.exc_traceback,
            )
        finally:
            self._in_hook.active = False

    def uninstall(self) -> None:
        """Restore the original ``sys.excepthook`` and ``threading.excepthook``."""
        with self._state_lock:
            if self._uninstalled:
                return
            self._uninstalled = True
            receiver = self._receiver
            relay = self._relay
            self._relay = None

        if sys.excepthook == self._handle_sys_exception:
            sys.excepthook = self._previous_sys_hook
        if threading.excepthook == self._handle_threading_exception:
            threading.excepthook = self._previous_threading_hook

        if receiver is not None and qt_is_valid(receiver):
            receiver.deactivate()
            if relay is not None and qt_is_valid(relay):
                with contextlib.suppress(RuntimeError, TypeError):
                    relay.exception_raised.disconnect(receiver.handle_exception)
            with contextlib.suppress(RuntimeError):
                receiver.deleteLater()

    def __enter__(self) -> InstalledExceptionHooks:
        return self

    def __exit__(self, *args: object) -> None:
        _ = args
        self.uninstall()


def install_exception_hooks(
    window: MainWindow | None = None,
    *,
    log_file: Path | None = None,
    parent: QObject | None = None,
) -> InstalledExceptionHooks:
    """Install GUI ``sys.excepthook`` and ``threading.excepthook`` handlers.

    Worker-thread exceptions are logged immediately and forwarded via a queued
    Qt signal to the UI thread, never constructing widgets from a background
    thread. Call ``.uninstall()`` (or use as a context manager) to restore the
    previous hooks.
    """
    return InstalledExceptionHooks(window=window, log_file=log_file, parent=parent)


def _safe_disconnect(signal: object, slot: object) -> None:
    """Disconnect a Qt signal-slot edge without raising if already detached."""
    disconnect = getattr(signal, "disconnect", None)
    if callable(disconnect):
        with contextlib.suppress(RuntimeError, TypeError):
            disconnect(slot)


class MainWindow(QMainWindow):
    """Top-level window hosting all screens, the shared table model, dialogs, and Settings."""

    def __init__(
        self,
        controller: RunController | None = None,
        parent: QWidget | None = None,
        *,
        qt_log_handler: QtLogHandler | None = None,
        log_file: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._owns_controller = controller is None
        self._controller: RunController = RunController(self) if controller is None else controller
        self._qt_log_handler = qt_log_handler
        self._log_file = Path(log_file) if log_file is not None else None
        self._settings_dialog: QDialog | None = None
        self._settings_screen: SettingsScreen | None = None
        self._exception_dialog: QMessageBox | None = None
        self._last_active_state: ControllerState = self._controller.state

        self.setObjectName("main_window")
        self.setWindowTitle(APP_DISPLAY_NAME)
        self.setWindowIcon(load_icon("app_icon"))
        self.setMinimumSize(_MIN_WINDOW_WIDTH, _MIN_WINDOW_HEIGHT)
        self.resize(_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT)
        set_accessible(
            self,
            APP_DISPLAY_NAME,
            "Desktop application for extracting and tagging MP3 audio from video files.",
        )

        # Single PlanTableModel shared by ReviewScreen and RunScreen (section 5.5).
        self.plan_model = PlanTableModel(self)

        # Central layout: read-only workflow header, optional inline error panel, and pages.
        central = QWidget(self)
        central.setObjectName("central_container")
        central.setProperty("afvSurface", "window")
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        self.workflow_header = WorkflowHeader(central)
        central_layout.addWidget(self.workflow_header)

        self.error_panel = self._build_error_panel(central)
        central_layout.addWidget(self.error_panel)

        self.stack = QStackedWidget(central)
        self.stack.setObjectName("workflow_stack")
        set_accessible(
            self.stack,
            "Workflow screens",
            "Contains the setup, inputs, planning, review, extraction, and summary screens.",
        )

        self.preflight_screen = self._build_preflight_screen(self.stack)
        self.setup_screen = SetupScreen(self._controller, self.stack)
        self.inputs_screen = InputsScreen(self._controller, self.stack)
        self.planning_screen = PlanningScreen(self._controller, self.stack)
        self.review_screen = ReviewScreen(self._controller, self.plan_model, self.stack)
        self.run_screen = RunScreen(self._controller, self.plan_model, self.stack)
        self.summary_screen = SummaryScreen(self._controller, self.stack)

        self._pages: dict[ControllerState, QWidget] = {
            ControllerState.PREFLIGHT: self.preflight_screen,
            ControllerState.SETUP: self.setup_screen,
            ControllerState.INPUTS: self.inputs_screen,
            ControllerState.PLANNING: self.planning_screen,
            ControllerState.REVIEW: self.review_screen,
            ControllerState.RUNNING: self.run_screen,
            ControllerState.SUMMARY: self.summary_screen,
        }
        for page in self._pages.values():
            self.stack.addWidget(page)

        central_layout.addWidget(self.stack, stretch=1)
        self.setCentralWidget(central)

        # Non-blocking prompt dialog host (lane I).
        self.dialog_host = DialogHost(self._controller, self)

        # Preferences keyboard shortcut: fall back to Ctrl+, when the platform plugin
        # defines no StandardKey.Preferences binding (Windows, minimal Linux, and offscreen).
        prefs_bindings = QKeySequence.keyBindings(QKeySequence.StandardKey.Preferences)
        prefs_sequence = prefs_bindings[0] if prefs_bindings else QKeySequence("Ctrl+,")
        self.settings_shortcut = QShortcut(prefs_sequence, self)
        self.settings_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.settings_shortcut.activated.connect(self.open_settings)

        # Wire screens, model, and controller using bound QObject methods only (Rule 2).
        self.setup_screen.settings_button.clicked.connect(self.open_settings)
        self.inputs_screen.settings_button.clicked.connect(self.open_settings)
        self.copy_error_button.clicked.connect(self._on_copy_error_details)
        self.open_error_log_button.clicked.connect(self._on_open_log_folder)
        self.dismiss_error_button.clicked.connect(self.clear_error_panel)

        self._controller.state_changed.connect(self._on_state_changed)
        self._controller.plan_ready.connect(self._on_plan_ready)
        self._controller.event_received.connect(self.plan_model.apply_event)
        self._controller.row_changed.connect(self.plan_model.refresh_row)
        self._controller.settings_saved.connect(self._on_settings_saved)
        self._controller.error.connect(self._on_controller_error)

        # Seed model if the window attaches to a controller that already holds a plan.
        if self._controller.plan_items:
            self.plan_model.set_output_root(self._controller.output_root)
            self.plan_model.set_items(list(self._controller.plan_items))

        # Restore persisted window geometry, dock/toolbar state, and review splitter sizes.
        self.restore_window_settings()

        # Sync stacked page and Settings button state to the controller's initial state.
        self._sync_page_for_state(self._controller.state)
        self._update_settings_availability(self._controller.state)

    def _build_preflight_screen(self, parent: QWidget) -> QWidget:
        """Create the lightweight startup check page shown during ``PREFLIGHT``."""
        page = QWidget(parent)
        page.setObjectName("preflight_screen")
        page.setProperty("afvSurface", "window")
        set_accessible(
            page,
            "Startup check screen",
            "Checking configuration files and FFmpeg availability before starting.",
        )

        outer = QVBoxLayout(page)
        outer.setContentsMargins(32, 32, 32, 32)
        outer.setSpacing(0)
        outer.addStretch(1)

        center_row = QHBoxLayout()
        center_row.addStretch(1)

        card = SectionCard(
            margins=(36, 30, 36, 30),
            spacing=10,
            parent=page,
        )
        card.setObjectName("preflight_card")
        card.setMinimumWidth(400)
        card.setMaximumWidth(480)

        self.preflight_icon = ThemedSvgLabel("app_icon", size=52, parent=card)
        self.preflight_icon.setObjectName("preflight_icon_label")
        card.body_layout.addWidget(self.preflight_icon, alignment=Qt.AlignmentFlag.AlignCenter)

        title = make_heading_label(APP_DISPLAY_NAME, scale=1.45, parent=card)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card.body_layout.addWidget(title)

        self.preflight_label = make_muted_label(
            "Checking configuration and FFmpeg…",
            scale=0.96,
            parent=card,
        )
        self.preflight_label.setObjectName("preflight_status_label")
        self.preflight_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        set_accessible(
            self.preflight_label,
            "Startup check status",
            "Displays startup check progress while configuration and FFmpeg are inspected.",
        )
        card.body_layout.addWidget(self.preflight_label)

        card.body_layout.addSpacing(4)

        self.preflight_progress = QProgressBar(card)
        self.preflight_progress.setObjectName("preflight_progress_bar")
        self.preflight_progress.setRange(0, 0)
        self.preflight_progress.setTextVisible(False)
        self.preflight_progress.setProperty("afvDensity", "compact")
        self.preflight_progress.setMinimumWidth(240)
        self.preflight_progress.setMaximumWidth(360)
        set_accessible(
            self.preflight_progress,
            "Startup check progress",
            "Busy indicator shown while configuration and FFmpeg binaries are inspected.",
        )
        card.body_layout.addWidget(self.preflight_progress, alignment=Qt.AlignmentFlag.AlignCenter)

        hint = make_muted_label(
            "Verifying local tools, encoder support, and saved defaults.",
            scale=0.85,
            parent=card,
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card.body_layout.addWidget(hint)

        center_row.addWidget(card)
        center_row.addStretch(1)
        outer.addLayout(center_row)

        outer.addStretch(1)
        return page

    def _build_error_panel(self, parent: QWidget) -> QFrame:
        """Create a dismissible top-level error panel for unexpected failures."""
        panel = QFrame(parent)
        panel.setObjectName("main_error_panel")
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setFrameShadow(QFrame.Shadow.Plain)
        panel.setProperty("afvBanner", True)
        panel.setProperty("afvTone", "error")
        repolish(panel)
        set_accessible(
            panel,
            "Application error panel",
            "Displays unexpected error details with options to copy details or open the log.",
        )

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(8)

        self.error_header = StatusLine(
            "status_error",
            "Unexpected error",
            icon_size=18,
            selectable=True,
            parent=panel,
        )
        self.error_header.text_label.setFont(make_bold_font(self.error_header.text_label))
        layout.addWidget(self.error_header)

        self.error_details = QPlainTextEdit(panel)
        self.error_details.setObjectName("main_error_details")
        self.error_details.setReadOnly(True)
        self.error_details.setProperty("afvReadonly", True)
        self.error_details.setFont(make_mono_font(self.error_details))
        self.error_details.setMaximumHeight(120)
        set_accessible(
            self.error_details,
            "Error details",
            "Diagnostic message and traceback for the reported error.",
        )
        layout.addWidget(self.error_details)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        self.copy_error_button = QPushButton("Copy &details", panel)
        self.copy_error_button.setIcon(load_icon("copy"))
        self.copy_error_button.setToolTip(
            "Copy the error summary and diagnostic details to the clipboard."
        )
        set_accessible(
            self.copy_error_button,
            "Copy error details",
            "Copy the error summary and diagnostic details to the clipboard.",
        )
        self.open_error_log_button = QPushButton("Open &log folder", panel)
        self.open_error_log_button.setIcon(load_icon("folder"))
        self.open_error_log_button.setToolTip(
            "Open the application log folder in the system file manager."
        )
        set_accessible(
            self.open_error_log_button,
            "Open log folder",
            "Open the application log folder in the system file manager.",
        )
        self.dismiss_error_button = QPushButton("&Dismiss", panel)
        apply_ghost_button_role(self.dismiss_error_button)
        self.dismiss_error_button.setToolTip("Hide this error notice.")
        set_accessible(
            self.dismiss_error_button,
            "Dismiss error panel",
            "Hide this error notice after reviewing or copying its details.",
        )
        button_row.addWidget(self.copy_error_button)
        button_row.addWidget(self.open_error_log_button)
        button_row.addStretch(1)
        button_row.addWidget(self.dismiss_error_button)
        layout.addLayout(button_row)

        self.error_log_feedback_label = QLabel(panel)
        self.error_log_feedback_label.setObjectName("main_error_log_feedback_label")
        self.error_log_feedback_label.setWordWrap(True)
        self.error_log_feedback_label.setTextFormat(Qt.TextFormat.PlainText)
        self.error_log_feedback_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        set_accessible(
            self.error_log_feedback_label,
            "Log folder notice",
            "Displays inline feedback when opening the log folder fails.",
        )
        self.error_log_feedback_label.setVisible(False)
        layout.addWidget(self.error_log_feedback_label)

        repolish(panel, include_children=True)
        panel.setVisible(False)
        return panel

    # -- Properties ------------------------------------------------------------

    @property
    def controller(self) -> RunController:
        """Return the ``RunController`` wired to this window."""
        return self._controller

    @property
    def current_screen(self) -> QWidget | None:
        """Return the currently visible widget in the workflow stack."""
        return self.stack.currentWidget()

    @property
    def settings_dialog(self) -> QDialog | None:
        """Return the open Settings ``QDialog``, or ``None`` if closed."""
        if self._settings_dialog is not None and qt_is_valid(self._settings_dialog):
            return self._settings_dialog
        return None

    @property
    def settings_screen(self) -> SettingsScreen | None:
        """Return the ``SettingsScreen`` inside the active Settings dialog, if open."""
        if self._settings_screen is not None and qt_is_valid(self._settings_screen):
            return self._settings_screen
        return None

    @property
    def exception_dialog(self) -> QMessageBox | None:
        """Return the active non-blocking exception notice dialog, if open."""
        if self._exception_dialog is not None and qt_is_valid(self._exception_dialog):
            return self._exception_dialog
        return None

    @property
    def log_file_path(self) -> Path | None:
        """Return the active GUI log file path, if configured."""
        return self._log_file

    # -- Window settings persistence ------------------------------------------

    def restore_window_settings(self) -> None:
        """Restore saved geometry, state, and review splitter sizes from ``QSettings``."""
        settings = open_qsettings(self)

        geom = _as_byte_array(settings.value(WINDOW_GEOMETRY_KEY))
        if geom is not None:
            self.restoreGeometry(geom)

        win_state = _as_byte_array(settings.value(WINDOW_STATE_KEY))
        if win_state is not None:
            self.restoreState(win_state)

        splitter_state = _as_byte_array(settings.value(REVIEW_SPLITTER_KEY))
        if splitter_state is not None:
            self.review_screen.splitter.restoreState(splitter_state)
        else:
            self.review_screen.splitter.setSizes([380, 220])

    def save_window_settings(self) -> None:
        """Persist window geometry, state, and review splitter sizes to ``QSettings``."""
        settings = open_qsettings(self)
        settings.setValue(WINDOW_GEOMETRY_KEY, self.saveGeometry())
        settings.setValue(WINDOW_STATE_KEY, self.saveState())
        settings.setValue(REVIEW_SPLITTER_KEY, self.review_screen.splitter.saveState())
        settings.sync()

    # -- Internal state helpers ------------------------------------------------

    def _can_open_settings(self, state: ControllerState) -> bool:
        return state in (ControllerState.SETUP, ControllerState.INPUTS)

    def _is_settings_dialog_open(self) -> bool:
        return (
            self._settings_dialog is not None
            and qt_is_valid(self._settings_dialog)
            and self._settings_dialog.isVisible()
        )

    def _sync_page_for_state(self, state: ControllerState) -> None:
        if state == ControllerState.CANCELLING:
            if self._last_active_state == ControllerState.RUNNING:
                self.stack.setCurrentWidget(self.run_screen)
            elif self._last_active_state == ControllerState.PLANNING:
                self.stack.setCurrentWidget(self.planning_screen)
            self.workflow_header.update_state(state, last_active_state=self._last_active_state)
            return

        target = self._pages.get(state)
        if target is not None:
            self.stack.setCurrentWidget(target)
        self._last_active_state = state
        self.workflow_header.update_state(state, last_active_state=self._last_active_state)

    def _update_settings_availability(self, state: ControllerState) -> None:
        enabled = self._can_open_settings(state)
        self.setup_screen.settings_button.setEnabled(enabled)
        self.inputs_screen.settings_button.setEnabled(enabled)
        self.settings_shortcut.setEnabled(enabled)
        if not enabled:
            self._close_settings_dialog()

    def _detach_settings_screen(self, screen: SettingsScreen) -> None:
        """Disconnect controller signals tied to ``screen`` before destroying it."""
        _safe_disconnect(self._controller.cache_stats_ready, screen._on_cache_stats_ready)
        _safe_disconnect(self._controller.cache_cleared, screen._on_cache_cleared)
        _safe_disconnect(self._controller.settings_saved, screen._on_settings_saved)
        _safe_disconnect(self._controller.error, screen._on_controller_error)

    def _close_settings_dialog(self) -> None:
        """Close and release any active Settings modal dialog without blocking."""
        dialog = self._settings_dialog
        screen = self._settings_screen
        self._settings_dialog = None
        self._settings_screen = None

        if screen is not None and qt_is_valid(screen):
            self._detach_settings_screen(screen)
        if dialog is not None and qt_is_valid(dialog):
            _safe_disconnect(dialog.finished, self._on_settings_dialog_finished)
            dialog.hide()
            dialog.deleteLater()

    def _close_exception_dialog(self) -> None:
        """Close and release any active non-blocking exception notice dialog."""
        box = self._exception_dialog
        self._exception_dialog = None
        if box is not None and qt_is_valid(box):
            _safe_disconnect(box.finished, self._on_exception_dialog_finished)
            box.hide()
            box.deleteLater()

    def close_dialogs_for_shutdown(self) -> None:
        """Close any window-owned modal dialogs prior to application teardown."""
        self._close_settings_dialog()
        self._close_exception_dialog()

    def _controller_has_active_workers(self) -> bool:
        """Return True when ``RunController`` currently owns a running background thread."""
        checker = getattr(self._controller, "_has_active_workers", None)
        if callable(checker):
            return bool(checker())
        return False

    # -- Public slots and helpers ---------------------------------------------

    @Slot()
    def open_settings(self) -> None:
        """Open the window-modal Settings dialog via non-blocking ``open()``."""
        if not self._can_open_settings(self._controller.state):
            return

        if self._settings_dialog is not None and qt_is_valid(self._settings_dialog):
            if not self._settings_dialog.isVisible():
                self._settings_dialog.open()
            self._settings_dialog.raise_()
            self._settings_dialog.activateWindow()
            return

        dialog = QDialog(self)
        dialog.setObjectName("settings_dialog")
        dialog.setWindowTitle(f"Settings — {APP_DISPLAY_NAME}")
        dialog.setWindowIcon(load_icon("app_icon"))
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumSize(_SETTINGS_DIALOG_MIN_WIDTH, _SETTINGS_DIALOG_MIN_HEIGHT)
        dialog.resize(_SETTINGS_DIALOG_WIDTH, _SETTINGS_DIALOG_HEIGHT)
        set_accessible(
            dialog,
            "Settings",
            "Edit saved defaults, view environment overrides, inspect or clear the cache, "
            "and configure startup and logging preferences.",
        )

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        screen = SettingsScreen(self._controller, dialog)
        layout.addWidget(screen)

        # Keep MainWindow._on_settings_saved ordered after SettingsScreen._on_settings_saved
        # so deferred QSettings writes (such as log/level committed after a config.toml
        # save succeeds) are already persisted before refresh_log_level() reads them.
        _safe_disconnect(self._controller.settings_saved, self._on_settings_saved)
        self._controller.settings_saved.connect(self._on_settings_saved)

        screen.close_button.clicked.connect(dialog.reject)
        dialog.finished.connect(self._on_settings_dialog_finished)

        self._settings_dialog = dialog
        self._settings_screen = screen

        dialog.open()
        screen.refresh_cache_stats()

    @Slot()
    def refresh_log_level(self) -> None:
        """Sync the active package logger level from ``QSettings['log/level']``."""
        apply_log_level_from_settings(self._qt_log_handler)

    def _clear_error_log_feedback(self) -> None:
        """Clear and hide any inline notice from opening the log folder."""
        self.error_log_feedback_label.clear()
        set_accessible(
            self.error_log_feedback_label,
            "Log folder notice",
            "Displays inline feedback when opening the log folder fails.",
        )
        self.error_log_feedback_label.setVisible(False)

    def show_error_panel(self, title: str, details: str) -> None:
        """Display the inline dismissible error panel with ``title`` and ``details``."""
        heading = title.strip() or "Unexpected error"
        self.error_header.set_status("status_error", heading)
        self.error_details.setPlainText(details.strip())
        self._clear_error_log_feedback()
        self.error_panel.setVisible(True)

    @Slot()
    def clear_error_panel(self) -> None:
        """Clear and hide the inline application error panel."""
        self._clear_error_log_feedback()
        self.error_details.clear()
        self.error_panel.setVisible(False)

    @Slot(str, str)
    def show_unhandled_exception(self, title: str, details: str) -> None:
        """Present an unhandled exception notice on the UI thread without blocking."""
        heading = title.strip() or "Unexpected error"
        cleaned = details.strip() or heading
        self.show_error_panel(heading, cleaned)

        target_log = self._log_file if self._log_file is not None else default_log_file()
        summary_text = "An unexpected error occurred. Details have been written to the log file."
        info_text = f"Log file: {target_log}"

        if self._exception_dialog is not None and qt_is_valid(self._exception_dialog):
            self._exception_dialog.setWindowTitle(heading)
            self._exception_dialog.setDetailedText(cleaned)
            if not self._exception_dialog.isVisible():
                self._exception_dialog.open()
            return

        box = QMessageBox(self)
        box.setObjectName("unhandled_exception_dialog")
        box.setWindowTitle(heading)
        box.setWindowModality(Qt.WindowModality.WindowModal)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setText(summary_text)
        box.setInformativeText(info_text)
        box.setDetailedText(cleaned)
        box.setStandardButtons(QMessageBox.StandardButton.Close)
        box.setDefaultButton(QMessageBox.StandardButton.Close)
        set_accessible(box, heading, summary_text)
        box.finished.connect(self._on_exception_dialog_finished)
        self._exception_dialog = box
        box.open()

    # -- Bound Qt slots (Rule 2) ----------------------------------------------

    @Slot(object)
    def _on_state_changed(self, state: object) -> None:
        if not isinstance(state, ControllerState):
            return
        previous = self._last_active_state

        if state == ControllerState.PLANNING:
            # Reset stale rows before Stage 1 emits new per-track events.
            self.plan_model.set_output_root(None)
            self.plan_model.set_items([])
            self.clear_error_panel()
        elif state == ControllerState.INPUTS and previous in (
            ControllerState.REVIEW,
            ControllerState.SUMMARY,
        ):
            self.plan_model.set_output_root(None)
            self.plan_model.set_items([])
        elif state in (
            ControllerState.SETUP,
            ControllerState.REVIEW,
            ControllerState.RUNNING,
        ):
            # Never clear SetupScreen.error_banner here; preserve setup_error text.
            self.clear_error_panel()

        self._sync_page_for_state(state)
        self._update_settings_availability(state)

    @Slot(object)
    def _on_plan_ready(self, items: object) -> None:
        self.plan_model.set_output_root(self._controller.output_root)
        plan_items = items if isinstance(items, list) else list(items)  # type: ignore[arg-type]
        self.plan_model.set_items(plan_items)

    @Slot(object)
    def _on_settings_saved(self, config: object) -> None:
        _ = config
        self.refresh_log_level()

    @Slot(str, str)
    def _on_controller_error(self, title: str, details: str) -> None:
        if self._is_settings_dialog_open():
            return
        state = self._controller.state
        if state in (
            ControllerState.PREFLIGHT,
            ControllerState.SETUP,
            ControllerState.RUNNING,
            ControllerState.SUMMARY,
        ):
            return
        if (
            state == ControllerState.CANCELLING
            and self._last_active_state == ControllerState.RUNNING
        ):
            # SummaryScreen handles extraction errors when ExecuteWorker exits mid-cancel.
            return
        if state == ControllerState.PLANNING:
            self.planning_screen.show_error(title, details)
        self.show_error_panel(title, details)

    @Slot(int)
    def _on_settings_dialog_finished(self, result: int) -> None:
        _ = result
        if open_qsettings().contains(LOG_LEVEL_KEY):
            self.refresh_log_level()
        self._close_settings_dialog()

    @Slot(int)
    def _on_exception_dialog_finished(self, result: int) -> None:
        _ = result
        self._close_exception_dialog()

    @Slot()
    def _on_copy_error_details(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return
        heading = self.error_header.text
        details = self.error_details.toPlainText()
        payload = f"{heading}\n\n{details}".strip() if details else heading
        clipboard.setText(payload)

    @Slot()
    def _on_open_log_folder(self) -> None:
        folder = self._log_file.parent if self._log_file is not None else default_log_dir()
        if open_local_path(folder):
            self._clear_error_log_feedback()
            return
        try:
            exists = folder.expanduser().exists()
        except OSError:
            exists = False
        notice = (
            f"Could not open log folder: {folder}."
            if exists
            else f"Log folder does not exist yet: {folder}."
        )
        self.error_log_feedback_label.setText(notice)
        set_accessible(self.error_log_feedback_label, "Log folder notice", notice)
        self.error_log_feedback_label.setVisible(True)

    # -- QWidget events -------------------------------------------------------

    def changeEvent(self, event: QEvent) -> None:
        """Refresh themed SVG pixmaps and primary icons on palette or style updates."""
        super().changeEvent(event)
        if not is_theme_updating() and event.type() in (
            QEvent.Type.PaletteChange,
            QEvent.Type.ApplicationPaletteChange,
            QEvent.Type.StyleChange,
        ):
            refresh_themed_widgets(self)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Follow shutdown order (§4.5): persist UI state, stop workers, then quit."""
        self.save_window_settings()
        self.close_dialogs_for_shutdown()

        if self._controller_has_active_workers():
            # Step 1 & 2 happen inside controller.shutdown() (token.cancel, withdraw prompt).
            event.ignore()
            self._controller.shutdown()
            # Step 3: hide the window while worker thread(s) unwind safely off-thread.
            self.hide()
            return

        self._controller.shutdown()
        event.accept()


def _drain_active_workers_before_exit(app: QApplication, controller: RunController) -> None:
    """Wait via the Qt event loop if a worker is still finishing after ``app.exec()``."""
    checker = getattr(controller, "_has_active_workers", None)
    if not callable(checker) or not checker():
        return
    controller.shutdown()
    while checker():
        app.processEvents()
        QThread.msleep(10)


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the ``afv-gui`` desktop application."""
    # Repeat the launcher's runtime check for callers that start the app directly.
    from audio_from_video.gui.launcher import unsupported_runtime_message

    unsupported = unsupported_runtime_message()
    if unsupported is not None:
        print(unsupported, file=sys.stderr)
        return 1

    qt_argv = list(sys.argv if argv is None else argv)
    if not qt_argv:
        qt_argv = [SETTINGS_APPLICATION]

    existing_app = QApplication.instance()
    if isinstance(existing_app, QApplication):
        app = existing_app
        owns_app = False
    else:
        app = QApplication(qt_argv)
        owns_app = True

    app.setOrganizationName(SETTINGS_ORGANIZATION)
    app.setApplicationName(SETTINGS_APPLICATION)
    app.setApplicationDisplayName(APP_DISPLAY_NAME)

    previous_quit_on_last = app.quitOnLastWindowClosed()
    app.setQuitOnLastWindowClosed(False)

    controller: RunController | None = None
    window: MainWindow | None = None
    qt_log_handler: QtLogHandler | None = None
    hooks: InstalledExceptionHooks | None = None
    theme_controller: ThemeController | None = None

    try:
        theme_controller = install_theme(app)
        app.setWindowIcon(load_icon("app_icon"))
        _pkg_logger, qt_log_handler, active_log_file = configure_gui_logging()
        controller = RunController(app)
        window = MainWindow(
            controller=controller,
            qt_log_handler=qt_log_handler,
            log_file=active_log_file,
        )
        hooks = install_exception_hooks(window=window, log_file=active_log_file)
        window.show()
        controller.start_preflight()
        return int(app.exec())
    finally:
        if hooks is not None:
            hooks.uninstall()
        if window is not None and qt_is_valid(window):
            window.close_dialogs_for_shutdown()
            window.hide()
        if controller is not None and qt_is_valid(controller):
            controller.shutdown()
            _drain_active_workers_before_exit(app, controller)
        if window is not None and qt_is_valid(window):
            window.deleteLater()
        if controller is not None and qt_is_valid(controller):
            controller.deleteLater()
        app.processEvents()
        if theme_controller is not None:
            theme_controller.uninstall()
        teardown_gui_logging(qt_log_handler)
        app.setQuitOnLastWindowClosed(previous_quit_on_last)
        if owns_app:
            app.processEvents()
