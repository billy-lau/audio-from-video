"""UI-thread controller state machine and binary status model for the desktop GUI."""

from __future__ import annotations

import logging
import os
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QCoreApplication, QObject, QSettings, Qt, QThread, Signal, Slot

from audio_from_video import config as _config_mod
from audio_from_video import runner as _runner_mod
from audio_from_video.binaries import BinaryResolution, ResolvedBinaries
from audio_from_video.cancel import CancelToken
from audio_from_video.events import NOT_RUN_MESSAGE, TrackEvent, TrackStage
from audio_from_video.gui.bridge import PromptBridge, PromptRequest
from audio_from_video.gui.qt_bridge import QtPromptAdapter
from audio_from_video.gui.workers import (
    CacheWorker,
    ExecuteWorker,
    PlanWorker,
    PreflightWorker,
)
from audio_from_video.models import PlanAction, PlanItem, RunSummary, TrackResult
from audio_from_video.session import ServicesFactory, create_online_services

if TYPE_CHECKING:
    from audio_from_video.config import AppConfig
    from audio_from_video.session import RunOptions

logger = logging.getLogger(__name__)

SETTINGS_ORGANIZATION = "audio-from-video"
SETTINGS_APPLICATION = "afv-gui"
_OFFLINE_SETTING_KEY = "startup/offline"
_LOG_LEVEL_SETTING_KEY = "log/level"
_DEFAULT_LOG_LEVEL = "INFO"
_CACHE_BUSY_MESSAGE = "Wait for the active task to finish before accessing the cache."
_CACHE_ACTION_PRIORITY: dict[str, int] = {
    "stats": 1,
    "clear_expired": 2,
    "clear_all": 3,
}


def _open_qsettings() -> QSettings:
    """Open the GUI ``QSettings`` store, honoring ``QSettings.defaultFormat()``.

    On macOS with PySide6 6.12, the two-argument ``QSettings(org, app)``
    constructor always opens ``NativeFormat``. Passing
    ``QSettings.defaultFormat()`` and ``UserScope`` explicitly preserves native
    storage in production while allowing tests to isolate settings via
    ``setDefaultFormat(IniFormat)`` and ``setPath(...)``.
    """
    return QSettings(
        QSettings.defaultFormat(),
        QSettings.Scope.UserScope,
        SETTINGS_ORGANIZATION,
        SETTINGS_APPLICATION,
    )


def _read_offline_setting() -> bool:
    """Read ``startup/offline`` from ``QSettings`` without touching real config files."""
    settings = _open_qsettings()
    raw = settings.value(_OFFLINE_SETTING_KEY, False, type=bool)
    return bool(raw)


def _read_log_level_setting() -> str:
    """Read normalized ``log/level`` from ``QSettings`` without touching real config files."""
    settings = _open_qsettings()
    raw = settings.value(_LOG_LEVEL_SETTING_KEY, _DEFAULT_LOG_LEVEL)
    if raw is None:
        return _DEFAULT_LOG_LEVEL
    cleaned = str(raw).strip().upper()
    return cleaned or _DEFAULT_LOG_LEVEL


def _write_offline_setting(offline: bool) -> None:
    """Persist ``startup/offline`` to ``QSettings`` and flush immediately."""
    settings = _open_qsettings()
    settings.setValue(_OFFLINE_SETTING_KEY, bool(offline))
    settings.sync()


def _extract_raw_binary_path(
    table: Mapping[str, object],
    primary_key: str,
    legacy_key: str,
) -> str | None:
    """Return a clean executable path string from a raw ``[binaries]`` TOML table.

    Mirrors ``load_config`` key precedence (``primary_key`` before
    ``legacy_key``) and ignores blank, non-string, or null-byte values so
    invalid entries fall back safely without raising.
    """
    for key in (primary_key, legacy_key):
        if key not in table:
            continue
        value = table[key]
        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        if not cleaned:
            continue
        if "\x00" in cleaned:
            return None
        return cleaned
    return None


@dataclass(frozen=True, slots=True)
class FallbackConfigState:
    """Read-only snapshot describing the active offline fallback ``AppConfig``."""

    config: AppConfig
    from_file: bool
    file_error: str | None


def _config_run_defaults_key(cfg: AppConfig) -> tuple[object, ...]:
    """Return a snapshot of the ``AppConfig`` fields that drive ``InputsScreen`` defaults."""
    return (
        cfg.audio.quality,
        cfg.audio.copy_mp3,
        cfg.output.output_dir,
        cfg.output.flat,
        cfg.output.on_conflict,
        cfg.execution.jobs,
    )


def build_fallback_config(
    config_path: Path | None,
    *,
    env: Mapping[str, str] | None = None,
    load_config_fn: Callable[..., AppConfig] | None = None,
) -> tuple[AppConfig, bool, str | None]:
    """Build a fallback ``AppConfig`` preserving valid ``config.toml`` or raw ``[binaries]``.

    Returns ``(config, from_file, file_error)``.

    When full ``load_config()`` fails during preflight (for example, because
    ``AFV_MB_CONTACT`` or ``AFV_MB_USER_AGENT`` is invalid, or because
    ``config.toml`` contains an invalid value), ``RunController._config``
    remains ``None``. First try reading ``config.toml`` with ``env={}`` so a
    valid file keeps its ``[audio]``, ``[output]``, ``[execution]``,
    ``[matching]``, and ``[binaries]`` defaults for offline use instead of
    discarding them. Loaded ``[musicbrainz]`` credentials are cleared so this
    fallback never carries them into online mode without re-validating
    environment overrides. If ``config.toml`` itself is unreadable or invalid,
    fall back to built-in defaults plus raw ``[binaries]`` paths parsed via
    ``tomllib``. Valid ``AFV_FFMPEG_PATH``, ``AFV_FFPROBE_PATH``, and
    ``AFV_OUTPUT_DIR`` environment overrides are applied cleanly on top.
    """
    active_env = os.environ if env is None else env
    loader = _config_mod.load_config if load_config_fn is None else load_config_fn
    cfg: AppConfig | None = None
    from_file = False
    file_error: str | None = None
    has_config_file = False

    if config_path is not None:
        try:
            has_config_file = config_path.is_file()
        except OSError as exc:
            file_error = str(exc)

    if file_error is None and config_path is not None and has_config_file:
        try:
            cfg = loader(config_path=config_path, env={})
            from_file = True
        except (_config_mod.ConfigError, OSError) as exc:
            file_error = str(exc)
            cfg = None

    if cfg is not None:
        # Never carry loaded file credentials across a failed preflight;
        # online runs must pass full validation first.
        cfg.musicbrainz.contact = None
        cfg.musicbrainz.user_agent_override = None
        if cfg.binaries.ffmpeg_path and "\x00" in cfg.binaries.ffmpeg_path:
            cfg.binaries.ffmpeg_path = None
        if cfg.binaries.ffprobe_path and "\x00" in cfg.binaries.ffprobe_path:
            cfg.binaries.ffprobe_path = None
    else:
        cfg = (
            _config_mod.AppConfig(config_path=config_path)
            if config_path is not None
            else _config_mod.AppConfig()
        )
        raw: object = None
        if config_path is not None and has_config_file:
            try:
                with config_path.open("rb") as fh:
                    raw = tomllib.load(fh)
            except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
                raw = None

        if isinstance(raw, dict):
            binaries_tbl = raw.get("binaries")
            if isinstance(binaries_tbl, dict):
                cfg.binaries.ffmpeg_path = _extract_raw_binary_path(
                    binaries_tbl, "ffmpeg", "ffmpeg_path"
                )
                cfg.binaries.ffprobe_path = _extract_raw_binary_path(
                    binaries_tbl, "ffprobe", "ffprobe_path"
                )

    env_ffmpeg = active_env.get("AFV_FFMPEG_PATH", "").strip()
    if env_ffmpeg and "\x00" not in env_ffmpeg:
        cfg.binaries.ffmpeg_path = env_ffmpeg
    env_ffprobe = active_env.get("AFV_FFPROBE_PATH", "").strip()
    if env_ffprobe and "\x00" not in env_ffprobe:
        cfg.binaries.ffprobe_path = env_ffprobe
    env_output_dir = active_env.get("AFV_OUTPUT_DIR", "").strip()
    if env_output_dir and "\x00" not in env_output_dir:
        cfg.output.output_dir = env_output_dir
    return cfg, from_file, file_error


def _fallback_config_with_raw_binaries(config_path: Path) -> AppConfig:
    """Return the fallback ``AppConfig`` built by :func:`build_fallback_config`."""
    return build_fallback_config(config_path)[0]


class ControllerState(StrEnum):
    """Top-level workflow states exposed by ``RunController``."""

    PREFLIGHT = "preflight"
    SETUP = "setup"
    INPUTS = "inputs"
    PLANNING = "planning"
    REVIEW = "review"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUMMARY = "summary"


@dataclass(frozen=True, slots=True)
class BinariesStatus:
    """Result of inspecting FFmpeg, FFprobe, and ``libmp3lame`` encoder support."""

    ffmpeg: BinaryResolution
    ffprobe: BinaryResolution
    has_libmp3lame: bool


class RunController(QObject):
    """UI-thread coordinator for preflight, planning, execution, cache actions, and shutdown.

    All signals are emitted on the UI thread. No command method blocks the UI
    thread. ``QSettings`` UI keys (organization ``audio-from-video``, application
    ``afv-gui``) are ``window/geometry``, ``window/state``, ``inputs/last_folder``,
    ``output/last_folder``, ``review/splitter``, ``log/level``, and
    ``startup/offline``.

    For Stage 3 ``MainWindow`` integration: on a window close request, call
    ``hide()`` on the window and invoke ``shutdown()``; if any worker is still
    running, ``closeEvent`` should ignore the event and allow ``shutdown()`` to
    wait asynchronously for every worker's ``finished`` signal, close any open
    ``RunSession`` once, and then call ``QCoreApplication.quit()``.
    """

    state_changed = Signal(object)  # ControllerState
    event_received = Signal(object)  # TrackEvent
    stats_changed = Signal(object)  # LookupStats
    plan_ready = Signal(object)  # list[PlanItem]
    row_changed = Signal(int)  # 0-based index whose PlanItem changed
    summary_ready = Signal(object, int, int)  # RunSummary, cancelled_count, not_run_count
    prompt_requested = Signal(object)  # PromptRequest
    prompt_closed = Signal(int)  # request_id withdrawn by a cancel
    notice = Signal(str)
    banner = Signal(str)  # Batch and cache warnings
    input_error = Signal(str, str)  # kind, message
    setup_error = Signal(str)
    binaries_status = Signal(object)  # BinariesStatus
    cache_stats_ready = Signal(object)  # list[NamespaceStats]
    cache_cleared = Signal(int)  # entries removed
    settings_saved = Signal(object)  # AppConfig
    error = Signal(str, str)  # title, details

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        services_factory: ServicesFactory = create_online_services,
        cache_dir: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._services_factory = services_factory
        self._cache_dir = cache_dir
        self._state = ControllerState.PREFLIGHT
        self._config: AppConfig | None = None
        self._fallback_state: tuple[AppConfig, bool, str | None] | None = None
        self._fallback_defaults_key: tuple[object, ...] | None = None
        self._plan_items: list[PlanItem] = []
        self._original_actions: list[PlanAction] = []
        self._output_root: Path | None = None
        self._offline_default = _read_offline_setting()

        self._binaries_info: BinariesStatus | None = None
        self._session: Any | None = None
        self._summary: RunSummary | None = None
        self._token: CancelToken | None = None

        # Active prompt routing during Stage 1 planning.
        self._prompt_adapter: QtPromptAdapter | None = None
        self._prompt_bridge: PromptBridge | None = None
        self._active_prompt_id: int | None = None

        # Preflight worker tracking.
        self._preflight_thread: QThread | None = None
        self._preflight_worker: PreflightWorker | None = None
        self._preflight_token: CancelToken | None = None
        self._preflight_recheck_only = False
        self._preflight_error: tuple[str, str] | None = None
        self._pending_binary_recheck = False

        # Plan worker tracking.
        self._plan_thread: QThread | None = None
        self._plan_worker: PlanWorker | None = None
        self._plan_result_items: list[PlanItem] | None = None
        self._plan_input_error: tuple[str, str] | None = None
        self._plan_failed: tuple[str, str] | None = None
        self._plan_was_cancelled = False

        # Execute worker tracking.
        self._execute_thread: QThread | None = None
        self._execute_worker: ExecuteWorker | None = None
        self._execute_summary: RunSummary | None = None
        self._execute_failed: tuple[str, str] | None = None
        self._execute_was_cancelled = False
        self._partial_results: list[TrackResult] = []
        self._cancelled_count = 0
        self._not_run_count = 0

        # Cache worker tracking.
        self._cache_thread: QThread | None = None
        self._cache_worker: CacheWorker | None = None
        self._cache_token: CancelToken | None = None
        self._cache_stats_result: object | None = None
        self._cache_cleared_result: int | None = None
        self._cache_error: tuple[str, str] | None = None
        self._pending_cache_action: str | None = None

        # Settings broadcast guard for deferred QSettings writes inside slots.
        self._emitting_settings_saved = False
        self._last_emitted_ui_prefs: tuple[bool, str] | None = None

        # Shutdown bookkeeping.
        self._cancel_source: ControllerState | None = None
        self._shutting_down = False
        self._shutdown_completed = False

    @property
    def state(self) -> ControllerState:
        """Return the current workflow state."""
        return self._state

    @property
    def config(self) -> AppConfig | None:
        """Return the loaded application configuration, if available."""
        return self._config

    @property
    def effective_config(self) -> AppConfig:
        """Return the active ``AppConfig`` used for planning (loaded or fallback)."""
        if self._config is not None:
            return self._config
        return self._ensure_fallback_state()[0]

    @property
    def fallback_config_state(self) -> FallbackConfigState | None:
        """Return the cached fallback snapshot when ``config`` is ``None``, else ``None``."""
        if self._config is not None or self._fallback_state is None:
            return None
        cfg, from_file, file_error = self._fallback_state
        return FallbackConfigState(config=cfg, from_file=from_file, file_error=file_error)

    @property
    def plan_items(self) -> list[PlanItem]:
        """Return the planned items for the current batch."""
        return self._plan_items

    @property
    def output_root(self) -> Path | None:
        """Return the resolved output root directory for the current plan."""
        return self._output_root

    @property
    def offline_default(self) -> bool:
        """Return whether startup defaults to offline mode (``startup/offline``)."""
        return self._offline_default

    # -- Internal helpers ------------------------------------------------------

    def _invalidate_fallback_state(self) -> None:
        """Drop any cached fallback ``AppConfig`` snapshot so the next query reloads disk."""
        self._fallback_state = None
        self._fallback_defaults_key = None

    def _ensure_fallback_state(self) -> tuple[AppConfig, bool, str | None]:
        """Return the cached ``(config, from_file, file_error)`` fallback tuple."""
        if self._fallback_state is None:
            state = build_fallback_config(self._config_path())
            self._fallback_state = state
            self._fallback_defaults_key = _config_run_defaults_key(state[0])
        return self._fallback_state

    def _set_state(self, new_state: ControllerState) -> None:
        if self._state == new_state:
            return
        prev_state = self._state
        self._state = new_state
        if new_state == ControllerState.PREFLIGHT:
            self._invalidate_fallback_state()
        elif (
            new_state == ControllerState.INPUTS
            and self._config is None
            and prev_state in (ControllerState.PREFLIGHT, ControllerState.SETUP)
        ):
            self._invalidate_fallback_state()
            self._ensure_fallback_state()
        self.state_changed.emit(new_state)

    def _emit_settings_saved(self, config: AppConfig | None) -> None:
        """Emit ``settings_saved`` and re-emit at most once if a slot changed QSettings.

        Deferred ``QSettings`` writers (such as ``SettingsScreen._on_settings_saved``)
        persist ``startup/offline`` and ``log/level`` inside the ``settings_saved``
        signal so failed file writes never mutate preferences. Re-emitting at most
        once after any mutation ensures earlier-connected listeners (such as
        ``InputsScreen`` and ``MainWindow``) observe refreshed ``offline_default``
        and ``log/level`` values without exiting ``SETUP`` prematurely, launching a
        duplicate ``PreflightWorker``, or recursing.
        """
        offline = _read_offline_setting()
        log_level = _read_log_level_setting()
        self._offline_default = offline
        self._last_emitted_ui_prefs = (offline, log_level)

        already_emitting = self._emitting_settings_saved
        self._emitting_settings_saved = True
        try:
            self.settings_saved.emit(config)

            # Re-read in case a slot persisted deferred QSettings during settings_saved.
            updated_offline = _read_offline_setting()
            updated_log_level = _read_log_level_setting()
            self._offline_default = updated_offline

            if (
                not already_emitting
                and not self._shutting_down
                and (updated_offline, updated_log_level) != self._last_emitted_ui_prefs
            ):
                self._last_emitted_ui_prefs = (updated_offline, updated_log_level)
                self.settings_saved.emit(config)
                # Refresh once more in case a slot changed startup/offline during the
                # re-emit, without triggering a third broadcast.
                self._offline_default = _read_offline_setting()
        finally:
            self._emitting_settings_saved = already_emitting

    def _has_active_workers(self) -> bool:
        return any(
            thread is not None
            for thread in (
                self._preflight_thread,
                self._plan_thread,
                self._execute_thread,
                self._cache_thread,
            )
        )

    def _is_batch_active(self) -> bool:
        return (
            self._state
            in (
                ControllerState.PLANNING,
                ControllerState.RUNNING,
                ControllerState.CANCELLING,
            )
            or self._plan_thread is not None
            or self._execute_thread is not None
        )

    def _is_cache_batch_blocked(self) -> bool:
        """Return ``True`` when an active batch, worker, or open session blocks cache access."""
        return (
            self._state
            in (
                ControllerState.PLANNING,
                ControllerState.REVIEW,
                ControllerState.RUNNING,
                ControllerState.CANCELLING,
            )
            or self._plan_thread is not None
            or self._execute_thread is not None
            or self._session is not None
        )

    def _cache_operation_allowed(self) -> bool:
        if self._shutting_down:
            return False
        if self._is_cache_batch_blocked():
            return False
        return self._preflight_thread is None

    def _queue_cache_action(self, action: str) -> None:
        current = self._pending_cache_action
        if current is None or _CACHE_ACTION_PRIORITY[action] > _CACHE_ACTION_PRIORITY[current]:
            self._pending_cache_action = action

    def _drain_pending_cache_action(self) -> None:
        """Launch the queued cache action once preflight and cache workers are idle."""
        if self._shutting_down:
            self._pending_cache_action = None
            return
        if self._pending_cache_action is None:
            return
        if self._preflight_thread is not None or self._cache_thread is not None:
            return
        queued_action = self._pending_cache_action
        self._pending_cache_action = None
        if self._is_cache_batch_blocked():
            self.error.emit("Cache busy", _CACHE_BUSY_MESSAGE)
            return
        self._launch_cache_worker(queued_action)

    def _launch_cache_worker(self, action: str) -> None:
        thread = QThread(self)
        token = CancelToken()
        if action == "stats":
            worker = CacheWorker(
                operation="stats",
                cache_dir=self._effective_cache_dir(),
                token=token,
            )
        else:
            worker = CacheWorker(
                operation="clear",
                cache_dir=self._effective_cache_dir(),
                expired_only=action == "clear_expired",
                token=token,
            )
        worker.moveToThread(thread)
        self._cache_thread = thread
        self._cache_worker = worker
        self._cache_token = token
        self._cache_stats_result = None
        self._cache_cleared_result = None
        self._cache_error = None

        thread.started.connect(worker.run, Qt.ConnectionType.QueuedConnection)
        if action == "stats":
            worker.stats_ready.connect(self._on_cache_stats_ready)
        else:
            worker.cleared.connect(self._on_cache_cleared)
        worker.failed.connect(self._on_cache_failed)
        worker._relay.finished.connect(thread.quit)
        thread.finished.connect(self._on_cache_thread_finished)
        thread.start()

    def _binaries_ready(self) -> bool:
        status = self._binaries_info
        return (
            status is not None
            and status.ffmpeg.found
            and status.ffprobe.found
            and status.has_libmp3lame
        )

    def _missing_binaries_message(self) -> str:
        status = self._binaries_info
        if status is None:
            return "FFmpeg and FFprobe have not been checked yet."
        problems: list[str] = []
        if not status.ffmpeg.found:
            problems.append(status.ffmpeg.error or "ffmpeg was not found.")
        if not status.ffprobe.found:
            problems.append(status.ffprobe.error or "ffprobe was not found.")
        if status.ffmpeg.found and not status.has_libmp3lame:
            problems.append("FFmpeg does not include the required libmp3lame encoder.")
        return " ".join(problems) if problems else "Required FFmpeg binaries are missing."

    def _check_contact(self, cfg: AppConfig | None) -> tuple[bool, str | None, bool]:
        """Return ``(is_ready, error_message, is_missing_only)`` for online lookups."""
        if cfg is None:
            return False, "Configuration has not been loaded yet.", True
        try:
            _config_mod.resolve_mb_user_agent(cfg)
            return True, None, False
        except _config_mod.MissingContactError as exc:
            return False, str(exc), True
        except _config_mod.ConfigError as exc:
            return False, str(exc), False

    def _config_path(self) -> Path:
        if self._config is not None:
            return self._config.config_path
        return _config_mod.default_config_path()

    def _fallback_config(self) -> AppConfig:
        """Build a fallback ``AppConfig`` preserving valid ``config.toml`` or raw ``[binaries]``."""
        return _fallback_config_with_raw_binaries(self._config_path())

    def _effective_cache_dir(self) -> Path:
        if self._cache_dir is not None:
            return self._cache_dir
        return _config_mod.default_cache_dir()

    def _close_session(self) -> None:
        """Close the currently open ``RunSession`` strictly once."""
        session = self._session
        if session is None:
            return
        self._session = None
        try:
            session.close()
        except Exception:
            logger.warning("Error closing RunSession during cleanup.", exc_info=True)

    def _withdraw_active_prompt(self) -> None:
        """Emit ``prompt_closed`` if a prompt is awaiting an answer."""
        if self._active_prompt_id is None:
            return
        request_id = self._active_prompt_id
        self._active_prompt_id = None
        self.prompt_closed.emit(request_id)

    def _cleanup_prompt_bridge(self) -> None:
        """Tear down ``QtPromptAdapter`` and ``PromptBridge`` once planning stops."""
        adapter = self._prompt_adapter
        self._prompt_adapter = None
        self._prompt_bridge = None
        self._active_prompt_id = None
        if adapter is not None:
            adapter._deactivate()
            adapter.deleteLater()

    def _quit_application(self) -> None:
        """Quit the active ``QCoreApplication`` event loop after shutdown cleanup."""
        app = QCoreApplication.instance()
        if app is not None:
            app.quit()

    def _maybe_finish_shutdown(self) -> None:
        """Close any open session and quit once every active worker has finished."""
        if not self._shutting_down or self._shutdown_completed:
            return
        if self._has_active_workers():
            return
        self._cleanup_prompt_bridge()
        self._close_session()
        self._shutdown_completed = True
        self._quit_application()

    def _launch_preflight_worker(
        self,
        *,
        binaries_only: bool,
        config: AppConfig | None = None,
    ) -> None:
        thread = QThread(self)
        token = CancelToken()
        worker = PreflightWorker(
            config=config,
            binaries_only=binaries_only,
            token=token,
        )
        worker.moveToThread(thread)

        self._preflight_thread = thread
        self._preflight_worker = worker
        self._preflight_token = token
        self._preflight_recheck_only = binaries_only
        self._preflight_error = None

        thread.started.connect(worker.run, Qt.ConnectionType.QueuedConnection)
        worker.config_loaded.connect(self._on_preflight_config_loaded)
        worker.binaries_checked.connect(self._on_preflight_binaries_checked)
        worker.failed.connect(self._on_preflight_failed)
        worker._relay.finished.connect(thread.quit)
        thread.finished.connect(self._on_preflight_thread_finished)
        thread.start()

    # -- Public commands (frozen section 5.5) ----------------------------------

    def start_preflight(self) -> None:
        """Load configuration and inspect FFmpeg binaries on a worker thread."""
        if self._shutting_down or self._preflight_thread is not None or self._is_batch_active():
            return
        self._offline_default = _read_offline_setting()
        self._pending_binary_recheck = False
        self._config = None
        self._invalidate_fallback_state()
        self._binaries_info = None
        self._preflight_error = None
        self._set_state(ControllerState.PREFLIGHT)
        self._launch_preflight_worker(binaries_only=False)

    def save_contact(self, contact: str) -> None:
        """Validate and save the MusicBrainz contact; report via ``setup_error`` or state change."""
        if self._shutting_down or self._is_batch_active():
            return
        try:
            validated = _config_mod.validate_mb_contact(contact)
        except _config_mod.ConfigError as exc:
            self.setup_error.emit(str(exc))
            return

        target_path = self._config_path()
        try:
            if not target_path.exists():
                try:
                    _config_mod.write_config_template(target_path, contact=validated)
                except FileExistsError:
                    _config_mod.save_mb_contact(validated, target_path)
            else:
                _config_mod.save_mb_contact(validated, target_path)
        except _config_mod.ConfigError as exc:
            self.setup_error.emit(str(exc))
            return
        except OSError as exc:
            self.setup_error.emit(
                f"Could not save contact to {target_path}: {exc}. "
                "Set the AFV_MB_CONTACT environment variable instead, or use offline mode."
            )
            return

        try:
            reloaded = _config_mod.load_config(config_path=target_path)
        except _config_mod.ConfigError as exc:
            self.setup_error.emit(str(exc))
            return

        prev_config = self._config
        self._config = reloaded
        self._invalidate_fallback_state()
        _write_offline_setting(False)
        self._offline_default = False
        # Report via setup_error or state_changed only (never settings_saved) so
        # a SETUP detour preserves any edited per-run options on InputsScreen.

        recovered_custom_binaries = prev_config is None and (
            reloaded.binaries.ffmpeg_path is not None or reloaded.binaries.ffprobe_path is not None
        )
        if self._binaries_ready() and not recovered_custom_binaries:
            if self._state == ControllerState.SETUP:
                self._set_state(ControllerState.INPUTS)
        elif self._binaries_info is None or recovered_custom_binaries:
            if self._preflight_thread is not None:
                self._pending_binary_recheck = True
            else:
                self._launch_preflight_worker(binaries_only=True, config=reloaded)
        else:
            self.setup_error.emit(self._missing_binaries_message())

    def use_offline_only(self) -> None:
        """Enable offline-only mode and persist ``startup/offline = true`` in ``QSettings``."""
        if self._shutting_down or self._is_batch_active():
            return
        _write_offline_setting(True)
        self._offline_default = True
        if self._binaries_ready():
            if self._state == ControllerState.SETUP:
                self._set_state(ControllerState.INPUTS)
        elif self._binaries_info is None:
            if self._preflight_thread is None:
                cfg = self._config if self._config is not None else self._fallback_config()
                self._launch_preflight_worker(binaries_only=True, config=cfg)
        else:
            self.setup_error.emit(self._missing_binaries_message())

    def set_ffmpeg_path(self, path: Path) -> None:
        """Persist a custom FFmpeg path with ``save_settings`` and re-check binaries."""
        if self._shutting_down or self._is_batch_active():
            return
        target_path = self._config_path()
        try:
            _config_mod.save_settings({"binaries.ffmpeg_path": path}, config_path=target_path)
            reloaded = _config_mod.load_config(config_path=target_path)
        except (_config_mod.ConfigError, ValueError, OSError) as exc:
            self.setup_error.emit(str(exc))
            return

        self._config = reloaded
        self._invalidate_fallback_state()
        # Report via binaries_status, setup_error, or state_changed only (never
        # settings_saved) so a SETUP detour preserves per-run options on InputsScreen.
        if self._preflight_thread is not None:
            self._pending_binary_recheck = True
        else:
            self._launch_preflight_worker(binaries_only=True, config=reloaded)

    def start_plan(self, options: RunOptions) -> None:
        """Begin Stage 1 discovery and planning on a worker thread.

        When planning cannot start because the MusicBrainz contact or FFmpeg is
        missing (``kind`` is ``"missing_contact"`` or ``"missing_ffmpeg"``),
        this emits both ``input_error(kind, message)`` and
        ``setup_error(message)``, transitions to ``ControllerState.SETUP``, and
        does not start a worker so the Setup screen can display the reason.
        """
        if self._shutting_down or self._state != ControllerState.INPUTS:
            return
        if self._plan_thread is not None or self._execute_thread is not None:
            return
        if (
            self._preflight_thread is not None
            or self._cache_thread is not None
            or self._pending_cache_action is not None
        ):
            self.input_error.emit(
                "busy",
                "Please wait for the background check or cache operation to finish.",
            )
            return

        user_agent: str | None = None
        if not options.offline:
            reloaded_from_fallback = False
            prev_defaults_key = self._fallback_defaults_key
            try:
                if self._config is None:
                    # Re-validate the full config file and AFV_* env overrides so
                    # online mode stays blocked after a failed preflight, and adopt
                    # the reloaded config when the file or environment is fixed.
                    cfg = _config_mod.load_config(config_path=self._config_path())
                    self._config = cfg
                    self._invalidate_fallback_state()
                    reloaded_from_fallback = True
                else:
                    cfg = self._config
                user_agent = _config_mod.resolve_mb_user_agent(cfg)
            except _config_mod.ConfigError as exc:
                msg = str(exc)
                self.input_error.emit("missing_contact", msg)
                self.setup_error.emit(msg)
                self._set_state(ControllerState.SETUP)
                return
            if reloaded_from_fallback:
                self._emit_settings_saved(cfg)
                if prev_defaults_key is not None and _config_run_defaults_key(cfg) != (
                    prev_defaults_key
                ):
                    self.input_error.emit(
                        "config_changed",
                        "config.toml changed outside the app; defaults on this screen "
                        "were updated. Review your options and click Scan and plan again.",
                    )
                    return
        elif self._config is not None:
            cfg = self._config
        else:
            prev_defaults_key = self._fallback_defaults_key
            fresh_state = build_fallback_config(self._config_path())
            cfg = fresh_state[0]
            new_defaults_key = _config_run_defaults_key(cfg)
            self._fallback_state = fresh_state
            self._fallback_defaults_key = new_defaults_key
            if prev_defaults_key is not None and new_defaults_key != prev_defaults_key:
                self._emit_settings_saved(None)
                self.input_error.emit(
                    "config_changed",
                    "config.toml changed outside the app; defaults on this screen "
                    "were updated. Review your options and click Scan and plan again.",
                )
                return

        if not self._binaries_ready():
            msg = self._missing_binaries_message()
            self.input_error.emit("missing_ffmpeg", msg)
            self.setup_error.emit(msg)
            self._set_state(ControllerState.SETUP)
            return

        status = self._binaries_info
        assert status is not None
        assert status.ffmpeg.path is not None and status.ffprobe.path is not None
        binaries = ResolvedBinaries(ffmpeg=status.ffmpeg.path, ffprobe=status.ffprobe.path)

        # Ensure any stale state from a previous batch is cleaned up.
        self._close_session()
        self._cleanup_prompt_bridge()
        self._plan_items = []
        self._original_actions = []
        self._output_root = None
        self._summary = None
        self._plan_result_items = None
        self._plan_input_error = None
        self._plan_failed = None
        self._plan_was_cancelled = False
        self._partial_results = []
        self._cancelled_count = 0
        self._not_run_count = 0
        self._cancel_source = None

        token = CancelToken()
        self._token = token

        adapter = QtPromptAdapter(self)
        adapter.prompt_requested.connect(self._on_prompt_requested)
        adapter.notice_received.connect(self._on_prompt_notice)
        bridge = PromptBridge(dispatch=adapter.dispatch, notify=adapter.notify, token=token)
        self._prompt_adapter = adapter
        self._prompt_bridge = bridge

        self._set_state(ControllerState.PLANNING)

        thread = QThread(self)
        worker = PlanWorker(
            options=options,
            config=cfg,
            binaries=binaries,
            token=token,
            user_agent=user_agent,
            cache_dir=self._effective_cache_dir(),
            services_factory=self._services_factory,
            prompt_bridge=bridge,
            defer_session_close=True,
        )
        worker.moveToThread(thread)
        self._plan_thread = thread
        self._plan_worker = worker

        thread.started.connect(worker.run, Qt.ConnectionType.QueuedConnection)
        worker.event.connect(self._on_plan_event)
        worker.stats.connect(self._on_plan_stats)
        worker.session_opened.connect(self._on_plan_session_opened)
        worker.plan_ready.connect(self._on_plan_ready)
        worker.input_error.connect(self._on_plan_input_error)
        worker.cancelled.connect(self._on_plan_cancelled)
        worker.failed.connect(self._on_plan_failed)
        worker._relay.output_root_ready.connect(self._on_plan_output_root_ready)
        worker._relay.batch_warning.connect(self._on_plan_batch_warning)
        worker._relay.finished.connect(thread.quit)
        thread.finished.connect(self._on_plan_thread_finished)
        thread.start()

    def cancel(self) -> None:
        """Cancel the active planning or execution worker and enter ``CANCELLING``."""
        if self._state not in (ControllerState.PLANNING, ControllerState.RUNNING):
            return
        self._cancel_source = self._state
        self._state = ControllerState.CANCELLING

        if self._token is not None:
            self._token.cancel()
        self._withdraw_active_prompt()
        self.state_changed.emit(ControllerState.CANCELLING)

    def back(self) -> None:
        """Close any active session and return from Review to Inputs."""
        if self._shutting_down:
            return
        if self._state == ControllerState.PLANNING:
            self.cancel()
            return
        if self._state != ControllerState.REVIEW:
            return
        self._close_session()
        self._plan_items = []
        self._original_actions = []
        self._output_root = None
        self._token = None
        self._set_state(ControllerState.INPUTS)

    def set_row_skipped(self, index: int, skipped: bool) -> None:
        """Toggle a row between ``SKIP_USER`` and its original action; emit ``row_changed``."""
        if self._state != ControllerState.REVIEW:
            return
        if not 0 <= index < len(self._plan_items):
            logger.warning(
                "Ignoring set_row_skipped for out-of-range index %d (plan has %d item(s)).",
                index,
                len(self._plan_items),
            )
            return
        target_action = PlanAction.SKIP_USER if skipped else self._original_actions[index]
        self._plan_items[index].action = target_action
        self.row_changed.emit(index)

    def start_run(self) -> None:
        """Begin Stage 2 extraction for the current plan."""
        if self._shutting_down or self._state != ControllerState.REVIEW:
            return
        if self._execute_thread is not None or self._plan_thread is not None:
            return
        if self._session is None or self._output_root is None:
            self.error.emit(
                "Cannot start extraction",
                "No open session is available for this plan. Please re-plan the batch.",
            )
            self._close_session()
            self._set_state(ControllerState.INPUTS)
            return

        self._summary = None
        self._execute_summary = None
        self._execute_failed = None
        self._execute_was_cancelled = False
        self._partial_results = []
        self._cancelled_count = 0
        self._not_run_count = 0
        self._cancel_source = None

        self._set_state(ControllerState.RUNNING)

        thread = QThread(self)
        worker = ExecuteWorker(
            session=self._session,
            plan_items=list(self._plan_items),
            output_root=self._output_root,
            token=self._token,
        )
        worker.moveToThread(thread)
        self._execute_thread = thread
        self._execute_worker = worker

        thread.started.connect(worker.run, Qt.ConnectionType.QueuedConnection)
        worker.event.connect(self._on_execute_event)
        worker.result.connect(self._on_execute_result)
        worker.summary_ready.connect(self._on_execute_summary_ready)
        worker.cancelled.connect(self._on_execute_cancelled)
        worker.failed.connect(self._on_execute_failed)
        worker._relay.finished.connect(thread.quit)
        thread.finished.connect(self._on_execute_thread_finished)
        thread.start()

    def answer_prompt(self, request_id: int, value: object) -> None:
        """Deliver a dialog response back to the active ``PromptBridge``."""
        bridge = self._prompt_bridge
        if bridge is None:
            return
        accepted = bridge.answer(request_id, value)
        if accepted and self._active_prompt_id == request_id:
            self._active_prompt_id = None

    def new_batch(self) -> None:
        """Close the current session and return from Summary to Inputs."""
        if self._shutting_down or self._state != ControllerState.SUMMARY:
            return
        self._close_session()
        self._plan_items = []
        self._original_actions = []
        self._output_root = None
        self._summary = None
        self._token = None
        self._partial_results = []
        self._cancelled_count = 0
        self._not_run_count = 0
        self._set_state(ControllerState.INPUTS)

    def save_settings(self, updates: Mapping[str, object]) -> None:
        """Persist updated configuration keys and emit ``settings_saved``."""
        if self._shutting_down:
            return
        if self._is_batch_active():
            self.error.emit(
                "Busy",
                "Settings cannot be saved while a batch is planning, extracting, or cancelling.",
            )
            return

        if not updates and self._emitting_settings_saved:
            # Safety guard for any nested save_settings({}) invoked inside a
            # settings_saved slot: refresh preferences without launching a
            # duplicate PreflightWorker, touching config.toml, or changing state.
            self._offline_default = _read_offline_setting()
            return

        target_path = self._config_path()
        prev_config = self._config
        if not updates:
            # Never create or touch config.toml for UI-only QSettings changes.
            reloaded: AppConfig | None = self._config
            if reloaded is None:
                try:
                    reloaded = _config_mod.load_config(config_path=target_path)
                except (_config_mod.ConfigError, OSError):
                    reloaded = None
                else:
                    self._config = reloaded

            if reloaded is None:
                self._invalidate_fallback_state()
                self._ensure_fallback_state()
            else:
                self._invalidate_fallback_state()
            self._emit_settings_saved(reloaded)
            if self._shutting_down:
                return

            recovered_custom_binaries = (
                prev_config is None
                and reloaded is not None
                and (
                    reloaded.binaries.ffmpeg_path is not None
                    or reloaded.binaries.ffprobe_path is not None
                )
            )
            if recovered_custom_binaries:
                if self._preflight_thread is not None:
                    self._pending_binary_recheck = True
                else:
                    self._launch_preflight_worker(binaries_only=True, config=reloaded)
                return

            if self._state == ControllerState.SETUP:
                contact_ok, _err, _missing = self._check_contact(reloaded)
                if self._binaries_ready() and (self._offline_default or contact_ok):
                    self._set_state(ControllerState.INPUTS)
                elif (
                    self._binaries_info is None
                    and self._preflight_thread is None
                    and (self._offline_default or reloaded is not None)
                ):
                    cfg_for_check = reloaded if reloaded is not None else self._fallback_config()
                    self._launch_preflight_worker(binaries_only=True, config=cfg_for_check)
            return

        try:
            _config_mod.save_settings(updates, config_path=target_path)
            reloaded_cfg = _config_mod.load_config(config_path=target_path)
        except (_config_mod.ConfigError, ValueError, OSError) as exc:
            message = str(exc)
            self.error.emit("Could not save settings", message)
            if self._state == ControllerState.SETUP:
                self.setup_error.emit(message)
            return

        self._config = reloaded_cfg
        self._invalidate_fallback_state()
        self._emit_settings_saved(reloaded_cfg)
        if self._shutting_down:
            return

        binaries_updated = "binaries.ffmpeg_path" in updates or "binaries.ffprobe_path" in updates
        recovered_custom_binaries = prev_config is None and (
            reloaded_cfg.binaries.ffmpeg_path is not None
            or reloaded_cfg.binaries.ffprobe_path is not None
        )
        if binaries_updated or recovered_custom_binaries:
            if self._preflight_thread is not None:
                self._pending_binary_recheck = True
            else:
                self._launch_preflight_worker(binaries_only=True, config=reloaded_cfg)
        elif self._state == ControllerState.SETUP:
            contact_ok, _err, _missing = self._check_contact(reloaded_cfg)
            if self._binaries_ready() and (self._offline_default or contact_ok):
                self._set_state(ControllerState.INPUTS)
            elif self._binaries_info is None and self._preflight_thread is None:
                self._launch_preflight_worker(binaries_only=True, config=reloaded_cfg)

    def request_cache_stats(self) -> None:
        """Inspect SQLite cache statistics on a worker thread."""
        if self._shutting_down:
            return
        if self._is_cache_batch_blocked():
            self.error.emit("Cache busy", _CACHE_BUSY_MESSAGE)
            return
        if self._preflight_thread is not None or self._cache_thread is not None:
            self._queue_cache_action("stats")
            return
        self._launch_cache_worker("stats")

    def clear_cache(self, *, expired_only: bool) -> None:
        """Clear cache entries on a worker thread and emit ``cache_cleared``."""
        if self._shutting_down:
            return
        if self._is_cache_batch_blocked():
            self.error.emit("Cache busy", _CACHE_BUSY_MESSAGE)
            return
        action = "clear_expired" if expired_only else "clear_all"
        if self._preflight_thread is not None or self._cache_thread is not None:
            self._queue_cache_action(action)
            return
        self._launch_cache_worker(action)

    def save_report(self, path: Path) -> None:
        """Write the latest run summary as a JSON report to ``path``."""
        if self._summary is None:
            self.error.emit(
                "No report available",
                "Complete a batch run before saving a JSON report.",
            )
            return
        target = Path(path)
        try:
            _runner_mod.write_json_report(self._summary, target)
        except OSError as exc:
            self.error.emit(
                "Could not save report",
                f"Failed to write report to {target}: {exc}",
            )

    def shutdown(self) -> None:
        """Stop any running worker in shutdown order (section 4.5) before exiting."""
        if self._shutting_down:
            return
        self._shutting_down = True
        self._pending_binary_recheck = False
        self._pending_cache_action = None

        if not self._has_active_workers():
            if self._token is not None:
                self._token.cancel()
            self._withdraw_active_prompt()
            self._maybe_finish_shutdown()
            return

        already_cancelling = self._state == ControllerState.CANCELLING
        self._state = ControllerState.CANCELLING

        # Step 1: cancel active token(s) so pending prompts fail and Stage 2 terminates FFmpeg.
        if self._token is not None:
            self._token.cancel()
        if self._preflight_token is not None:
            self._preflight_token.cancel()
        if self._cache_token is not None:
            self._cache_token.cancel()

        # Step 2: withdraw any open prompt dialog without answering it.
        self._withdraw_active_prompt()

        if not already_cancelling:
            self.state_changed.emit(ControllerState.CANCELLING)

    # -- Bound UI-thread slots for worker signals (Rule 2) ---------------------

    @Slot(object)
    def _on_preflight_config_loaded(self, cfg: object) -> None:
        self._config = cfg  # type: ignore[assignment]
        self._invalidate_fallback_state()

    @Slot(object)
    def _on_preflight_binaries_checked(self, status: object) -> None:
        self._binaries_info = status  # type: ignore[assignment]
        if not self._shutting_down:
            self.binaries_status.emit(status)

    @Slot(str, str)
    def _on_preflight_failed(self, title: str, details: str) -> None:
        self._preflight_error = (title, details)

    @Slot()
    def _on_preflight_thread_finished(self) -> None:
        thread = self._preflight_thread
        worker = self._preflight_worker
        recheck_only = self._preflight_recheck_only
        error_info = self._preflight_error
        self._preflight_thread = None
        self._preflight_worker = None
        self._preflight_token = None
        self._preflight_recheck_only = False
        self._preflight_error = None

        if thread is not None:
            thread.wait()
            thread.deleteLater()
        if worker is not None:
            worker.deleteLater()

        if self._shutting_down:
            self._pending_cache_action = None
            self._maybe_finish_shutdown()
            return

        if self._pending_binary_recheck:
            self._pending_binary_recheck = False
            cfg = self._config if self._config is not None else self._fallback_config()
            self._launch_preflight_worker(binaries_only=True, config=cfg)
            return

        if error_info is not None:
            title, details = error_info
            self.setup_error.emit(details)
            self.error.emit(title, details)
            self._set_state(ControllerState.SETUP)
            self._drain_pending_cache_action()
            return

        self._offline_default = _read_offline_setting()
        binaries_ok = self._binaries_ready()
        contact_ok, contact_error, is_missing_only = self._check_contact(self._config)

        if recheck_only:
            if self._state == ControllerState.SETUP:
                if binaries_ok and (self._offline_default or contact_ok):
                    self._set_state(ControllerState.INPUTS)
                elif not binaries_ok:
                    self.setup_error.emit(self._missing_binaries_message())
            self._drain_pending_cache_action()
            return

        if binaries_ok and (self._offline_default or contact_ok):
            self._set_state(ControllerState.INPUTS)
            self._drain_pending_cache_action()
            return

        if not self._offline_default and not contact_ok and not is_missing_only and contact_error:
            self.setup_error.emit(contact_error)
        self._set_state(ControllerState.SETUP)
        self._drain_pending_cache_action()

    @Slot(object)
    def _on_prompt_requested(self, request: object) -> None:
        if (
            self._shutting_down
            or self._state != ControllerState.PLANNING
            or (self._token is not None and self._token.is_cancelled)
        ):
            return
        if isinstance(request, PromptRequest):
            self._active_prompt_id = request.request_id
        self.prompt_requested.emit(request)

    @Slot(str)
    def _on_prompt_notice(self, message: str) -> None:
        if self._shutting_down:
            return
        if self._state in (ControllerState.PLANNING, ControllerState.REVIEW):
            self.notice.emit(message)

    @Slot(object)
    def _on_plan_output_root_ready(self, output_root: object) -> None:
        self._output_root = output_root  # type: ignore[assignment]

    @Slot(str)
    def _on_plan_batch_warning(self, message: str) -> None:
        if self._shutting_down:
            return
        if self._state == ControllerState.PLANNING:
            self.banner.emit(message)

    @Slot(object)
    def _on_plan_event(self, event: object) -> None:
        if self._shutting_down:
            return
        if self._state == ControllerState.PLANNING:
            self.event_received.emit(event)

    @Slot(object)
    def _on_plan_stats(self, stats: object) -> None:
        if self._shutting_down:
            return
        if self._state == ControllerState.PLANNING:
            self.stats_changed.emit(stats)

    @Slot(object)
    def _on_plan_session_opened(self, session: object) -> None:
        self._session = session
        warning = getattr(session, "cache_warning", None)
        if warning and not self._shutting_down and self._state == ControllerState.PLANNING:
            self.banner.emit(str(warning))

    @Slot(object)
    def _on_plan_ready(self, items: object) -> None:
        self._plan_result_items = list(items)  # type: ignore[arg-type]

    @Slot(str, str)
    def _on_plan_input_error(self, kind: str, message: str) -> None:
        self._plan_input_error = (kind, message)

    @Slot()
    def _on_plan_cancelled(self) -> None:
        self._plan_was_cancelled = True

    @Slot(str, str)
    def _on_plan_failed(self, title: str, details: str) -> None:
        self._plan_failed = (title, details)

    @Slot()
    def _on_plan_thread_finished(self) -> None:
        thread = self._plan_thread
        worker = self._plan_worker
        self._plan_thread = None
        self._plan_worker = None

        if thread is not None:
            thread.wait()
            thread.deleteLater()

        # Ensure ownership of any opened session even if queued signal did not run yet.
        if worker is not None:
            if self._session is None and worker._opened_session is not None:
                self._session = worker._opened_session
            worker._opened_session = None
            worker.deleteLater()

        self._cleanup_prompt_bridge()

        plan_items = self._plan_result_items
        input_error_info = self._plan_input_error
        failed_info = self._plan_failed
        was_cancelled = self._plan_was_cancelled
        self._plan_result_items = None
        self._plan_input_error = None
        self._plan_failed = None
        self._plan_was_cancelled = False

        if self._shutting_down:
            self._maybe_finish_shutdown()
            return

        if (
            self._state == ControllerState.CANCELLING
            or was_cancelled
            or (self._token is not None and self._token.is_cancelled)
        ):
            self._plan_items = []
            self._original_actions = []
            self._output_root = None
            self._close_session()
            self._token = None
            self._cancel_source = None
            self._set_state(ControllerState.INPUTS)
            return

        if input_error_info is not None:
            kind, message = input_error_info
            self._plan_items = []
            self._original_actions = []
            self._output_root = None
            self._close_session()
            self._token = None
            self.input_error.emit(kind, message)
            self._set_state(ControllerState.INPUTS)
            return

        if failed_info is not None:
            title, details = failed_info
            self._plan_items = []
            self._original_actions = []
            self._output_root = None
            self._close_session()
            self._token = None
            self.error.emit(title, details)
            self._set_state(ControllerState.INPUTS)
            return

        if plan_items is not None:
            self._plan_items = list(plan_items)
            self._original_actions = [item.action for item in self._plan_items]
            self.plan_ready.emit(self._plan_items)
            self._set_state(ControllerState.REVIEW)
            return

        self._plan_items = []
        self._original_actions = []
        self._output_root = None
        self._close_session()
        self._token = None
        self._set_state(ControllerState.INPUTS)

    @Slot(object)
    def _on_execute_event(self, event: object) -> None:
        if (
            isinstance(event, TrackEvent)
            and event.index >= 0
            and event.stage == TrackStage.CANCELLED
        ):
            if event.message == NOT_RUN_MESSAGE:
                self._not_run_count += 1
            else:
                self._cancelled_count += 1
        if not self._shutting_down:
            self.event_received.emit(event)

    @Slot(object, int, int)
    def _on_execute_result(self, result: object, completed: int, total: int) -> None:
        _ = completed, total
        self._partial_results.append(result)  # type: ignore[arg-type]

    @Slot(object)
    def _on_execute_summary_ready(self, summary: object) -> None:
        self._execute_summary = summary  # type: ignore[assignment]

    @Slot()
    def _on_execute_cancelled(self) -> None:
        self._execute_was_cancelled = True

    @Slot(str, str)
    def _on_execute_failed(self, title: str, details: str) -> None:
        self._execute_failed = (title, details)

    @Slot()
    def _on_execute_thread_finished(self) -> None:
        thread = self._execute_thread
        worker = self._execute_worker
        self._execute_thread = None
        self._execute_worker = None

        if thread is not None:
            thread.wait()
            thread.deleteLater()
        if worker is not None:
            worker.deleteLater()

        summary = self._execute_summary
        failed_info = self._execute_failed
        was_cancelled = (
            self._execute_was_cancelled
            or self._state == ControllerState.CANCELLING
            or (self._token is not None and self._token.is_cancelled)
        )
        self._execute_summary = None
        self._execute_failed = None
        self._execute_was_cancelled = False

        if self._shutting_down:
            self._maybe_finish_shutdown()
            return

        # Close the session only after the worker thread has finished (section 4.5).
        self._close_session()
        self._token = None
        self._cancel_source = None

        output_root = self._output_root if self._output_root is not None else Path(".")
        if summary is None:
            summary = RunSummary(
                output_root=output_root,
                results=list(self._partial_results),
                interrupted=was_cancelled,
            )

        cancelled_count = self._cancelled_count
        not_run_count = self._not_run_count
        unaccounted = max(
            0,
            len(self._plan_items) - len(summary.results) - cancelled_count - not_run_count,
        )
        if unaccounted > 0:
            if summary.interrupted:
                cancelled_count += unaccounted
            elif failed_info is not None:
                not_run_count += unaccounted

        self._summary = summary

        if failed_info is not None:
            title, details = failed_info
            self.error.emit(title, details)

        self.summary_ready.emit(summary, cancelled_count, not_run_count)
        self._set_state(ControllerState.SUMMARY)

    @Slot(object)
    def _on_cache_stats_ready(self, stats: object) -> None:
        self._cache_stats_result = stats

    @Slot(int)
    def _on_cache_cleared(self, removed: int) -> None:
        self._cache_cleared_result = removed

    @Slot(str, str)
    def _on_cache_failed(self, title: str, details: str) -> None:
        self._cache_error = (title, details)

    @Slot()
    def _on_cache_thread_finished(self) -> None:
        thread = self._cache_thread
        worker = self._cache_worker
        stats_result = self._cache_stats_result
        cleared_result = self._cache_cleared_result
        cache_error = self._cache_error

        self._cache_thread = None
        self._cache_worker = None
        self._cache_token = None
        self._cache_stats_result = None
        self._cache_cleared_result = None
        self._cache_error = None

        if thread is not None:
            thread.wait()
            thread.deleteLater()
        if worker is not None:
            worker.deleteLater()

        if self._shutting_down:
            self._pending_cache_action = None
            self._maybe_finish_shutdown()
            return

        auto_refresh_stats = False
        if cache_error is not None:
            title, details = cache_error
            self.error.emit(title, details)
        elif stats_result is not None:
            # Suppress stale stats while a clear is already queued; the clear's
            # post-clear refresh will emit the authoritative cache_stats_ready.
            if self._pending_cache_action not in ("clear_expired", "clear_all"):
                self.cache_stats_ready.emit(stats_result)
        elif cleared_result is not None:
            auto_refresh_stats = True
            self.cache_cleared.emit(cleared_result)

        if self._shutting_down:
            self._pending_cache_action = None
            self._maybe_finish_shutdown()
            return

        if self._cache_thread is not None:
            # A signal listener already started the next cache worker synchronously.
            return

        if self._pending_cache_action is not None:
            self._drain_pending_cache_action()
        elif auto_refresh_stats and not self._is_cache_batch_blocked():
            if self._preflight_thread is not None:
                self._queue_cache_action("stats")
            else:
                self._launch_cache_worker("stats")
