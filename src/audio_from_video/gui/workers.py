"""Worker QObjects moved to QThreads for non-blocking GUI background operations."""

from __future__ import annotations

import logging
import sqlite3
import subprocess
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, Signal, Slot

from audio_from_video import binaries as _binaries_mod
from audio_from_video import cache as _cache_mod
from audio_from_video import config as _config_mod
from audio_from_video import discovery as _discovery_mod
from audio_from_video import planner as _planner_mod
from audio_from_video import session as _session_mod
from audio_from_video.cancel import CancelToken, OperationCancelled
from audio_from_video.events import TrackEvent, TrackStage
from audio_from_video.session import ServicesFactory, create_online_services

if TYPE_CHECKING:
    from audio_from_video.binaries import ResolvedBinaries
    from audio_from_video.config import AppConfig
    from audio_from_video.gui.bridge import PromptBridge
    from audio_from_video.models import PlanItem, TrackResult
    from audio_from_video.session import RunOptions, RunSession

logger = logging.getLogger(__name__)


def _format_failure_details(exc: BaseException) -> str:
    """Return a concise exception summary followed by a formatted traceback."""
    message = str(exc).strip()
    header = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
    if tb_text and tb_text != header:
        return f"{header}\n\n{tb_text}"
    return header


class _CompletionRelay(QObject):
    """Child ``QObject`` signalling worker completion without adding public signals."""

    finished = Signal()


class _PlanRelay(QObject):
    """Child ``QObject`` emitting internal ``PlanWorker`` metadata and completion."""

    output_root_ready = Signal(object)  # Path
    batch_warning = Signal(str)
    finished = Signal()


class PreflightWorker(QObject):
    """Loads ``AppConfig`` and checks FFmpeg, FFprobe, and ``libmp3lame`` off the UI thread."""

    config_loaded = Signal(object)  # AppConfig
    binaries_checked = Signal(object)  # BinariesStatus
    failed = Signal(str, str)  # title, details

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        config: AppConfig | None = None,
        config_path: Path | None = None,
        binaries_only: bool = False,
        token: CancelToken | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._config_path = config_path
        self._binaries_only = binaries_only
        self._token = token
        self._relay = _CompletionRelay(self)

    def _raise_if_cancelled(self) -> None:
        if self._token is not None:
            self._token.raise_if_cancelled()

    @Slot()
    def run(self) -> None:
        """Run preflight config loading and binary inspection."""
        from audio_from_video.gui.controller import BinariesStatus

        try:
            self._raise_if_cancelled()
            if self._binaries_only:
                cfg = self._config
            else:
                cfg = _config_mod.load_config(config_path=self._config_path)
                self._raise_if_cancelled()
                self.config_loaded.emit(cfg)

            self._raise_if_cancelled()
            ffmpeg_res, ffprobe_res = _binaries_mod.inspect_binaries(config=cfg)
            self._raise_if_cancelled()

            has_lame = False
            if ffmpeg_res.found and ffmpeg_res.path is not None:
                try:
                    has_lame = bool(_binaries_mod.has_libmp3lame(ffmpeg_res.path))
                except subprocess.SubprocessError:
                    logger.warning(
                        "Timed out or failed while checking libmp3lame support for %s.",
                        ffmpeg_res.path,
                        exc_info=True,
                    )
                    has_lame = False

            self._raise_if_cancelled()
            status = BinariesStatus(
                ffmpeg=ffmpeg_res,
                ffprobe=ffprobe_res,
                has_libmp3lame=has_lame,
            )
            self.binaries_checked.emit(status)
        except OperationCancelled:
            return
        except _config_mod.ConfigError as exc:
            if self._token is not None and self._token.is_cancelled:
                return
            self.failed.emit("Configuration error", str(exc))
        except Exception as exc:
            if self._token is not None and self._token.is_cancelled:
                return
            self.failed.emit("Preflight failed", _format_failure_details(exc))
        finally:
            self._relay.finished.emit()


class PlanWorker(QObject):
    """Runs discovery, override validation, ``RunSession.open``, and Stage 1 planning."""

    event = Signal(object)  # TrackEvent
    stats = Signal(object)  # LookupStats
    session_opened = Signal(object)  # RunSession
    plan_ready = Signal(object)  # list[PlanItem]
    input_error = Signal(str, str)  # kind, message
    cancelled = Signal()
    failed = Signal(str, str)  # title, details

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        options: RunOptions | None = None,
        config: AppConfig | None = None,
        binaries: ResolvedBinaries | None = None,
        token: CancelToken | None = None,
        user_agent: str | None = None,
        cache_dir: Path | None = None,
        services_factory: ServicesFactory = create_online_services,
        prompt_bridge: PromptBridge | None = None,
        defer_session_close: bool = False,
    ) -> None:
        super().__init__(parent)
        self._options = options
        self._config = config
        self._binaries = binaries
        self._token = token
        self._user_agent = user_agent
        self._cache_dir = cache_dir
        self._services_factory = services_factory
        self._prompt_bridge = prompt_bridge
        self._defer_session_close = defer_session_close
        self._opened_session: Any | None = None
        self._relay = _PlanRelay(self)

    def _emit_stats(self, session: Any) -> None:
        """Read ``LookupStats`` on the worker thread and forward via signal."""
        lookup_stats = getattr(session, "lookup_stats", None)
        if callable(lookup_stats):
            self.stats.emit(lookup_stats())

    @Slot()
    def run(self) -> None:
        """Run Stage 1 discovery and planning."""
        succeeded = False
        token = self._token if self._token is not None else CancelToken()
        try:
            if self._options is None or self._config is None or self._binaries is None:
                self.failed.emit(
                    "Planning failed",
                    "PlanWorker requires options, config, and resolved binaries.",
                )
                return

            options = self._options
            config = self._config
            binaries = self._binaries

            token.raise_if_cancelled()

            # Validate per-run numeric and encoding overrides before scanning inputs.
            if options.quality is not None:
                _config_mod.parse_quality(options.quality)
            raw_jobs = config.execution.jobs if options.jobs is None else options.jobs
            _config_mod.validate_jobs(raw_jobs)
            if options.overrides.track is not None and options.overrides.track < 1:
                raise _planner_mod.PlanValidationError(
                    f"--track must be a positive integer (got {options.overrides.track})."
                )
            if options.stream is not None and options.stream < 0:
                raise _planner_mod.PlanValidationError(
                    f"--stream must be a non-negative integer (got {options.stream})."
                )

            effective_output_dir = (
                options.output_dir
                if options.output_dir is not None
                else (Path(config.output.output_dir) if config.output.output_dir else None)
            )
            media_inputs, output_root = _discovery_mod.discover_inputs(
                options.inputs,
                recursive=options.recursive,
                output_dir=effective_output_dir,
            )

            # Discovery cannot inspect CancelToken directly; discard its result if cancelled.
            token.raise_if_cancelled()

            if not media_inputs:
                self.input_error.emit("no_inputs", "No supported video files found.")
                return

            _planner_mod.validate_manual_overrides(media_inputs, options.overrides)
            token.raise_if_cancelled()

            # Announce the total discovered track count before probing starts.
            self.event.emit(
                TrackEvent(
                    index=-1,
                    source=None,
                    stage=TrackStage.PROBING,
                    message=f"total_tracks={len(media_inputs)}",
                )
            )

            cache_dir = (
                self._cache_dir if self._cache_dir is not None else _config_mod.default_cache_dir()
            )
            session = _session_mod.RunSession.open(
                options,
                config,
                binaries,
                token=token,
                user_agent=None if options.offline else self._user_agent,
                cache_dir=cache_dir,
                services_factory=self._services_factory,
            )
            self._opened_session = session
            self._relay.output_root_ready.emit(output_root)
            self.session_opened.emit(session)

            token.raise_if_cancelled()

            prompt_callbacks = (
                self._prompt_bridge.callbacks(offline=options.offline)
                if self._prompt_bridge is not None
                else None
            )

            def _on_event(ev: TrackEvent) -> None:
                self.event.emit(ev)
                if not options.offline and ev.stage == TrackStage.LOOKUP:
                    self._emit_stats(session)

            def _on_lookup_progress(done: int, total_items: int, active_label: str) -> None:
                _ = done, total_items, active_label
                self._emit_stats(session)

            def _on_lookup_complete() -> None:
                self._emit_stats(session)

            def _on_batch_warning(message: str) -> None:
                self._relay.batch_warning.emit(message)

            if not options.offline:
                self._emit_stats(session)

            plan_items = session.build_plan(
                media_inputs,
                output_root=output_root,
                prompt_callbacks=prompt_callbacks,
                non_interactive=prompt_callbacks is None,
                on_event=_on_event,
                on_lookup_progress=_on_lookup_progress if not options.offline else None,
                on_lookup_complete=_on_lookup_complete if not options.offline else None,
                on_batch_warning=_on_batch_warning,
            )

            token.raise_if_cancelled()

            if not options.offline:
                self._emit_stats(session)

            succeeded = True
            self.plan_ready.emit(plan_items)
        except OperationCancelled:
            self.cancelled.emit()
        except _discovery_mod.MultipleInputRootsError as exc:
            if token.is_cancelled:
                self.cancelled.emit()
            else:
                self.input_error.emit("multiple_roots", str(exc))
        except _discovery_mod.DiscoveryError as exc:
            if token.is_cancelled:
                self.cancelled.emit()
            else:
                self.input_error.emit("discovery", str(exc))
        except (_planner_mod.PlanValidationError, _config_mod.ConfigError) as exc:
            if token.is_cancelled:
                self.cancelled.emit()
            else:
                self.input_error.emit("validation", str(exc))
        except Exception as exc:
            if token.is_cancelled:
                self.cancelled.emit()
            else:
                self.failed.emit("Planning failed", _format_failure_details(exc))
        finally:
            if not succeeded and not self._defer_session_close and self._opened_session is not None:
                session_to_close = self._opened_session
                self._opened_session = None
                try:
                    session_to_close.close()
                except Exception:
                    logger.warning(
                        "Error closing RunSession after PlanWorker failure.",
                        exc_info=True,
                    )
            self._relay.finished.emit()


class ExecuteWorker(QObject):
    """Runs Stage 2 extraction via ``RunSession.execute`` off the UI thread."""

    event = Signal(object)  # TrackEvent
    result = Signal(object, int, int)  # TrackResult, completed, total
    summary_ready = Signal(object)  # RunSummary
    cancelled = Signal()
    failed = Signal(str, str)  # title, details

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        session: RunSession | None = None,
        plan_items: Sequence[PlanItem] = (),
        output_root: Path | None = None,
        token: CancelToken | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._plan_items = list(plan_items)
        self._output_root = output_root
        self._token = token
        self._relay = _CompletionRelay(self)

    @Slot()
    def run(self) -> None:
        """Run Stage 2 execution."""
        try:
            if self._session is None or self._output_root is None:
                self.failed.emit(
                    "Extraction failed",
                    "ExecuteWorker requires an open session and output_root.",
                )
                return

            if self._token is not None:
                self._token.raise_if_cancelled()

            def _on_event(ev: TrackEvent) -> None:
                self.event.emit(ev)

            def _on_result(track_result: TrackResult, completed: int, total: int) -> None:
                self.result.emit(track_result, completed, total)

            summary = self._session.execute(
                self._plan_items,
                output_root=self._output_root,
                on_result=_on_result,
                on_event=_on_event,
                install_signal_handlers=False,
            )
            self.summary_ready.emit(summary)
            if getattr(summary, "interrupted", False):
                self.cancelled.emit()
        except OperationCancelled:
            self.cancelled.emit()
        except Exception as exc:
            if self._token is not None and self._token.is_cancelled:
                self.cancelled.emit()
            else:
                self.failed.emit("Extraction failed", _format_failure_details(exc))
        finally:
            self._relay.finished.emit()


def _delete_corrupt_backups(cache_file: Path) -> None:
    """Remove any moved-aside corrupt SQLite cache copies, logging failures."""
    for backup in _cache_mod.find_corrupt_backups(cache_file):
        try:
            backup.unlink()
        except OSError:
            logger.warning("Could not delete moved-aside corrupt cache file %s.", backup)


class CacheWorker(QObject):
    """Reads cache statistics or clears cached entries when no run is active."""

    stats_ready = Signal(object)  # list[NamespaceStats]
    cleared = Signal(int)  # entries removed
    failed = Signal(str, str)  # title, details

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        operation: str = "stats",
        cache_dir: Path | None = None,
        expired_only: bool = False,
        token: CancelToken | None = None,
    ) -> None:
        super().__init__(parent)
        self._operation = operation
        self._cache_dir = cache_dir
        self._expired_only = expired_only
        self._token = token
        self._relay = _CompletionRelay(self)

    def _raise_if_cancelled(self) -> None:
        if self._token is not None:
            self._token.raise_if_cancelled()

    @Slot()
    def run(self) -> None:
        """Execute the requested cache operation."""
        try:
            self._raise_if_cancelled()
            active_cache_dir = (
                self._cache_dir if self._cache_dir is not None else _config_mod.default_cache_dir()
            )
            cache_file = active_cache_dir / _cache_mod.CACHE_FILENAME

            if self._operation == "stats":
                stats = _cache_mod.TTLCache.read_stats(cache_file)
                self._raise_if_cancelled()
                self.stats_ready.emit(stats)
                return

            if self._operation == "clear":
                if not cache_file.is_file():
                    _delete_corrupt_backups(cache_file)
                    self._raise_if_cancelled()
                    self.cleared.emit(0)
                    return

                before_rows: int | None = None
                try:
                    before_stats = _cache_mod.TTLCache.read_stats(cache_file)
                    before_rows = sum(entry.rows for entry in before_stats)
                except (OSError, sqlite3.Error):
                    before_rows = None

                self._raise_if_cancelled()
                ttl_cache = _cache_mod.TTLCache(cache_file)
                try:
                    if ttl_cache.disabled:
                        message = ttl_cache.warning or "Could not open the cache for clearing."
                        self.failed.emit("Could not clear cache", message)
                        return
                    clear_removed = int(ttl_cache.clear(expired_only=self._expired_only))
                finally:
                    ttl_cache.close()

                _delete_corrupt_backups(cache_file)

                removed = clear_removed
                if before_rows is not None:
                    try:
                        after_stats = _cache_mod.TTLCache.read_stats(cache_file)
                        after_rows = sum(entry.rows for entry in after_stats)
                        removed = max(clear_removed, max(0, before_rows - after_rows))
                    except (OSError, sqlite3.Error):
                        removed = clear_removed

                self._raise_if_cancelled()
                self.cleared.emit(removed)
                return

            self.failed.emit(
                "Cache operation failed",
                f"Unknown cache operation {self._operation!r}.",
            )
        except OperationCancelled:
            return
        except (OSError, sqlite3.Error) as exc:
            if self._token is not None and self._token.is_cancelled:
                return
            title = (
                "Could not read cache stats"
                if self._operation == "stats"
                else "Could not clear cache"
            )
            self.failed.emit(title, str(exc))
        except Exception as exc:
            if self._token is not None and self._token.is_cancelled:
                return
            self.failed.emit("Cache operation failed", _format_failure_details(exc))
        finally:
            self._relay.finished.emit()
