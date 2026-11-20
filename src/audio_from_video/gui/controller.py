"""UI-thread controller state machine and binary status model for the desktop GUI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

from audio_from_video.session import ServicesFactory, create_online_services

if TYPE_CHECKING:
    from audio_from_video.binaries import BinaryResolution
    from audio_from_video.config import AppConfig
    from audio_from_video.models import PlanItem
    from audio_from_video.session import RunOptions

SETTINGS_ORGANIZATION = "audio-from-video"
SETTINGS_APPLICATION = "afv-gui"


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
        self._plan_items: list[PlanItem] = []
        self._output_root: Path | None = None
        self._offline_default = False

    @property
    def state(self) -> ControllerState:
        """Return the current workflow state."""
        return self._state

    @property
    def config(self) -> AppConfig | None:
        """Return the loaded application configuration, if available."""
        return self._config

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

    def start_preflight(self) -> None:
        """Load configuration and inspect FFmpeg binaries on a worker thread."""
        raise NotImplementedError("RunController is implemented in Stage 2 lane G.")

    def save_contact(self, contact: str) -> None:
        """Validate and save the MusicBrainz contact; report via ``setup_error`` or state change."""
        _ = contact
        raise NotImplementedError("RunController is implemented in Stage 2 lane G.")

    def use_offline_only(self) -> None:
        """Enable offline-only mode and persist ``startup/offline = true`` in ``QSettings``."""
        raise NotImplementedError("RunController is implemented in Stage 2 lane G.")

    def set_ffmpeg_path(self, path: Path) -> None:
        """Persist a custom FFmpeg path with ``save_settings`` and re-check binaries."""
        _ = path
        raise NotImplementedError("RunController is implemented in Stage 2 lane G.")

    def start_plan(self, options: RunOptions) -> None:
        """Begin Stage 1 discovery and planning on a worker thread.

        When planning cannot start because the MusicBrainz contact or FFmpeg is
        missing (``kind`` is ``"missing_contact"`` or ``"missing_ffmpeg"``),
        this emits both ``input_error(kind, message)`` and
        ``setup_error(message)``, transitions to ``ControllerState.SETUP``, and
        does not start a worker so the Setup screen can display the reason.
        """
        _ = options
        raise NotImplementedError("RunController is implemented in Stage 2 lane G.")

    def cancel(self) -> None:
        """Cancel the active planning or execution worker and enter ``CANCELLING``."""
        raise NotImplementedError("RunController is implemented in Stage 2 lane G.")

    def back(self) -> None:
        """Close any active session and return from Review to Inputs."""
        raise NotImplementedError("RunController is implemented in Stage 2 lane G.")

    def set_row_skipped(self, index: int, skipped: bool) -> None:
        """Toggle a row between ``SKIP_USER`` and its original action; emit ``row_changed``."""
        _ = index, skipped
        raise NotImplementedError("RunController is implemented in Stage 2 lane G.")

    def start_run(self) -> None:
        """Begin Stage 2 extraction for the current plan."""
        raise NotImplementedError("RunController is implemented in Stage 2 lane G.")

    def answer_prompt(self, request_id: int, value: object) -> None:
        """Deliver a dialog response back to the active ``PromptBridge``."""
        _ = request_id, value
        raise NotImplementedError("RunController is implemented in Stage 2 lane G.")

    def new_batch(self) -> None:
        """Close the current session and return from Summary to Inputs."""
        raise NotImplementedError("RunController is implemented in Stage 2 lane G.")

    def save_settings(self, updates: Mapping[str, object]) -> None:
        """Persist updated configuration keys and emit ``settings_saved``."""
        _ = updates
        raise NotImplementedError("RunController is implemented in Stage 2 lane G.")

    def request_cache_stats(self) -> None:
        """Inspect SQLite cache statistics on a worker thread."""
        raise NotImplementedError("RunController is implemented in Stage 2 lane G.")

    def clear_cache(self, *, expired_only: bool) -> None:
        """Clear cache entries on a worker thread and emit ``cache_cleared``."""
        _ = expired_only
        raise NotImplementedError("RunController is implemented in Stage 2 lane G.")

    def save_report(self, path: Path) -> None:
        """Write the latest run summary as a JSON report to ``path``."""
        _ = path
        raise NotImplementedError("RunController is implemented in Stage 2 lane G.")

    def shutdown(self) -> None:
        """Stop any running worker in shutdown order (section 4.5) before exiting."""
        raise NotImplementedError("RunController is implemented in Stage 2 lane G.")
