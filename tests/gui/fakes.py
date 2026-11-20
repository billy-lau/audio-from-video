"""Test double mirroring ``RunController`` commands, properties, and Qt signals."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

from audio_from_video.gui.controller import ControllerState

if TYPE_CHECKING:
    from audio_from_video.config import AppConfig
    from audio_from_video.models import PlanItem
    from audio_from_video.session import RunOptions


class FakeController(QObject):
    """Recording ``QObject`` mirroring ``RunController`` for Lanes H and I widget tests.

    Exposes the exact frozen signals, read-only properties, and command
    signatures of ``RunController`` (section 5.5). Each command appends a
    ``(method_name, args, kwargs)`` entry to :attr:`calls` without starting any
    background thread, touching disk, or modifying configuration.
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
        state: ControllerState = ControllerState.PREFLIGHT,
        config: AppConfig | None = None,
        plan_items: list[PlanItem] | None = None,
        output_root: Path | None = None,
        offline_default: bool = False,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._config = config
        self._plan_items: list[PlanItem] = list(plan_items) if plan_items is not None else []
        self._output_root = output_root
        self._offline_default = offline_default
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _record(self, name: str, *args: object, **kwargs: object) -> None:
        self.calls.append((name, args, dict(kwargs)))

    def calls_for(self, name: str) -> list[tuple[tuple[object, ...], dict[str, object]]]:
        """Return ``(args, kwargs)`` tuples recorded for ``name``, in call order."""
        return [(args, kwargs) for call_name, args, kwargs in self.calls if call_name == name]

    def clear_calls(self) -> None:
        """Remove all recorded command calls."""
        self.calls.clear()

    # -- Read-only properties matching RunController ---------------------------

    @property
    def state(self) -> ControllerState:
        """Return the current workflow state."""
        return self._state

    @property
    def config(self) -> AppConfig | None:
        """Return the configured ``AppConfig``, if any."""
        return self._config

    @property
    def plan_items(self) -> list[PlanItem]:
        """Return the planned items for the current batch."""
        return self._plan_items

    @property
    def output_root(self) -> Path | None:
        """Return the resolved output root directory, if any."""
        return self._output_root

    @property
    def offline_default(self) -> bool:
        """Return whether startup defaults to offline mode."""
        return self._offline_default

    # -- Test helper mutators --------------------------------------------------

    def set_state(self, state: ControllerState, *, emit: bool = True) -> None:
        """Update the fake state and optionally emit ``state_changed``."""
        self._state = state
        if emit:
            self.state_changed.emit(state)

    def set_config(self, config: AppConfig | None) -> None:
        """Replace the stored ``AppConfig`` snapshot for widget tests."""
        self._config = config

    def set_plan_items(self, items: list[PlanItem], *, emit: bool = False) -> None:
        """Replace stored plan items and optionally emit ``plan_ready``."""
        self._plan_items = list(items)
        if emit:
            self.plan_ready.emit(self._plan_items)

    def set_output_root(self, output_root: Path | None) -> None:
        """Replace the stored output root directory."""
        self._output_root = output_root

    def set_offline_default(self, offline: bool) -> None:
        """Update the stored ``offline_default`` flag."""
        self._offline_default = offline

    def emit_setup_redirect(self, kind: str, message: str) -> None:
        """Simulate a ``missing_contact`` or ``missing_ffmpeg`` rejection from ``start_plan``.

        Mirrors ``RunController.start_plan``: emits ``input_error(kind, message)``,
        emits ``setup_error(message)``, and transitions to ``ControllerState.SETUP``.
        """
        self.input_error.emit(kind, message)
        self.setup_error.emit(message)
        self.set_state(ControllerState.SETUP, emit=True)

    # -- Recorded commands matching RunController ------------------------------

    def start_preflight(self) -> None:
        """Record a ``start_preflight`` call."""
        self._record("start_preflight")

    def save_contact(self, contact: str) -> None:
        """Record a ``save_contact`` call."""
        self._record("save_contact", contact)

    def use_offline_only(self) -> None:
        """Record a ``use_offline_only`` call."""
        self._record("use_offline_only")

    def set_ffmpeg_path(self, path: Path) -> None:
        """Record a ``set_ffmpeg_path`` call."""
        self._record("set_ffmpeg_path", path)

    def start_plan(self, options: RunOptions) -> None:
        """Record a ``start_plan`` call."""
        self._record("start_plan", options)

    def cancel(self) -> None:
        """Record a ``cancel`` call."""
        self._record("cancel")

    def back(self) -> None:
        """Record a ``back`` call."""
        self._record("back")

    def set_row_skipped(self, index: int, skipped: bool) -> None:
        """Record a ``set_row_skipped`` call."""
        self._record("set_row_skipped", index, skipped)

    def start_run(self) -> None:
        """Record a ``start_run`` call."""
        self._record("start_run")

    def answer_prompt(self, request_id: int, value: object) -> None:
        """Record an ``answer_prompt`` call."""
        self._record("answer_prompt", request_id, value)

    def new_batch(self) -> None:
        """Record a ``new_batch`` call."""
        self._record("new_batch")

    def save_settings(self, updates: Mapping[str, object]) -> None:
        """Record a ``save_settings`` call."""
        self._record("save_settings", updates)

    def request_cache_stats(self) -> None:
        """Record a ``request_cache_stats`` call."""
        self._record("request_cache_stats")

    def clear_cache(self, *, expired_only: bool) -> None:
        """Record a ``clear_cache`` call."""
        self._record("clear_cache", expired_only=expired_only)

    def save_report(self, path: Path) -> None:
        """Record a ``save_report`` call."""
        self._record("save_report", path)

    def shutdown(self) -> None:
        """Record a ``shutdown`` call."""
        self._record("shutdown")
