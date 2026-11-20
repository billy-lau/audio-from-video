"""Worker QObjects moved to QThreads for non-blocking GUI background operations."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot


class PreflightWorker(QObject):
    """Loads ``AppConfig`` and checks FFmpeg, FFprobe, and ``libmp3lame`` off the UI thread."""

    config_loaded = Signal(object)  # AppConfig
    binaries_checked = Signal(object)  # BinariesStatus
    failed = Signal(str, str)  # title, details

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

    @Slot()
    def run(self) -> None:
        """Run preflight config loading and binary inspection."""
        raise NotImplementedError("PreflightWorker is implemented in Stage 2 lane G.")


class PlanWorker(QObject):
    """Runs discovery, override validation, ``RunSession.open``, and Stage 1 planning."""

    event = Signal(object)  # TrackEvent
    stats = Signal(object)  # LookupStats
    session_opened = Signal(object)  # RunSession
    plan_ready = Signal(object)  # list[PlanItem]
    input_error = Signal(str, str)  # kind, message
    cancelled = Signal()
    failed = Signal(str, str)  # title, details

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

    @Slot()
    def run(self) -> None:
        """Run Stage 1 discovery and planning."""
        raise NotImplementedError("PlanWorker is implemented in Stage 2 lane G.")


class ExecuteWorker(QObject):
    """Runs Stage 2 extraction via ``RunSession.execute`` off the UI thread."""

    event = Signal(object)  # TrackEvent
    result = Signal(object, int, int)  # TrackResult, completed, total
    summary_ready = Signal(object)  # RunSummary
    cancelled = Signal()
    failed = Signal(str, str)  # title, details

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

    @Slot()
    def run(self) -> None:
        """Run Stage 2 execution."""
        raise NotImplementedError("ExecuteWorker is implemented in Stage 2 lane G.")


class CacheWorker(QObject):
    """Reads cache statistics or clears cached entries when no run is active."""

    stats_ready = Signal(object)  # list[NamespaceStats]
    cleared = Signal(int)  # entries removed
    failed = Signal(str, str)  # title, details

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

    @Slot()
    def run(self) -> None:
        """Execute the requested cache operation."""
        raise NotImplementedError("CacheWorker is implemented in Stage 2 lane G.")
