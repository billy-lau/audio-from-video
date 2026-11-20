"""Shared ``QAbstractTableModel`` displaying planned and executing tracks."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    QSortFilterProxyModel,
    Qt,
    Slot,
)
from PySide6.QtGui import QIcon

from audio_from_video.events import NOT_RUN_MESSAGE, TrackEvent, TrackStage
from audio_from_video.gui.resources import load_icon
from audio_from_video.models import (
    MatchDecision,
    MatchStatus,
    PlanAction,
    PlanItem,
    RunOutcome,
)

PLAN_ITEM_ROLE: int = int(Qt.ItemDataRole.UserRole)
TRACK_STAGE_ROLE: int = int(Qt.ItemDataRole.UserRole) + 1

COL_STATUS = 0
COL_SOURCE = 1
COL_DESTINATION = 2
COL_ACTION = 3
COL_MATCH = 4
COL_WARNINGS = 5


class PlanFilterMode(StrEnum):
    """Row filter modes used by ``ReviewScreen`` via ``PlanFilterProxyModel``."""

    ALL = "all"
    NEEDS_ATTENTION = "needs_attention"
    SKIPPED = "skipped"
    FAILED = "failed"


def is_skipped_action(action: PlanAction) -> bool:
    """Return True when ``action`` skips extraction for the row."""
    return action in (
        PlanAction.SKIP_EXISTING,
        PlanAction.SKIP_NO_AUDIO,
        PlanAction.SKIP_USER,
    )


def is_failed_action(action: PlanAction) -> bool:
    """Return True when ``action`` marks a planning-time failure."""
    return action in (PlanAction.FAIL_NO_AUDIO, PlanAction.FAIL_ERROR)


_OFFLINE_SOURCE_DESCRIPTIONS: frozenset[str] = frozenset(
    {"offline", "offline_folder", "offline_unmatched"}
)


def is_offline_item(item: PlanItem) -> bool:
    """Return True when ``item`` uses offline-only metadata without an online mismatch."""
    metadata = item.metadata
    if metadata is None:
        return False
    if item.match_decision in (MatchDecision.AMBIGUOUS, MatchDecision.UNMATCHED):
        return False
    return metadata.source_description in _OFFLINE_SOURCE_DESCRIPTIONS


def is_needs_attention(item: PlanItem, *, outcome: RunOutcome | None = None) -> bool:
    """Return True when ``item`` has warnings, errors, conflicts, or uncertain matching.

    Offline rows without failures, warnings, or conflicts are exempted so the
    ``Needs attention`` filter stays meaningful for offline batches.
    """
    if outcome == RunOutcome.FAILED or is_failed_action(item.action):
        return True
    if item.error or item.warnings or item.conflict_note:
        return True
    if item.match_decision in (MatchDecision.AMBIGUOUS, MatchDecision.UNMATCHED):
        return True
    if item.metadata is None or is_offline_item(item):
        return False
    return item.metadata.match_status in (MatchStatus.PARTIAL, MatchStatus.UNMATCHED)


def format_destination(output_path: Path | None, output_root: Path | None) -> str:
    """Return ``output_path`` formatted relative to ``output_root`` when possible."""
    if output_path is None:
        return "—"
    if output_root is not None:
        try:
            return output_path.relative_to(output_root).as_posix()
        except ValueError:
            try:
                return output_path.resolve().relative_to(output_root.resolve()).as_posix()
            except (OSError, ValueError):
                pass
    if output_path.is_absolute():
        return str(output_path)
    return output_path.as_posix()


def format_action_text(item: PlanItem) -> str:
    """Return a concise, human-readable label for a ``PlanItem`` disposition."""
    if item.action == PlanAction.EXTRACT:
        if item.extraction is not None and item.extraction.stream_copy:
            mode = "copy"
        elif item.extraction is not None:
            if item.extraction.quality.is_vbr:
                mode = item.extraction.quality.label
            else:
                kbps = (
                    item.extraction.effective_bitrate_kbps or item.extraction.quality.bitrate_kbps
                )
                mode = f"{kbps}k" if kbps is not None else item.extraction.quality.label
            if item.extraction.normalize:
                mode = f"{mode}+norm"
        else:
            mode = "encode"
        suffix = f" ({item.conflict_note})" if item.conflict_note else ""
        return f"Extract [{mode}]{suffix}"
    if item.action == PlanAction.SKIP_EXISTING:
        return "Skip (exists)"
    if item.action == PlanAction.SKIP_NO_AUDIO:
        return "Skip (no audio)"
    if item.action == PlanAction.SKIP_USER:
        return "Skip (user)"
    if item.action == PlanAction.FAIL_NO_AUDIO:
        return "Fail (no audio)"
    return "Fail (error)"


def format_match_text(item: PlanItem) -> str:
    """Return a concise match label preserving artist, title, and source provenance."""
    metadata = item.metadata
    if metadata is None:
        return "—"
    status_label = {
        MatchStatus.EXACT: "Exact",
        MatchStatus.PARTIAL: "Partial",
        MatchStatus.UNMATCHED: "Unmatched",
        MatchStatus.MANUAL: "Manual",
    }.get(metadata.match_status, metadata.match_status.value.title())

    src = metadata.source_description.strip() if metadata.source_description else ""
    if src and src not in ("offline", "manual"):
        provenance = f"{status_label} ({src})"
    elif src == "offline":
        provenance = f"{status_label} (offline)"
    else:
        provenance = status_label

    parts = [part for part in (metadata.artist, metadata.title) if part]
    if parts:
        return f"{provenance} — {' - '.join(parts)}"
    return provenance


def format_match_tooltip(item: PlanItem) -> str:
    """Return full, untruncated metadata summary text for tooltips."""
    metadata = item.metadata
    if metadata is None:
        return "No resolved metadata."
    lines = [f"Status: {metadata.match_status.value} ({metadata.source_description or 'unknown'})"]
    if metadata.artist:
        lines.append(f"Artist: {metadata.artist}")
    if metadata.album_artist and metadata.album_artist != metadata.artist:
        lines.append(f"Album artist: {metadata.album_artist}")
    if metadata.title:
        lines.append(f"Title: {metadata.title}")
    if metadata.album:
        lines.append(f"Album: {metadata.album}")
    if metadata.track_number is not None:
        if metadata.track_total:
            lines.append(f"Track: {metadata.track_number}/{metadata.track_total}")
        else:
            lines.append(f"Track: {metadata.track_number}")
    if metadata.date:
        lines.append(f"Date: {metadata.date}")
    if metadata.genre:
        lines.append(f"Genre: {metadata.genre}")
    return "\n".join(lines)


class PlanTableModel(QAbstractTableModel):
    """Single table model owned by ``MainWindow`` and shared by Review and Run screens.

    Columns: Status, Source, Destination (relative to the output root), Action,
    Match, and Warnings.

    Roles: ``DisplayRole``, ``ToolTipRole`` (full text), ``DecorationRole``
    (status icon), ``AccessibleTextRole``, ``UserRole`` (``PlanItem``), and
    ``UserRole + 1`` (latest ``TrackStage``).
    """

    COLUMNS: tuple[str, ...] = (
        "Status",
        "Source",
        "Destination",
        "Action",
        "Match",
        "Warnings",
    )

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._items: list[PlanItem] = []
        self._output_root: Path | None = None
        self._stages: list[TrackStage] = []
        self._outcomes: list[RunOutcome | None] = []
        self._messages: list[str | None] = []
        self._event_warnings: list[list[str]] = []
        self._icon_cache: dict[str, QIcon] = {}

    @property
    def output_root(self) -> Path | None:
        """Return the output root directory used for relative Destination display."""
        return self._output_root

    def set_output_root(self, output_root: Path | None) -> None:
        """Set the output root directory used to format relative Destination paths."""
        if self._output_root == output_root:
            return
        self._output_root = output_root
        if self._items:
            top = self.index(0, COL_DESTINATION)
            bottom = self.index(len(self._items) - 1, COL_DESTINATION)
            self.dataChanged.emit(
                top,
                bottom,
                [
                    int(Qt.ItemDataRole.DisplayRole),
                    int(Qt.ItemDataRole.ToolTipRole),
                    int(Qt.ItemDataRole.AccessibleTextRole),
                ],
            )

    def items(self) -> list[PlanItem]:
        """Return a shallow copy of the current planned items list."""
        return list(self._items)

    def item_at(self, row: int) -> PlanItem | None:
        """Return the ``PlanItem`` at ``row``, or ``None`` if out of range."""
        if 0 <= row < len(self._items):
            return self._items[row]
        return None

    def stage_at(self, row: int) -> TrackStage | None:
        """Return the latest ``TrackStage`` for ``row``, or ``None`` if out of range."""
        if 0 <= row < len(self._stages):
            return self._stages[row]
        return None

    def outcome_at(self, row: int) -> RunOutcome | None:
        """Return the terminal Stage 2 ``RunOutcome`` for ``row``, if any."""
        if 0 <= row < len(self._outcomes):
            return self._outcomes[row]
        return None

    def message_at(self, row: int) -> str | None:
        """Return the latest event message (for errors or cancellations) for ``row``."""
        if 0 <= row < len(self._messages):
            return self._messages[row]
        return None

    def warnings_at(self, row: int) -> tuple[str, ...]:
        """Return deduplicated error, warning, and event notes for ``row``."""
        if not (0 <= row < len(self._items)):
            return ()
        item = self._items[row]
        notes: list[str] = []
        seen: set[str] = set()

        def _append(text: str | None) -> None:
            if not text:
                return
            cleaned = text.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                notes.append(cleaned)

        _append(item.error)
        _append(self._messages[row])
        for warning in item.warnings:
            _append(warning)
        for warning in self._event_warnings[row]:
            _append(warning)
        return tuple(notes)

    def status_info_at(self, row: int) -> tuple[str, str]:
        """Return ``(status_label, svg_icon_name)`` for ``row``."""
        item = self._items[row]
        stage = self._stages[row]
        outcome = self._outcomes[row]
        message = self._messages[row]

        if stage == TrackStage.PROBING:
            return ("Probing", "status_working")
        if stage == TrackStage.LOOKUP:
            return ("Looking up", "status_working")
        if stage == TrackStage.AWAITING_INPUT:
            return ("Waiting", "status_info")
        if stage == TrackStage.QUEUED:
            return ("Queued", "status_pending")
        if stage == TrackStage.EXTRACTING:
            return ("Extracting", "status_working")
        if stage == TrackStage.FINISHED:
            if outcome == RunOutcome.SUCCEEDED or outcome is None:
                return ("Done", "status_ok")
            if outcome == RunOutcome.UNMATCHED:
                return ("Done (unmatched)", "status_unmatched")
            if outcome == RunOutcome.SKIPPED:
                return ("Skipped", "status_skip")
            if outcome == RunOutcome.SKIPPED_NO_AUDIO:
                return ("Skipped (no audio)", "status_skip")
            if outcome == RunOutcome.FAILED:
                return ("Failed", "status_error")
        if stage == TrackStage.CANCELLED:
            if message == NOT_RUN_MESSAGE:
                return ("Not run", "status_cancelled")
            return ("Cancelled", "status_cancelled")

        # Default Stage 1 / PLANNED display derived from PlanItem.action and flags.
        if item.action == PlanAction.EXTRACT:
            if item.error:
                return ("Failed", "status_error")
            if item.warnings or item.conflict_note:
                return ("Attention", "status_warn")
            if item.match_decision == MatchDecision.AMBIGUOUS:
                return ("Attention", "status_warn")
            if item.metadata is not None and not is_offline_item(item):
                if item.metadata.match_status == MatchStatus.PARTIAL:
                    return ("Partial", "status_warn")
                if item.metadata.match_status == MatchStatus.UNMATCHED:
                    return ("Unmatched", "status_unmatched")
            return ("Ready", "status_extract")
        if is_skipped_action(item.action):
            return ("Skipped", "status_skip")
        return ("Failed", "status_error")

    def _icon_for(self, icon_name: str) -> QIcon:
        cached = self._icon_cache.get(icon_name)
        if cached is None:
            cached = load_icon(icon_name)
            self._icon_cache[icon_name] = cached
        return cached

    def flags(self, index: QModelIndex | QPersistentModelIndex) -> Qt.ItemFlag:
        """Return enabled, selectable read-only item flags."""
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def rowCount(
        self,
        parent: QModelIndex | QPersistentModelIndex = QModelIndex(),
    ) -> int:
        """Return the number of rows in the model."""
        if parent.isValid():
            return 0
        return len(self._items)

    def columnCount(
        self,
        parent: QModelIndex | QPersistentModelIndex = QModelIndex(),
    ) -> int:
        """Return the number of table columns."""
        if parent.isValid():
            return 0
        return len(self.COLUMNS)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = int(Qt.ItemDataRole.DisplayRole),
    ) -> object:
        """Return column header labels and accessible metadata."""
        if orientation == Qt.Orientation.Horizontal and 0 <= section < len(self.COLUMNS):
            label = self.COLUMNS[section]
            if role in (
                int(Qt.ItemDataRole.DisplayRole),
                int(Qt.ItemDataRole.ToolTipRole),
                int(Qt.ItemDataRole.AccessibleTextRole),
            ):
                return label
        return None

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = int(Qt.ItemDataRole.DisplayRole),
    ) -> object:
        """Return display, tooltip, decoration, accessible, or user data for ``index``."""
        if not index.isValid():
            return None
        row = index.row()
        col = index.column()
        if not (0 <= row < len(self._items)) or not (0 <= col < len(self.COLUMNS)):
            return None

        item = self._items[row]
        if role == PLAN_ITEM_ROLE:
            return item
        if role == TRACK_STAGE_ROLE:
            return self._stages[row]

        if role == int(Qt.ItemDataRole.DecorationRole):
            if col == COL_STATUS:
                _, icon_name = self.status_info_at(row)
                return self._icon_for(icon_name)
            return None

        if role == int(Qt.ItemDataRole.DisplayRole):
            return self._display_text(row, col, item)

        if role == int(Qt.ItemDataRole.ToolTipRole):
            return self._tooltip_text(row, col, item)

        if role == int(Qt.ItemDataRole.AccessibleTextRole):
            header = self.COLUMNS[col]
            text = self._display_text(row, col, item)
            if col == COL_WARNINGS and not text:
                return f"{header}: None"
            if col == COL_STATUS:
                return f"{header}: {text}, {item.media_input.path.name}"
            return f"{header}: {text}"

        return None

    def _display_text(self, row: int, col: int, item: PlanItem) -> str:
        if col == COL_STATUS:
            label, _ = self.status_info_at(row)
            return label
        if col == COL_SOURCE:
            return item.media_input.path.name
        if col == COL_DESTINATION:
            return format_destination(item.output_path, self._output_root)
        if col == COL_ACTION:
            return format_action_text(item)
        if col == COL_MATCH:
            return format_match_text(item)
        if col == COL_WARNINGS:
            notes = self.warnings_at(row)
            return "; ".join(notes)
        return ""

    def _tooltip_text(self, row: int, col: int, item: PlanItem) -> str:
        if col == COL_STATUS:
            label, _ = self.status_info_at(row)
            stage = self._stages[row]
            message = self._messages[row] or item.error
            if message:
                return f"{label} ({stage.value}): {message}"
            return f"{label} ({stage.value})"
        if col == COL_SOURCE:
            return str(item.media_input.path)
        if col == COL_DESTINATION:
            return str(item.output_path) if item.output_path is not None else "No destination path."
        if col == COL_ACTION:
            text = format_action_text(item)
            if item.conflict_note and item.conflict_note not in text:
                return f"{text} — {item.conflict_note}"
            return text
        if col == COL_MATCH:
            return format_match_tooltip(item)
        if col == COL_WARNINGS:
            notes = self.warnings_at(row)
            return "\n".join(notes) if notes else "No warnings."
        return ""

    @Slot(object)
    def set_items(self, items: list[PlanItem]) -> None:
        """Replace the current list of planned items and reset per-row runtime state."""
        new_items = items if isinstance(items, list) else list(items)
        self.beginResetModel()
        self._items = new_items
        count = len(self._items)
        self._stages = [TrackStage.PLANNED] * count
        self._outcomes = [None] * count
        self._messages = [None] * count
        self._event_warnings = [[] for _ in range(count)]
        self.endResetModel()

    @Slot(object)
    def apply_event(self, event: TrackEvent) -> None:
        """Update row state from a Stage 1 or Stage 2 ``TrackEvent``."""
        row = event.index
        if row < 0 or row >= len(self._items):
            return
        self._stages[row] = event.stage
        if event.stage in (TrackStage.QUEUED, TrackStage.EXTRACTING):
            self._outcomes[row] = None
        if event.outcome is not None:
            self._outcomes[row] = event.outcome
        if event.message is not None:
            self._messages[row] = event.message
        if event.warnings:
            bucket = self._event_warnings[row]
            for warn in event.warnings:
                cleaned = warn.strip()
                if cleaned and cleaned not in bucket:
                    bucket.append(cleaned)
        left = self.index(row, 0)
        right = self.index(row, len(self.COLUMNS) - 1)
        self.dataChanged.emit(left, right)

    @Slot(int)
    def refresh_row(self, row: int) -> None:
        """Notify attached views that ``row`` has changed."""
        if 0 <= row < len(self._items):
            left = self.index(row, 0)
            right = self.index(row, len(self.COLUMNS) - 1)
            self.dataChanged.emit(left, right)


class PlanFilterProxyModel(QSortFilterProxyModel):
    """Filter proxy supporting All, Needs attention, Skipped, and Failed views."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._mode = PlanFilterMode.ALL
        self.setDynamicSortFilter(False)

    @property
    def filter_mode(self) -> PlanFilterMode:
        """Return the active ``PlanFilterMode``."""
        return self._mode

    def set_filter_mode(self, mode: PlanFilterMode) -> None:
        """Update the active filter mode and re-evaluate visible rows."""
        if self._mode == mode:
            return
        self.beginFilterChange()
        self._mode = mode
        self.endFilterChange()
        self.setDynamicSortFilter(mode != PlanFilterMode.ALL)

    def filterAcceptsRow(
        self,
        source_row: int,
        source_parent: QModelIndex | QPersistentModelIndex,
    ) -> bool:
        """Return True if ``source_row`` matches the active filter mode."""
        if self._mode == PlanFilterMode.ALL:
            return True
        source = self.sourceModel()
        if source is None:
            return True
        idx = source.index(source_row, COL_STATUS, source_parent)
        item = source.data(idx, PLAN_ITEM_ROLE)
        if not isinstance(item, PlanItem):
            return True
        outcome = source.outcome_at(source_row) if isinstance(source, PlanTableModel) else None
        if self._mode == PlanFilterMode.NEEDS_ATTENTION:
            return is_needs_attention(item, outcome=outcome)
        if self._mode == PlanFilterMode.SKIPPED:
            return is_skipped_action(item.action)
        if self._mode == PlanFilterMode.FAILED:
            return is_failed_action(item.action) or outcome == RunOutcome.FAILED
        return True
