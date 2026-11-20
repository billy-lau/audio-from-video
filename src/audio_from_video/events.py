"""Progress events and lookup statistics emitted by Stage 1 and Stage 2 core helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from audio_from_video.models import PlanAction, RunOutcome

NOT_RUN_MESSAGE = "not run: stopped at first failure"


class TrackStage(StrEnum):
    """Lifecycle stage reported for a single track during planning and execution."""

    PROBING = "probing"
    LOOKUP = "lookup"
    AWAITING_INPUT = "awaiting_input"
    PLANNED = "planned"
    QUEUED = "queued"
    EXTRACTING = "extracting"
    FINISHED = "finished"
    CANCELLED = "cancelled"  # FINISHED and CANCELLED are terminal.


@dataclass(frozen=True, slots=True)
class TrackEvent:
    """Progress event emitted for one input item (or batch-level notice at index -1)."""

    index: int  # 0-based position in media_inputs; -1 for a batch-level notice.
    source: Path | None
    stage: TrackStage
    action: PlanAction | None = None
    outcome: RunOutcome | None = None
    warnings: tuple[str, ...] = ()
    message: str | None = None  # For CANCELLED: NOT_RUN_MESSAGE when stopped by fail-fast.


@dataclass(frozen=True, slots=True)
class LookupStats:
    """Snapshot of Stage 1 online cache hits and MusicBrainz HTTP requests."""

    cache_hits: int  # cache.hits plus both clients' memo_hits.
    mb_requests: int  # mb_client.fetcher.request_count.


EventSink = Callable[[TrackEvent], None]
