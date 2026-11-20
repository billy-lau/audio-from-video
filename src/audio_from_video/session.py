"""Shared pipeline session, run options, and online service wiring for CLI and GUI."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

from audio_from_video.artwork import ArtworkFetcher
from audio_from_video.binaries import ResolvedBinaries
from audio_from_video.cache import TTLCache
from audio_from_video.cancel import CancelToken
from audio_from_video.config import AppConfig
from audio_from_video.events import EventSink, LookupStats
from audio_from_video.extractor import ProcessRegistry
from audio_from_video.lookup import TrackResolver
from audio_from_video.models import (
    ConflictPolicy,
    ManualOverrides,
    MediaInput,
    PlanItem,
    RunSummary,
    TrackResult,
)
from audio_from_video.planner import PromptCallbacks
from audio_from_video.providers.itunes import ITunesClient
from audio_from_video.providers.musicbrainz import MusicBrainzClient


@dataclass(frozen=True, slots=True)
class RunOptions:
    """Normalized per-run options shared by the CLI and the GUI."""

    inputs: tuple[Path, ...]
    recursive: bool = False
    output_dir: Path | None = None
    offline: bool = False
    always_prompt: bool = False
    keep_native_missing: bool = False
    quality: str | None = None
    reencode: bool = False
    normalize: bool = False
    no_artwork: bool = False
    flat: bool | None = None  # None uses the config default.
    on_conflict: ConflictPolicy | None = None  # None uses the config default.
    jobs: int | None = None  # None uses the config default.
    fail_fast: bool = False
    lang: str | None = None
    stream: int | None = None
    overrides: ManualOverrides = field(default_factory=ManualOverrides)


ServicesFactory = Callable[
    [str, AppConfig, TTLCache],
    tuple[MusicBrainzClient, ITunesClient, TrackResolver, ArtworkFetcher],
]


def create_online_services(
    user_agent: str,
    cfg: AppConfig,
    cache: TTLCache,
) -> tuple[MusicBrainzClient, ITunesClient, TrackResolver, ArtworkFetcher]:
    """Construct online MusicBrainz, iTunes, TrackResolver, and ArtworkFetcher services."""
    mb_client = MusicBrainzClient(user_agent, cache=cache)
    itunes_client = ITunesClient(cache=cache)
    resolver = TrackResolver(mb_client, itunes_client, cfg.matching)
    artwork_fetcher = ArtworkFetcher(mb_client, itunes_client, cache=cache)
    return mb_client, itunes_client, resolver, artwork_fetcher


class RunSession:
    """Context manager holding resources for one planning and extraction attempt."""

    options: RunOptions
    config: AppConfig
    binaries: ResolvedBinaries
    token: CancelToken
    registry: ProcessRegistry
    cache: TTLCache | None
    resolver: TrackResolver | None
    artwork_fetcher: ArtworkFetcher | None
    cache_warning: str | None

    def __enter__(self) -> RunSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        _ = exc_type, exc_val, exc_tb
        self.close()

    @classmethod
    def open(
        cls,
        options: RunOptions,
        config: AppConfig,
        binaries: ResolvedBinaries,
        *,
        token: CancelToken,
        user_agent: str | None,
        cache_dir: Path,
        services_factory: ServicesFactory = create_online_services,
    ) -> RunSession:
        """Open a fresh RunSession with its cache, online clients, and ProcessRegistry."""
        _ = options, config, binaries, token, user_agent, cache_dir, services_factory
        raise NotImplementedError("RunSession.open is implemented in Stage 1 lane D.")

    def build_plan(
        self,
        media_inputs: Sequence[MediaInput],
        *,
        output_root: Path,
        prompt_callbacks: PromptCallbacks | None,
        non_interactive: bool,
        on_event: EventSink | None = None,
        on_item_planned: Callable[[int, int, MediaInput], None] | None = None,
        on_lookup_progress: Callable[[int, int, str], None] | None = None,
        on_lookup_complete: Callable[[], None] | None = None,
        on_batch_warning: Callable[[str], None] | None = None,
    ) -> list[PlanItem]:
        """Run Stage 1 planning with this session's options, config, resolver, and token."""
        _ = (
            media_inputs,
            output_root,
            prompt_callbacks,
            non_interactive,
            on_event,
            on_item_planned,
            on_lookup_progress,
            on_lookup_complete,
            on_batch_warning,
        )
        raise NotImplementedError("RunSession.build_plan is implemented in Stage 1 lane D.")

    def execute(
        self,
        plan_items: Sequence[PlanItem],
        *,
        output_root: Path,
        on_result: Callable[[TrackResult, int, int], None] | None = None,
        on_event: EventSink | None = None,
        install_signal_handlers: bool | None = None,
    ) -> RunSummary:
        """Run Stage 2 execution with this session's options, binaries, registry, and token."""
        _ = plan_items, output_root, on_result, on_event, install_signal_handlers
        raise NotImplementedError("RunSession.execute is implemented in Stage 1 lane D.")

    def lookup_stats(self) -> LookupStats:
        """Return a snapshot of cache hits and MusicBrainz requests (worker-thread only)."""
        raise NotImplementedError("RunSession.lookup_stats is implemented in Stage 1 lane D.")

    def close(self) -> None:
        """Close HTTP fetchers and then the SQLite cache; idempotent."""
        raise NotImplementedError("RunSession.close is implemented in Stage 1 lane D.")
