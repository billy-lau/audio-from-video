"""Shared pipeline session, run options, and online service wiring for CLI and GUI."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

from audio_from_video import planner, runner
from audio_from_video.artwork import ArtworkFetcher
from audio_from_video.binaries import ResolvedBinaries
from audio_from_video.cache import TTLCache
from audio_from_video.cancel import CancelToken
from audio_from_video.config import AppConfig, validate_jobs
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

logger = logging.getLogger(__name__)


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


def _iter_fetchers(
    mb_client: MusicBrainzClient | None,
    itunes_client: ITunesClient | None,
) -> list[Any]:
    """Collect unique provider HTTP fetchers in deterministic shutdown order."""
    fetchers: list[Any] = []
    seen_ids: set[int] = set()
    for client in (mb_client, itunes_client):
        if client is None:
            continue
        for attr in ("fetcher", "art_fetcher"):
            fetcher = getattr(client, attr, None)
            if fetcher is not None and id(fetcher) not in seen_ids:
                seen_ids.add(id(fetcher))
                fetchers.append(fetcher)
    return fetchers


def _close_resources(
    mb_client: MusicBrainzClient | None,
    itunes_client: ITunesClient | None,
    cache: TTLCache | None,
) -> None:
    """Close each HTTP fetcher first, and then close the SQLite cache.

    Errors are logged rather than raised so one failing fetcher never prevents
    remaining fetchers or the cache from closing cleanly.
    """
    try:
        for fetcher in _iter_fetchers(mb_client, itunes_client):
            close_fn = getattr(fetcher, "close", None)
            if not callable(close_fn):
                continue
            try:
                close_fn()
            except Exception:
                logger.warning("Error closing HTTP fetcher %r.", fetcher, exc_info=True)
    finally:
        if cache is not None:
            close_cache = getattr(cache, "close", None)
            if callable(close_cache):
                try:
                    close_cache()
                except Exception:
                    logger.warning("Error closing TTL cache %r.", cache, exc_info=True)


def create_online_services(
    user_agent: str,
    cfg: AppConfig,
    cache: TTLCache,
) -> tuple[MusicBrainzClient, ITunesClient, TrackResolver, ArtworkFetcher]:
    """Construct online MusicBrainz, iTunes, TrackResolver, and ArtworkFetcher services."""
    mb_client: MusicBrainzClient | None = None
    itunes_client: ITunesClient | None = None
    try:
        mb_client = MusicBrainzClient(user_agent, cache=cache)
        itunes_client = ITunesClient(cache=cache)
        resolver = TrackResolver(mb_client, itunes_client, cfg.matching)
        artwork_fetcher = ArtworkFetcher(mb_client, itunes_client, cache=cache)
    except BaseException:
        _close_resources(mb_client, itunes_client, None)
        raise
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

    def __init__(
        self,
        *,
        options: RunOptions,
        config: AppConfig,
        binaries: ResolvedBinaries,
        token: CancelToken,
        registry: ProcessRegistry,
        cache: TTLCache | None,
        mb_client: MusicBrainzClient | None,
        itunes_client: ITunesClient | None,
        resolver: TrackResolver | None,
        artwork_fetcher: ArtworkFetcher | None,
        cache_warning: str | None,
    ) -> None:
        self.options = options
        self.config = config
        self.binaries = binaries
        self.token = token
        self.registry = registry
        self.cache = cache
        self._mb_client = mb_client
        self._itunes_client = itunes_client
        self.resolver = resolver
        self.artwork_fetcher = artwork_fetcher
        self.cache_warning = cache_warning
        self._close_lock = threading.Lock()
        self._closed = False

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
        registry = ProcessRegistry()
        if options.offline:
            return cls(
                options=options,
                config=config,
                binaries=binaries,
                token=token,
                registry=registry,
                cache=None,
                mb_client=None,
                itunes_client=None,
                resolver=None,
                artwork_fetcher=None,
                cache_warning=None,
            )

        if not user_agent or not user_agent.strip():
            raise ValueError(
                "A non-empty user_agent is required when opening an online RunSession."
            )

        cache = TTLCache.open_default(cache_dir)
        cache_warning = cache.warning
        mb_client: MusicBrainzClient | None = None
        itunes_client: ITunesClient | None = None
        try:
            mb_client, itunes_client, resolver, artwork_fetcher = services_factory(
                user_agent, config, cache
            )
            sleep_fn = token.sleep
            for fetcher in _iter_fetchers(mb_client, itunes_client):
                set_sleep = getattr(fetcher, "set_sleep", None)
                if callable(set_sleep):
                    set_sleep(sleep_fn)
        except BaseException:
            _close_resources(mb_client, itunes_client, cache)
            raise

        return cls(
            options=options,
            config=config,
            binaries=binaries,
            token=token,
            registry=registry,
            cache=cache,
            mb_client=mb_client,
            itunes_client=itunes_client,
            resolver=resolver,
            artwork_fetcher=artwork_fetcher,
            cache_warning=cache_warning,
        )

    def _effective_flat(self) -> bool:
        return self.config.output.flat if self.options.flat is None else self.options.flat

    def _effective_conflict(self) -> ConflictPolicy:
        return (
            self.config.output.on_conflict
            if self.options.on_conflict is None
            else self.options.on_conflict
        )

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
        if self._closed:
            raise RuntimeError("Cannot call build_plan on a closed RunSession.")
        return planner.build_plan(
            media_inputs,
            output_root=output_root,
            ffprobe_path=self.binaries.ffprobe,
            config=self.config,
            resolver=self.resolver,
            overrides=self.options.overrides,
            quality_override=self.options.quality,
            reencode=self.options.reencode,
            normalize=self.options.normalize,
            flat=self._effective_flat(),
            on_conflict=self._effective_conflict(),
            lang=self.options.lang,
            stream_index=self.options.stream,
            always_prompt=self.options.always_prompt,
            non_interactive=non_interactive,
            keep_native_missing=self.options.keep_native_missing,
            prompt_callbacks=prompt_callbacks,
            on_item_planned=on_item_planned,
            on_lookup_progress=on_lookup_progress,
            on_lookup_complete=on_lookup_complete,
            on_batch_warning=on_batch_warning,
            cancel=self.token,
            on_event=on_event,
        )

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
        if self._closed:
            raise RuntimeError("Cannot call execute on a closed RunSession.")
        raw_jobs = self.config.execution.jobs if self.options.jobs is None else self.options.jobs
        effective_jobs = validate_jobs(raw_jobs)
        return runner.execute_plan(
            plan_items,
            ffmpeg_path=self.binaries.ffmpeg,
            output_root=output_root,
            jobs=effective_jobs,
            flat=self._effective_flat(),
            on_conflict=self._effective_conflict(),
            no_artwork=self.options.no_artwork,
            fail_fast=self.options.fail_fast,
            artwork_fetcher=self.artwork_fetcher,
            registry=self.registry,
            on_result=on_result,
            install_signal_handlers=install_signal_handlers,
            cancel=self.token,
            on_event=on_event,
        )

    def lookup_stats(self) -> LookupStats:
        """Return a snapshot of cache hits and MusicBrainz requests (worker-thread only)."""
        cache_hits = (
            (self.cache.hits if self.cache is not None else 0)
            + (getattr(self._mb_client, "memo_hits", 0) if self._mb_client is not None else 0)
            + (
                getattr(self._itunes_client, "memo_hits", 0)
                if self._itunes_client is not None
                else 0
            )
        )
        mb_fetcher = (
            getattr(self._mb_client, "fetcher", None) if self._mb_client is not None else None
        )
        mb_requests = getattr(mb_fetcher, "request_count", 0) if mb_fetcher is not None else 0
        return LookupStats(cache_hits=cache_hits, mb_requests=mb_requests)

    def close(self) -> None:
        """Close HTTP fetchers and then the SQLite cache; idempotent."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            mb_client = self._mb_client
            itunes_client = self._itunes_client
            cache = self.cache
        _close_resources(mb_client, itunes_client, cache)
