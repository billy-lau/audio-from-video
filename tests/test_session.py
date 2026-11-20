"""Tests for RunOptions, create_online_services, RunSession, and CLI test seams (C8)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from audio_from_video import planner, runner
from audio_from_video import session as session_module
from audio_from_video.artwork import ArtworkFetcher
from audio_from_video.binaries import ResolvedBinaries, resolve_binaries
from audio_from_video.cache import CACHE_FILENAME, TTLCache
from audio_from_video.cancel import CancelToken, OperationCancelled
from audio_from_video.cli import main as cli_main
from audio_from_video.config import AppConfig, ConfigError, load_config
from audio_from_video.events import LookupStats
from audio_from_video.extractor import ProcessRegistry
from audio_from_video.lookup import TrackResolver
from audio_from_video.models import (
    ConflictPolicy,
    ManualOverrides,
    MediaInput,
    PlanAction,
    RunSummary,
)
from audio_from_video.planner import PromptCallbacks
from audio_from_video.providers.base import HttpFetcher, RateLimiter
from audio_from_video.providers.itunes import ITunesClient
from audio_from_video.providers.musicbrainz import MusicBrainzClient
from audio_from_video.session import RunOptions, RunSession, create_online_services
from conftest import make_video_with_audio
from provider_fixtures import (
    TEST_UA,
    Router,
    credit,
    jpeg_bytes,
    medium,
    recording,
    release,
    release_lookup,
    search_payload,
)


def _dummy_binaries() -> ResolvedBinaries:
    return ResolvedBinaries(
        ffmpeg=Path("/fake/ffmpeg"),
        ffprobe=Path("/fake/ffprobe"),
    )


class _SpyFetcher:
    """Minimal fake fetcher that records set_sleep and close calls."""

    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        fail_close: bool = False,
        fail_set_sleep: bool = False,
        request_count: int = 0,
    ) -> None:
        self.name = name
        self._events = events
        self._fail_close = fail_close
        self._fail_set_sleep = fail_set_sleep
        self.request_count = request_count
        self.sleep_calls: list[Any] = []
        self.close_count = 0

    def set_sleep(self, fn: Any) -> None:
        self.sleep_calls.append(fn)
        if self._fail_set_sleep:
            raise RuntimeError(f"set_sleep failed for {self.name}")

    def close(self) -> None:
        self.close_count += 1
        self._events.append(f"close:{self.name}")
        if self._fail_close:
            raise RuntimeError(f"close failed for {self.name}")


def _build_spy_clients(
    events: list[str],
    *,
    fail_close_names: set[str] | None = None,
    fail_set_sleep_name: str | None = None,
) -> tuple[Any, Any, Any, Any, list[_SpyFetcher]]:
    bad_close = fail_close_names or set()
    mb_fetcher = _SpyFetcher(
        "mb.fetcher",
        events,
        fail_close="mb.fetcher" in bad_close,
        fail_set_sleep=fail_set_sleep_name == "mb.fetcher",
    )
    mb_art = _SpyFetcher(
        "mb.art_fetcher",
        events,
        fail_close="mb.art_fetcher" in bad_close,
        fail_set_sleep=fail_set_sleep_name == "mb.art_fetcher",
    )
    it_fetcher = _SpyFetcher(
        "itunes.fetcher",
        events,
        fail_close="itunes.fetcher" in bad_close,
        fail_set_sleep=fail_set_sleep_name == "itunes.fetcher",
    )
    it_art = _SpyFetcher(
        "itunes.art_fetcher",
        events,
        fail_close="itunes.art_fetcher" in bad_close,
        fail_set_sleep=fail_set_sleep_name == "itunes.art_fetcher",
    )
    mb_client = SimpleNamespace(fetcher=mb_fetcher, art_fetcher=mb_art, memo_hits=0)
    it_client = SimpleNamespace(fetcher=it_fetcher, art_fetcher=it_art, memo_hits=0)
    resolver = SimpleNamespace(tag="spy-resolver")
    artwork_fetcher = SimpleNamespace(tag="spy-artwork")
    return mb_client, it_client, resolver, artwork_fetcher, [mb_fetcher, mb_art, it_fetcher, it_art]


def _make_mock_daft_punk_router() -> Router:
    """Build a mock provider router matching 'Daft Punk - One More Time'."""
    credits = credit(("Daft Punk", "", "artist-dp", "Daft Punk"))
    rel = release(
        "rel-disc-orig",
        "Discovery",
        "rg-discovery",
        track_number=1,
        medium_track_count=14,
        date="2001-03-12",
        country="FR",
    )
    search_resp = search_payload(
        recording("rec-omt", "One More Time", credits, [rel], length_ms=600)
    )
    rel_detail = release_lookup(
        "rel-disc-orig",
        "Discovery",
        artist_credits=credits,
        rg_id="rg-discovery",
        rg_first_date="2001-03-12",
        date="2001-03-12",
        media=[medium(1, [("rec-omt", "One More Time")])],
        genres=[{"name": "Electronic", "count": 5}],
    )
    cover_bytes = jpeg_bytes(500, 500)

    router = Router()
    router.json(
        lambda r: r.url.host == "musicbrainz.org" and "/ws/2/recording/" in r.url.path,
        search_resp,
    )
    router.json(
        lambda r: r.url.host == "musicbrainz.org" and "/ws/2/release/rel-disc-orig" in r.url.path,
        rel_detail,
    )
    router.add(
        lambda r: r.url.host == "coverartarchive.org",
        lambda _r: httpx.Response(200, content=cover_bytes),
    )
    return router


# -- RunSession.open (offline and online) -----------------------------------


def test_offline_session_skips_cache_and_services_and_close_is_idempotent(tmp_path: Path) -> None:
    cache_dir = tmp_path / "never_created_cache"
    token = CancelToken()
    factory_called = False

    def _forbid_factory(*_args: Any) -> Any:
        nonlocal factory_called
        factory_called = True
        raise AssertionError("services_factory must not run in offline mode.")

    options = RunOptions(inputs=(tmp_path / "song.mp4",), offline=True)
    with RunSession.open(
        options,
        AppConfig(),
        _dummy_binaries(),
        token=token,
        user_agent=None,
        cache_dir=cache_dir,
        services_factory=_forbid_factory,
    ) as session:
        assert session.options is options
        assert session.token is token
        assert isinstance(session.registry, ProcessRegistry)
        assert session.cache is None
        assert session.resolver is None
        assert session.artwork_fetcher is None
        assert session.cache_warning is None
        assert session.lookup_stats() == LookupStats(cache_hits=0, mb_requests=0)
        assert not factory_called
        assert not cache_dir.exists()

    # Repeated closes after __exit__ remain no-ops and stats stay safe.
    session.close()
    session.close()
    assert session.lookup_stats() == LookupStats(cache_hits=0, mb_requests=0)


def test_each_session_gets_a_fresh_process_registry(tmp_path: Path) -> None:
    opts = RunOptions(inputs=(), offline=True)
    with (
        RunSession.open(
            opts,
            AppConfig(),
            _dummy_binaries(),
            token=CancelToken(),
            user_agent=None,
            cache_dir=tmp_path / "c1",
        ) as s1,
        RunSession.open(
            opts,
            AppConfig(),
            _dummy_binaries(),
            token=CancelToken(),
            user_agent=None,
            cache_dir=tmp_path / "c2",
        ) as s2,
    ):
        assert s1.registry is not s2.registry
        s1.registry.terminate_all()
        assert s1.registry.is_interrupted is True
        assert s2.registry.is_interrupted is False


@pytest.mark.parametrize("bad_ua", [None, "", "   "])
def test_online_session_requires_non_empty_user_agent(tmp_path: Path, bad_ua: str | None) -> None:
    cache_dir = tmp_path / "cache"
    factory_called = False

    def _forbid_factory(*_args: Any) -> Any:
        nonlocal factory_called
        factory_called = True
        raise AssertionError("Factory should not run without a user_agent.")

    with pytest.raises(ValueError, match="user_agent"):
        RunSession.open(
            RunOptions(inputs=(), offline=False),
            AppConfig(),
            _dummy_binaries(),
            token=CancelToken(),
            user_agent=bad_ua,
            cache_dir=cache_dir,
            services_factory=_forbid_factory,
        )
    assert not factory_called
    assert not cache_dir.exists()


def test_online_session_surfaces_cache_warning_for_corrupt_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / CACHE_FILENAME).write_bytes(b"not a valid sqlite database payload!")

    events: list[str] = []
    mb, it, resolver, artwork, _fetchers = _build_spy_clients(events)

    with RunSession.open(
        RunOptions(inputs=(), offline=False),
        AppConfig(),
        _dummy_binaries(),
        token=CancelToken(),
        user_agent=TEST_UA,
        cache_dir=cache_dir,
        services_factory=lambda _ua, _cfg, _cache: (mb, it, resolver, artwork),
    ) as session:
        assert session.cache is not None
        assert session.cache_warning is not None
        assert session.cache_warning == session.cache.warning
        assert "unreadable" in session.cache_warning.lower()


def test_create_online_services_and_cli_delegate_without_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("No HTTP request may happen when wiring services.")

    monkeypatch.setattr(httpx.Client, "send", _boom)

    cfg = AppConfig()
    cache = TTLCache.open_default(tmp_path / "cache")
    try:
        mb, it, resolver, artwork = create_online_services(TEST_UA, cfg, cache)
        try:
            assert isinstance(mb, MusicBrainzClient)
            assert isinstance(it, ITunesClient)
            assert isinstance(resolver, TrackResolver)
            assert isinstance(artwork, ArtworkFetcher)
            assert mb.cache is cache
            assert it.cache is cache
        finally:
            mb.fetcher.close()
            mb.art_fetcher.close()
            it.fetcher.close()
            it.art_fetcher.close()

        # Verify cli_main._create_online_services delegates to session.create_online_services.
        calls: list[tuple[str, AppConfig, TTLCache]] = []
        sentinel = (object(), object(), object(), object())

        def _fake_create(ua: str, c: AppConfig, ch: TTLCache) -> Any:
            calls.append((ua, c, ch))
            return sentinel

        monkeypatch.setattr(session_module, "create_online_services", _fake_create)
        assert cli_main._create_online_services(TEST_UA, cfg, cache) is sentinel
        assert calls == [(TEST_UA, cfg, cache)]
    finally:
        cache.close()


# -- Cancel-aware sleep wiring on all four fetchers -------------------------


def test_online_session_installs_token_sleep_on_all_four_spy_fetchers(tmp_path: Path) -> None:
    events: list[str] = []
    mb, it, resolver, artwork, fetchers = _build_spy_clients(events)
    token = CancelToken()

    with RunSession.open(
        RunOptions(inputs=(), offline=False),
        AppConfig(),
        _dummy_binaries(),
        token=token,
        user_agent=TEST_UA,
        cache_dir=tmp_path / "cache",
        services_factory=lambda _ua, _cfg, _cache: (mb, it, resolver, artwork),
    ):
        for fetcher in fetchers:
            assert len(fetcher.sleep_calls) == 1
            assert fetcher.sleep_calls[0] == token.sleep


def test_online_session_installs_token_sleep_on_all_four_real_http_fetchers(
    tmp_path: Path,
) -> None:
    token = CancelToken()
    router = Router()
    created: dict[str, Any] = {}

    def _factory(ua: str, cfg: AppConfig, cache: TTLCache) -> Any:
        mb = MusicBrainzClient(
            ua,
            fetcher=HttpFetcher(transport=router.transport, limiter=RateLimiter(1.1)),
            art_fetcher=HttpFetcher(transport=router.transport, limiter=RateLimiter(0.0)),
            cache=cache,
        )
        it = ITunesClient(
            fetcher=HttpFetcher(transport=router.transport, limiter=RateLimiter(3.0)),
            art_fetcher=HttpFetcher(transport=router.transport, limiter=RateLimiter(0.0)),
            cache=cache,
        )
        created["mb"] = mb
        created["it"] = it
        return mb, it, TrackResolver(mb, it, cfg.matching), ArtworkFetcher(mb, it, cache=cache)

    with RunSession.open(
        RunOptions(inputs=(), offline=False),
        AppConfig(),
        _dummy_binaries(),
        token=token,
        user_agent=TEST_UA,
        cache_dir=tmp_path / "cache",
        services_factory=_factory,
    ):
        all_four = (
            created["mb"].fetcher,
            created["mb"].art_fetcher,
            created["it"].fetcher,
            created["it"].art_fetcher,
        )
        for fetcher in all_four:
            assert fetcher._sleep == token.sleep
            assert fetcher.limiter._sleep == token.sleep

        # Cancelling the token interrupts the installed sleep immediately.
        token.cancel()
        for fetcher in all_four:
            with pytest.raises(OperationCancelled):
                fetcher._sleep(10.0)
            with pytest.raises(OperationCancelled):
                fetcher.limiter._sleep(10.0)


def test_online_session_handles_shared_fetcher_and_missing_set_sleep(tmp_path: Path) -> None:
    events: list[str] = []
    shared = _SpyFetcher("shared", events)
    bare = SimpleNamespace(close=lambda: events.append("close:bare"))
    mb_client = SimpleNamespace(fetcher=shared, art_fetcher=shared, memo_hits=0)
    it_client = SimpleNamespace(fetcher=bare, art_fetcher=None, memo_hits=0)
    token = CancelToken()

    with RunSession.open(
        RunOptions(inputs=(), offline=False),
        AppConfig(),
        _dummy_binaries(),
        token=token,
        user_agent=TEST_UA,
        cache_dir=tmp_path / "cache",
        services_factory=lambda _ua, _cfg, _cache: (
            mb_client,  # type: ignore[return-value]
            it_client,  # type: ignore[return-value]
            SimpleNamespace(),  # type: ignore[return-value]
            SimpleNamespace(),  # type: ignore[return-value]
        ),
    ):
        assert len(shared.sleep_calls) == 1
        assert shared.sleep_calls[0] == token.sleep

    assert shared.close_count == 1
    assert events == ["close:shared", "close:bare"]


# -- Cleanup on partial failure and close() idempotency / ordering ----------


def test_open_closes_cache_when_services_factory_raises(tmp_path: Path) -> None:
    opened_caches: list[TTLCache] = []

    def _failing_factory(_ua: str, _cfg: AppConfig, cache: TTLCache) -> Any:
        opened_caches.append(cache)
        assert cache._conn is not None
        raise RuntimeError("boom in factory")

    with pytest.raises(RuntimeError, match="boom in factory"):
        RunSession.open(
            RunOptions(inputs=(), offline=False),
            AppConfig(),
            _dummy_binaries(),
            token=CancelToken(),
            user_agent=TEST_UA,
            cache_dir=tmp_path / "cache",
            services_factory=_failing_factory,
        )

    assert len(opened_caches) == 1
    assert opened_caches[0]._conn is None


def test_open_closes_fetchers_and_cache_when_set_sleep_raises(tmp_path: Path) -> None:
    events: list[str] = []
    mb, it, resolver, artwork, fetchers = _build_spy_clients(
        events, fail_set_sleep_name="itunes.fetcher"
    )
    opened_caches: list[TTLCache] = []

    def _factory(_ua: str, _cfg: AppConfig, cache: TTLCache) -> Any:
        opened_caches.append(cache)
        orig_close = cache.close

        def _tracked_close() -> None:
            events.append("close:cache")
            orig_close()

        cache.close = _tracked_close  # type: ignore[method-assign]
        return mb, it, resolver, artwork

    with pytest.raises(RuntimeError, match=r"set_sleep failed for itunes\.fetcher"):
        RunSession.open(
            RunOptions(inputs=(), offline=False),
            AppConfig(),
            _dummy_binaries(),
            token=CancelToken(),
            user_agent=TEST_UA,
            cache_dir=tmp_path / "cache",
            services_factory=_factory,
        )

    assert events == [
        "close:mb.fetcher",
        "close:mb.art_fetcher",
        "close:itunes.fetcher",
        "close:itunes.art_fetcher",
        "close:cache",
    ]
    assert all(f.close_count == 1 for f in fetchers)
    assert opened_caches[0]._conn is None


def test_close_runs_fetchers_before_cache_and_is_idempotent(tmp_path: Path) -> None:
    events: list[str] = []
    mb, it, resolver, artwork, fetchers = _build_spy_clients(events)

    def _factory(_ua: str, _cfg: AppConfig, cache: TTLCache) -> Any:
        orig_close = cache.close

        def _tracked_close() -> None:
            events.append("close:cache")
            orig_close()

        cache.close = _tracked_close  # type: ignore[method-assign]
        return mb, it, resolver, artwork

    with RunSession.open(
        RunOptions(inputs=(), offline=False),
        AppConfig(),
        _dummy_binaries(),
        token=CancelToken(),
        user_agent=TEST_UA,
        cache_dir=tmp_path / "cache",
        services_factory=_factory,
    ) as session:
        session.close()
        session.close()

    # Exiting the `with` block calls close() a third time, still only one close each.
    session.close()

    assert events == [
        "close:mb.fetcher",
        "close:mb.art_fetcher",
        "close:itunes.fetcher",
        "close:itunes.art_fetcher",
        "close:cache",
    ]
    assert all(f.close_count == 1 for f in fetchers)
    assert session.cache is not None
    assert session.cache._conn is None


def test_close_continues_closing_remaining_resources_if_fetcher_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    events: list[str] = []
    mb, it, resolver, artwork, fetchers = _build_spy_clients(
        events, fail_close_names={"mb.fetcher", "itunes.fetcher"}
    )

    def _factory(_ua: str, _cfg: AppConfig, cache: TTLCache) -> Any:
        orig_close = cache.close

        def _tracked_close() -> None:
            events.append("close:cache")
            orig_close()

        cache.close = _tracked_close  # type: ignore[method-assign]
        return mb, it, resolver, artwork

    caplog.set_level(logging.WARNING, logger="audio_from_video.session")
    session = RunSession.open(
        RunOptions(inputs=(), offline=False),
        AppConfig(),
        _dummy_binaries(),
        token=CancelToken(),
        user_agent=TEST_UA,
        cache_dir=tmp_path / "cache",
        services_factory=_factory,
    )
    session.close()
    session.close()

    assert events == [
        "close:mb.fetcher",
        "close:mb.art_fetcher",
        "close:itunes.fetcher",
        "close:itunes.art_fetcher",
        "close:cache",
    ]
    assert all(f.close_count == 1 for f in fetchers)
    assert session.cache is not None
    assert session.cache._conn is None
    assert any("Error closing HTTP fetcher" in rec.message for rec in caplog.records)


def test_context_manager_closes_on_body_exception_even_if_close_errors(tmp_path: Path) -> None:
    events: list[str] = []
    mb, it, resolver, artwork, _fetchers = _build_spy_clients(
        events, fail_close_names={"mb.art_fetcher"}
    )

    def _factory(_ua: str, _cfg: AppConfig, cache: TTLCache) -> Any:
        orig_close = cache.close

        def _tracked_close() -> None:
            events.append("close:cache")
            orig_close()

        cache.close = _tracked_close  # type: ignore[method-assign]
        return mb, it, resolver, artwork

    with (
        pytest.raises(ValueError, match="caller failure"),
        RunSession.open(
            RunOptions(inputs=(), offline=False),
            AppConfig(),
            _dummy_binaries(),
            token=CancelToken(),
            user_agent=TEST_UA,
            cache_dir=tmp_path / "cache",
            services_factory=_factory,
        ),
    ):
        raise ValueError("caller failure")

    assert events == [
        "close:mb.fetcher",
        "close:mb.art_fetcher",
        "close:itunes.fetcher",
        "close:itunes.art_fetcher",
        "close:cache",
    ]


def test_concurrent_close_calls_close_each_resource_once(tmp_path: Path) -> None:
    events: list[str] = []
    mb, it, resolver, artwork, fetchers = _build_spy_clients(events)

    def _factory(_ua: str, _cfg: AppConfig, cache: TTLCache) -> Any:
        orig_close = cache.close

        def _tracked_close() -> None:
            events.append("close:cache")
            orig_close()

        cache.close = _tracked_close  # type: ignore[method-assign]
        return mb, it, resolver, artwork

    session = RunSession.open(
        RunOptions(inputs=(), offline=False),
        AppConfig(),
        _dummy_binaries(),
        token=CancelToken(),
        user_agent=TEST_UA,
        cache_dir=tmp_path / "cache",
        services_factory=_factory,
    )
    barrier = threading.Barrier(4)
    threads = [threading.Thread(target=lambda: (barrier.wait(), session.close())) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert all(f.close_count == 1 for f in fetchers)
    assert events.count("close:cache") == 1


def test_closed_session_rejects_build_plan_and_execute(tmp_path: Path) -> None:
    session = RunSession.open(
        RunOptions(inputs=(), offline=True),
        AppConfig(),
        _dummy_binaries(),
        token=CancelToken(),
        user_agent=None,
        cache_dir=tmp_path / "cache",
    )
    session.close()

    with pytest.raises(RuntimeError, match="closed"):
        session.build_plan(
            [],
            output_root=tmp_path / "out",
            prompt_callbacks=None,
            non_interactive=True,
        )
    with pytest.raises(RuntimeError, match="closed"):
        session.execute([], output_root=tmp_path / "out")


# -- lookup_stats() ---------------------------------------------------------


def test_lookup_stats_aggregates_cache_memo_and_mb_requests(tmp_path: Path) -> None:
    events: list[str] = []
    mb, it, resolver, artwork, _fetchers = _build_spy_clients(events)
    mb.memo_hits = 4
    it.memo_hits = 7
    mb.fetcher.request_count = 11
    mb.art_fetcher.request_count = 99  # Excluded: only mb_client.fetcher counts.
    it.fetcher.request_count = 55  # Excluded: only MusicBrainz requests count.

    with RunSession.open(
        RunOptions(inputs=(), offline=False),
        AppConfig(),
        _dummy_binaries(),
        token=CancelToken(),
        user_agent=TEST_UA,
        cache_dir=tmp_path / "cache",
        services_factory=lambda _ua, _cfg, _cache: (mb, it, resolver, artwork),
    ) as session:
        assert session.cache is not None
        session.cache.hits = 3
        assert session.lookup_stats() == LookupStats(cache_hits=14, mb_requests=11)

        mb.memo_hits += 2
        mb.fetcher.request_count += 1
        assert session.lookup_stats() == LookupStats(cache_hits=16, mb_requests=12)

    # Reading the final snapshot after close remains safe.
    assert session.lookup_stats() == LookupStats(cache_hits=16, mb_requests=12)


# -- RunOptions mapping in build_plan() and execute() -----------------------


def test_build_plan_and_execute_inherit_config_defaults_and_forward_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_calls: list[dict[str, Any]] = []
    exec_calls: list[dict[str, Any]] = []

    def _fake_build_plan(media_inputs: Any, **kwargs: Any) -> list[Any]:
        plan_calls.append({"media_inputs": media_inputs, **kwargs})
        return []

    def _fake_execute_plan(plan_items: Any, **kwargs: Any) -> RunSummary:
        exec_calls.append({"plan_items": plan_items, **kwargs})
        return RunSummary(output_root=kwargs["output_root"])

    monkeypatch.setattr(planner, "build_plan", _fake_build_plan)
    monkeypatch.setattr(runner, "execute_plan", _fake_execute_plan)

    cfg = AppConfig()
    cfg.output.flat = True
    cfg.output.on_conflict = ConflictPolicy.RENAME
    cfg.execution.jobs = 3

    binaries = _dummy_binaries()
    token = CancelToken()
    out_root = tmp_path / "out"
    media = [MediaInput(path=tmp_path / "a.mp4", from_directory_scan=False)]

    # Case 1: flat, on_conflict, and jobs omitted -> inherit AppConfig defaults.
    events_cb = lambda _e: None  # noqa: E731
    probe_cb = lambda _i, _t, _m: None  # noqa: E731
    lookup_cb = lambda _d, _t, _l: None  # noqa: E731
    done_cb = lambda: None  # noqa: E731
    warn_cb = lambda _w: None  # noqa: E731
    result_cb = lambda _r, _d, _t: None  # noqa: E731
    prompts = PromptCallbacks()

    with RunSession.open(
        RunOptions(inputs=(tmp_path / "a.mp4",), offline=True),
        cfg,
        binaries,
        token=token,
        user_agent=None,
        cache_dir=tmp_path / "cache",
    ) as session:
        session.build_plan(
            media,
            output_root=out_root,
            prompt_callbacks=prompts,
            non_interactive=False,
            on_event=events_cb,
            on_item_planned=probe_cb,
            on_lookup_progress=lookup_cb,
            on_lookup_complete=done_cb,
            on_batch_warning=warn_cb,
        )
        session.execute(
            [],
            output_root=out_root,
            on_result=result_cb,
            on_event=events_cb,
            install_signal_handlers=False,
        )

    assert plan_calls[-1]["flat"] is True
    assert plan_calls[-1]["on_conflict"] == ConflictPolicy.RENAME
    assert plan_calls[-1]["ffprobe_path"] == binaries.ffprobe
    assert plan_calls[-1]["config"] is cfg
    assert plan_calls[-1]["resolver"] is None
    assert plan_calls[-1]["cancel"] is token
    assert plan_calls[-1]["on_event"] is events_cb
    assert plan_calls[-1]["prompt_callbacks"] is prompts
    assert plan_calls[-1]["non_interactive"] is False
    assert plan_calls[-1]["on_item_planned"] is probe_cb
    assert plan_calls[-1]["on_lookup_progress"] is lookup_cb
    assert plan_calls[-1]["on_lookup_complete"] is done_cb
    assert plan_calls[-1]["on_batch_warning"] is warn_cb

    assert exec_calls[-1]["flat"] is True
    assert exec_calls[-1]["on_conflict"] == ConflictPolicy.RENAME
    assert exec_calls[-1]["jobs"] == 3
    assert exec_calls[-1]["ffmpeg_path"] == binaries.ffmpeg
    assert exec_calls[-1]["registry"] is session.registry
    assert exec_calls[-1]["artwork_fetcher"] is None
    assert exec_calls[-1]["on_result"] is result_cb
    assert exec_calls[-1]["on_event"] is events_cb
    assert exec_calls[-1]["install_signal_handlers"] is False
    assert exec_calls[-1]["cancel"] is token

    # Case 2: explicit False / overrides win over AppConfig defaults.
    overrides = ManualOverrides(artist="Artist A", album="Album B", title="Title C", track=2)
    explicit_opts = RunOptions(
        inputs=(tmp_path / "a.mp4",),
        offline=False,
        always_prompt=True,
        keep_native_missing=True,
        quality="192k",
        reencode=True,
        normalize=True,
        no_artwork=True,
        flat=False,
        on_conflict=ConflictPolicy.OVERWRITE,
        jobs=2,
        fail_fast=True,
        lang="jpn",
        stream=1,
        overrides=overrides,
    )
    events: list[str] = []
    mb, it, resolver, artwork, _fetchers = _build_spy_clients(events)
    online_token = CancelToken()

    with RunSession.open(
        explicit_opts,
        cfg,
        binaries,
        token=online_token,
        user_agent=TEST_UA,
        cache_dir=tmp_path / "cache2",
        services_factory=lambda _ua, _cfg, _cache: (mb, it, resolver, artwork),
    ) as online_session:
        online_session.build_plan(
            media,
            output_root=out_root,
            prompt_callbacks=None,
            non_interactive=True,
        )
        online_session.execute([], output_root=out_root)

    p = plan_calls[-1]
    assert p["flat"] is False
    assert p["on_conflict"] == ConflictPolicy.OVERWRITE
    assert p["quality_override"] == "192k"
    assert p["reencode"] is True
    assert p["normalize"] is True
    assert p["lang"] == "jpn"
    assert p["stream_index"] == 1
    assert p["always_prompt"] is True
    assert p["non_interactive"] is True
    assert p["keep_native_missing"] is True
    assert p["overrides"] is overrides
    assert p["resolver"] is resolver
    assert p["cancel"] is online_token

    e = exec_calls[-1]
    assert e["flat"] is False
    assert e["on_conflict"] == ConflictPolicy.OVERWRITE
    assert e["jobs"] == 2
    assert e["no_artwork"] is True
    assert e["fail_fast"] is True
    assert e["artwork_fetcher"] is artwork
    assert e["registry"] is online_session.registry
    assert e["install_signal_handlers"] is None
    assert e["cancel"] is online_token


def test_execute_validates_jobs_count(tmp_path: Path) -> None:
    with (
        RunSession.open(
            RunOptions(inputs=(), offline=True, jobs=0),
            AppConfig(),
            _dummy_binaries(),
            token=CancelToken(),
            user_agent=None,
            cache_dir=tmp_path / "cache",
        ) as session,
        pytest.raises(ConfigError, match="jobs"),
    ):
        session.execute([], output_root=tmp_path / "out")


def test_offline_session_end_to_end_extracts_mp3(tmp_path: Path) -> None:
    clip = make_video_with_audio(
        tmp_path / "Artist One - Sample Track.mp4",
        duration=0.4,
        metadata={"artist": "Artist One", "title": "Sample Track", "album": "Single"},
    )
    out_dir = tmp_path / "out"
    cfg = AppConfig()
    binaries = resolve_binaries(config=cfg)
    options = RunOptions(inputs=(clip,), offline=True, flat=True, jobs=1)

    with RunSession.open(
        options,
        cfg,
        binaries,
        token=CancelToken(),
        user_agent=None,
        cache_dir=tmp_path / "unused_cache",
    ) as session:
        items = session.build_plan(
            [MediaInput(path=clip, from_directory_scan=False)],
            output_root=out_dir,
            prompt_callbacks=None,
            non_interactive=True,
            on_event=lambda _ev: None,
        )
        assert len(items) == 1
        assert items[0].action == PlanAction.EXTRACT
        summary = session.execute(items, output_root=out_dir, install_signal_handlers=False)

    assert summary.exit_code == 0
    assert len(summary.results) == 1
    assert summary.results[0].output_path is not None
    assert summary.results[0].output_path.is_file()
    assert not (tmp_path / "unused_cache").exists()


# -- Real-HTTP guard tests for CLI test seams -------------------------------


def test_cli_extract_never_performs_real_http_when_services_are_monkeypatched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard test: CLI online run uses monkeypatched seams and touches zero real HTTP."""

    def _forbid_real_http(self: Any, request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected real HTTP request to {request.url}!")

    orig_send = httpx.Client.send

    def _checked_send(self: httpx.Client, request: httpx.Request, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(self._transport, httpx.MockTransport):
            raise AssertionError(f"Non-mock httpx.Client attempted request to {request.url}!")
        return orig_send(self, request, *args, **kwargs)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _forbid_real_http)
    monkeypatch.setattr(httpx.Client, "send", _checked_send)

    clip = make_video_with_audio(tmp_path / "Daft Punk - One More Time.mp4", duration=0.5)
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    cfg_path = tmp_path / "cfg" / "config.toml"
    report_path = tmp_path / "report.json"

    router = _make_mock_daft_punk_router()
    factory_calls: list[tuple[str, Path]] = []
    close_counts: dict[str, int] = {
        "mb.fetcher": 0,
        "mb.art_fetcher": 0,
        "it.fetcher": 0,
        "it.art_fetcher": 0,
    }
    sleep_installed: dict[str, int] = dict.fromkeys(close_counts, 0)

    def _wrap_fetcher(fetcher: HttpFetcher, key: str) -> HttpFetcher:
        orig_set_sleep = fetcher.set_sleep
        orig_close = fetcher.close

        def _tracked_set_sleep(fn: Any) -> None:
            sleep_installed[key] += 1
            orig_set_sleep(fn)

        def _tracked_close() -> None:
            close_counts[key] += 1
            orig_close()

        fetcher.set_sleep = _tracked_set_sleep  # type: ignore[method-assign]
        fetcher.close = _tracked_close  # type: ignore[method-assign]
        return fetcher

    def _mock_services(user_agent: str, cfg: AppConfig, cache: TTLCache) -> Any:
        factory_calls.append((user_agent, cache.path))
        mb = MusicBrainzClient(
            user_agent,
            fetcher=_wrap_fetcher(
                HttpFetcher(transport=router.transport, limiter=RateLimiter(0.0), max_retries=0),
                "mb.fetcher",
            ),
            art_fetcher=_wrap_fetcher(
                HttpFetcher(transport=router.transport, limiter=RateLimiter(0.0), max_retries=0),
                "mb.art_fetcher",
            ),
            cache=cache,
        )
        it = ITunesClient(
            fetcher=_wrap_fetcher(
                HttpFetcher(transport=router.transport, limiter=RateLimiter(0.0), max_retries=0),
                "it.fetcher",
            ),
            art_fetcher=_wrap_fetcher(
                HttpFetcher(transport=router.transport, limiter=RateLimiter(0.0), max_retries=0),
                "it.art_fetcher",
            ),
            cache=cache,
        )
        return mb, it, TrackResolver(mb, it, cfg.matching), ArtworkFetcher(mb, it, cache=cache)

    monkeypatch.delenv("AFV_MB_CONTACT", raising=False)
    monkeypatch.delenv("AFV_MB_USER_AGENT", raising=False)
    monkeypatch.setenv("AFV_CONFIG_PATH", str(cfg_path))
    monkeypatch.setattr(cli_main, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr(cli_main, "prompt_mb_contact", lambda **_kw: "relay@duck.com")
    monkeypatch.setattr(cli_main, "_create_online_services", _mock_services)
    monkeypatch.setattr(cli_main, "default_cache_dir", lambda: cache_dir)

    res = CliRunner().invoke(
        cli_main.app,
        ["-o", str(out_dir), "--report", str(report_path), str(clip)],
    )
    assert res.exit_code == 0, res.output
    assert len(factory_calls) == 1
    assert factory_calls[0][1] == cache_dir / CACHE_FILENAME
    assert load_config(config_path=cfg_path, env={}).musicbrainz.contact == "relay@duck.com"
    assert sleep_installed == dict.fromkeys(close_counts, 1)
    assert close_counts == dict.fromkeys(close_counts, 1)
    assert report_path.is_file()
    assert router.log  # Requests were served by MockTransport.


def test_cli_extract_blocks_httpx_client_send_unconditionally_with_fake_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even with httpx.Client.send raised hard, monkeypatched _create_online_services runs."""

    def _no_httpx_send(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("httpx.Client.send must never be called!")

    monkeypatch.setattr(httpx.Client, "send", _no_httpx_send)

    events: list[str] = []
    mb, it, _unused_resolver, _unused_artwork, fetchers = _build_spy_clients(events)

    class _FakeResolver:
        providers_unavailable = False

        def resolve_batch(
            self,
            items: Any,
            *,
            on_progress: Any = None,
            cancel: Any = None,
            on_item_started: Any = None,
        ) -> list[Any]:
            _ = cancel, on_item_started
            from audio_from_video.lookup import MatchResult
            from audio_from_video.models import MatchDecision, MatchStatus, TrackMetadata

            if on_progress is not None:
                on_progress(len(items), len(items), "done")
            return [
                MatchResult(
                    decision=MatchDecision.UNMATCHED,
                    best=None,
                    alternatives=[],
                    score=0.0,
                    margin=0.0,
                    queries_tried=[],
                    metadata=TrackMetadata(
                        artist="Unknown Artist",
                        title=item.source_path.stem if item.source_path else "Unknown",
                        match_status=MatchStatus.UNMATCHED,
                        source_description="offline",
                    ),
                )
                for item in items
            ]

    clip = make_video_with_audio(tmp_path / "Unknown Track.mp4", duration=0.4)
    cache_dir = tmp_path / "cache"
    factory_called = False

    def _fake_services(_ua: str, _cfg: AppConfig, _cache: TTLCache) -> Any:
        nonlocal factory_called
        factory_called = True
        return mb, it, _FakeResolver(), SimpleNamespace()

    monkeypatch.setenv("AFV_MB_CONTACT", "relay@duck.com")
    monkeypatch.setenv("AFV_CONFIG_PATH", str(tmp_path / "config.toml"))
    monkeypatch.setattr(cli_main, "_create_online_services", _fake_services)
    monkeypatch.setattr(cli_main, "default_cache_dir", lambda: cache_dir)

    res = CliRunner().invoke(
        cli_main.app,
        ["--dry-run", "--non-interactive", "-o", str(tmp_path / "out"), str(clip)],
    )
    assert res.exit_code == 0, res.output
    assert factory_called is True
    assert all(len(f.sleep_calls) == 1 for f in fetchers)
    assert all(f.close_count == 1 for f in fetchers)


def test_cli_surfaces_cache_warning_from_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / CACHE_FILENAME).write_bytes(b"corrupt cache bytes")

    clip = make_video_with_audio(tmp_path / "Daft Punk - One More Time.mp4", duration=0.4)
    router = _make_mock_daft_punk_router()

    def _mock_services(user_agent: str, cfg: AppConfig, cache: TTLCache) -> Any:
        mb = MusicBrainzClient(
            user_agent,
            fetcher=HttpFetcher(
                transport=router.transport, limiter=RateLimiter(0.0), max_retries=0
            ),
            art_fetcher=HttpFetcher(
                transport=router.transport, limiter=RateLimiter(0.0), max_retries=0
            ),
            cache=cache,
        )
        it = ITunesClient(
            fetcher=HttpFetcher(
                transport=router.transport, limiter=RateLimiter(0.0), max_retries=0
            ),
            art_fetcher=HttpFetcher(
                transport=router.transport, limiter=RateLimiter(0.0), max_retries=0
            ),
            cache=cache,
        )
        return mb, it, TrackResolver(mb, it, cfg.matching), ArtworkFetcher(mb, it, cache=cache)

    monkeypatch.setenv("AFV_MB_CONTACT", "relay@duck.com")
    monkeypatch.setenv("AFV_CONFIG_PATH", str(tmp_path / "config.toml"))
    monkeypatch.setattr(cli_main, "_create_online_services", _mock_services)
    monkeypatch.setattr(cli_main, "default_cache_dir", lambda: cache_dir)

    res = CliRunner().invoke(
        cli_main.app,
        ["--dry-run", "--non-interactive", "-o", str(tmp_path / "out"), str(clip)],
    )
    assert res.exit_code == 0, res.output
    assert "Warning:" in res.output
    assert "unreadable" in res.output.lower()


def test_lookup_stats_updates_from_real_clients_and_cache(tmp_path: Path) -> None:
    router = _make_mock_daft_punk_router()
    cache_dir = tmp_path / "cache"
    clients: dict[str, Any] = {}

    def _factory(ua: str, cfg: AppConfig, cache: TTLCache) -> Any:
        mb = MusicBrainzClient(
            ua,
            fetcher=HttpFetcher(
                transport=router.transport, limiter=RateLimiter(0.0), max_retries=0
            ),
            art_fetcher=HttpFetcher(
                transport=router.transport, limiter=RateLimiter(0.0), max_retries=0
            ),
            cache=cache,
        )
        it = ITunesClient(
            fetcher=HttpFetcher(
                transport=router.transport, limiter=RateLimiter(0.0), max_retries=0
            ),
            art_fetcher=HttpFetcher(
                transport=router.transport, limiter=RateLimiter(0.0), max_retries=0
            ),
            cache=cache,
        )
        clients["mb"] = mb
        clients["it"] = it
        return mb, it, TrackResolver(mb, it, cfg.matching), ArtworkFetcher(mb, it, cache=cache)

    # First session populates the SQLite cache and exercises client memo hits.
    with RunSession.open(
        RunOptions(inputs=(), offline=False),
        AppConfig(),
        _dummy_binaries(),
        token=CancelToken(),
        user_agent=TEST_UA,
        cache_dir=cache_dir,
        services_factory=_factory,
    ) as session1:
        assert session1.lookup_stats() == LookupStats(cache_hits=0, mb_requests=0)
        clients["mb"].search_recordings("One More Time", "Daft Punk")
        first = session1.lookup_stats()
        assert first.mb_requests == 1
        assert first.cache_hits == 0

        # Repeat search triggers a MusicBrainz in-memory memo hit without a new HTTP call.
        clients["mb"].search_recordings("One More Time", "Daft Punk")
        second = session1.lookup_stats()
        assert second.mb_requests == 1
        assert second.cache_hits == 1

    # Second session against the same cache_dir serves the query from SQLite (cache.hits == 1).
    with RunSession.open(
        RunOptions(inputs=(), offline=False),
        AppConfig(),
        _dummy_binaries(),
        token=CancelToken(),
        user_agent=TEST_UA,
        cache_dir=cache_dir,
        services_factory=_factory,
    ) as session2:
        clients["mb"].search_recordings("One More Time", "Daft Punk")
        cached = session2.lookup_stats()
        assert cached.mb_requests == 0
        assert cached.cache_hits == 1


def test_cli_cache_warning_suppressed_with_quiet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / CACHE_FILENAME).write_bytes(b"corrupt cache bytes")

    clip = make_video_with_audio(tmp_path / "Daft Punk - One More Time.mp4", duration=0.4)
    router = _make_mock_daft_punk_router()

    def _mock_services(user_agent: str, cfg: AppConfig, cache: TTLCache) -> Any:
        mb = MusicBrainzClient(
            user_agent,
            fetcher=HttpFetcher(
                transport=router.transport, limiter=RateLimiter(0.0), max_retries=0
            ),
            art_fetcher=HttpFetcher(
                transport=router.transport, limiter=RateLimiter(0.0), max_retries=0
            ),
            cache=cache,
        )
        it = ITunesClient(
            fetcher=HttpFetcher(
                transport=router.transport, limiter=RateLimiter(0.0), max_retries=0
            ),
            art_fetcher=HttpFetcher(
                transport=router.transport, limiter=RateLimiter(0.0), max_retries=0
            ),
            cache=cache,
        )
        return mb, it, TrackResolver(mb, it, cfg.matching), ArtworkFetcher(mb, it, cache=cache)

    monkeypatch.setenv("AFV_MB_CONTACT", "relay@duck.com")
    monkeypatch.setenv("AFV_CONFIG_PATH", str(tmp_path / "config.toml"))
    monkeypatch.setattr(cli_main, "_create_online_services", _mock_services)
    monkeypatch.setattr(cli_main, "default_cache_dir", lambda: cache_dir)

    res = CliRunner().invoke(
        cli_main.app,
        ["--dry-run", "--quiet", "--non-interactive", "-o", str(tmp_path / "out"), str(clip)],
    )
    assert res.exit_code == 0, res.output
    assert res.output == ""


def test_close_logs_and_swallows_cache_close_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    events: list[str] = []
    mb, it, resolver, artwork, fetchers = _build_spy_clients(events)

    def _factory(_ua: str, _cfg: AppConfig, cache: TTLCache) -> Any:
        orig_close = cache.close

        def _failing_close() -> None:
            events.append("close:cache")
            orig_close()
            raise RuntimeError("sqlite close boom")

        cache.close = _failing_close  # type: ignore[method-assign]
        return mb, it, resolver, artwork

    caplog.set_level(logging.WARNING, logger="audio_from_video.session")
    with RunSession.open(
        RunOptions(inputs=(), offline=False),
        AppConfig(),
        _dummy_binaries(),
        token=CancelToken(),
        user_agent=TEST_UA,
        cache_dir=tmp_path / "cache",
        services_factory=_factory,
    ):
        pass

    assert all(f.close_count == 1 for f in fetchers)
    assert events[-1] == "close:cache"
    assert any("Error closing TTL cache" in rec.message for rec in caplog.records)


def test_create_online_services_closes_mb_client_on_itunes_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[str] = []

    class _FakeMB:
        def __init__(self, _ua: str, *, cache: TTLCache) -> None:
            _ = cache
            self.fetcher = SimpleNamespace(close=lambda: closed.append("mb.fetcher"))
            self.art_fetcher = SimpleNamespace(close=lambda: closed.append("mb.art_fetcher"))

    def _boom_itunes(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("itunes init failed")

    monkeypatch.setattr(session_module, "MusicBrainzClient", _FakeMB)
    monkeypatch.setattr(session_module, "ITunesClient", _boom_itunes)

    cache = TTLCache.open_default(tmp_path / "cache")
    try:
        with pytest.raises(RuntimeError, match="itunes init failed"):
            create_online_services(TEST_UA, AppConfig(), cache)
    finally:
        cache.close()

    assert closed == ["mb.fetcher", "mb.art_fetcher"]
