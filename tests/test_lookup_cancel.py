"""Tests for cooperative cancellation in TrackResolver and provider fetchers (C3 and C4).

All tests use httpx.MockTransport and synthetic responses; no test contacts a live
MusicBrainz, iTunes, or Cover Art Archive endpoint.
"""

from __future__ import annotations

import ast
import inspect
import threading
import time
from datetime import UTC, datetime
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest

from audio_from_video.cache import TTLCache
from audio_from_video.cancel import CancelToken, OperationCancelled
from audio_from_video.lookup import BatchItem, TrackResolver
from audio_from_video.models import Candidate, MatchDecision, ParsedHints, ReleaseRef
from audio_from_video.providers import base as providers_base
from audio_from_video.providers import itunes as providers_itunes
from audio_from_video.providers import musicbrainz as providers_musicbrainz
from audio_from_video.providers.base import HttpFetcher, RateLimiter
from audio_from_video.providers.itunes import ITunesClient
from audio_from_video.providers.musicbrainz import MusicBrainzClient
from provider_fixtures import (
    TEST_UA,
    Router,
    credit,
    fetcher,
    make_itunes,
    make_mb,
    medium,
    path_is,
    query_has,
    recording,
    release,
    release_lookup,
    search_payload,
)


def _auto_accept_payload(
    title: str,
    artist: str = "Artist",
    *,
    rec_id: str,
    rel_id: str,
    rg_id: str = "rg-album",
    album: str = "Studio Album",
) -> dict:
    return search_payload(
        recording(
            rec_id,
            title,
            credit((artist, "")),
            [release(rel_id, album, rg_id)],
            length_ms=200_000,
        )
    )


def _release_payload(
    rel_id: str,
    title: str,
    rec_id: str,
    *,
    artist: str = "Artist",
    rg_id: str = "rg-album",
    album: str = "Studio Album",
) -> dict:
    return release_lookup(
        rel_id,
        album,
        artist_credits=credit((artist, "")),
        rg_id=rg_id,
        rg_first_date="2020-01-01",
        date="2020-01-01",
        media=[medium(1, [(rec_id, title)])],
        rg_genres=[{"name": "rock", "count": 5}],
    )


def _mb_router_for_tracks(titles: list[str], *, album: str = "Studio Album") -> Router:
    router = Router()
    for idx, title in enumerate(titles):
        rec_id = f"rec-{idx}"
        rel_id = f"rel-{idx}"
        router.json(
            query_has(title),
            _auto_accept_payload(title, rec_id=rec_id, rel_id=rel_id, album=album),
        )
        router.json(
            path_is(f"/ws/2/release/{rel_id}"),
            _release_payload(rel_id, title, rec_id, album=album),
        )
    return router


def _block_time_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(seconds: float) -> None:
        raise AssertionError(f"Unexpected direct time.sleep({seconds}) call")

    monkeypatch.setattr(time, "sleep", _fail)


# -- C3: resolve_batch cancellation and on_item_started ----------------------------


def test_resolve_batch_empty_and_precancelled() -> None:
    router = Router()
    resolver = TrackResolver(make_mb(router), make_itunes(router))
    started: list[int] = []
    progress: list[tuple[int, int, str]] = []

    assert (
        resolver.resolve_batch(
            [],
            on_progress=lambda d, t, lbl: progress.append((d, t, lbl)),
            on_item_started=started.append,
        )
        == []
    )
    assert started == []
    assert progress == []

    token = CancelToken()
    token.cancel()
    with pytest.raises(OperationCancelled):
        resolver.resolve_batch([], cancel=token, on_item_started=started.append)
    assert started == []
    assert router.count() == 0

    items = [BatchItem(ParsedHints(artist="Artist", title="Alpha"), duration_seconds=200.0)]
    with pytest.raises(OperationCancelled):
        resolver.resolve_batch(
            items,
            cancel=token,
            on_progress=lambda d, t, lbl: progress.append((d, t, lbl)),
            on_item_started=started.append,
        )
    assert started == []
    assert progress == []
    assert router.count() == 0
    assert resolver.providers_unavailable is False


def test_resolve_batch_on_item_started_single_items_and_hintless() -> None:
    router = _mb_router_for_tracks(["Alpha", "Gamma"])
    resolver = TrackResolver(make_mb(router), None)
    items = [
        BatchItem(ParsedHints(artist="Artist", title="Alpha"), duration_seconds=200.0),
        BatchItem(ParsedHints(is_hintless=True, clean_stem="VID_0001")),
        BatchItem(ParsedHints(artist="Artist", title="Gamma"), duration_seconds=200.0),
    ]
    started: list[int] = []
    results = resolver.resolve_batch(items, on_item_started=started.append)
    assert len(results) == 3
    assert results[0].decision == MatchDecision.AUTO_ACCEPT
    assert results[1].decision == MatchDecision.SKIPPED
    assert results[2].decision == MatchDecision.AUTO_ACCEPT
    assert started == [0, 1, 2]


def test_resolve_batch_on_item_started_grouped_and_interleaved_no_duplicates() -> None:
    titles = [f"Track {i}" for i in range(7)]
    router = _mb_router_for_tracks(titles)
    resolver = TrackResolver(make_mb(router), None)
    items = [
        BatchItem(ParsedHints(artist="Artist", title="Track 0"), duration_seconds=200.0),
        BatchItem(
            ParsedHints(artist="Artist", title="Track 1"),
            duration_seconds=200.0,
            group_key="album-a",
        ),
        BatchItem(ParsedHints(artist="Artist", title="Track 2"), duration_seconds=200.0),
        BatchItem(
            ParsedHints(artist="Artist", title="Track 3"),
            duration_seconds=200.0,
            group_key="album-a",
        ),
        BatchItem(
            ParsedHints(artist="Artist", title="Track 4", parent_album_hint="Album B"),
            duration_seconds=200.0,
            source_path=Path("dir_b/01.mp4"),
        ),
        BatchItem(
            ParsedHints(artist="Artist", title="Track 5", parent_album_hint="Album B"),
            duration_seconds=200.0,
            source_path=Path("dir_b/02.mp4"),
        ),
        # Single-item folder is treated as ungrouped.
        BatchItem(
            ParsedHints(artist="Artist", title="Track 6", parent_album_hint="Solo"),
            duration_seconds=200.0,
            source_path=Path("solo/01.mp4"),
        ),
    ]
    started: list[int] = []
    results = resolver.resolve_batch(items, on_item_started=started.append)
    assert len(results) == 7
    assert started == [0, 1, 3, 2, 4, 5, 6]
    assert len(started) == len(set(started)) == len(items)
    assert all(0 <= idx < len(items) for idx in started)
    assert [r.metadata.title for r in results] == titles


def test_resolve_batch_cancel_after_first_single_item_via_progress() -> None:
    router = _mb_router_for_tracks(["Alpha", "Beta", "Gamma"])
    resolver = TrackResolver(make_mb(router), None)
    token = CancelToken()
    started: list[int] = []

    def on_progress(done: int, _total: int, _label: str) -> None:
        if done == 1:
            token.cancel()

    items = [
        BatchItem(ParsedHints(artist="Artist", title=name), duration_seconds=200.0)
        for name in ("Alpha", "Beta", "Gamma")
    ]
    with pytest.raises(OperationCancelled):
        resolver.resolve_batch(
            items,
            cancel=token,
            on_progress=on_progress,
            on_item_started=started.append,
        )

    assert started == [0]
    queried = [req.url.params.get("query", "") for req in router.log]
    assert any("Alpha" in q for q in queried)
    assert not any("Beta" in q or "Gamma" in q for q in queried)
    assert resolver.providers_unavailable is False


def test_resolve_batch_cancel_inside_transport_skips_enrichment_and_remaining_items() -> None:
    token = CancelToken()
    router = Router()

    def _search_alpha(_req: httpx.Request) -> httpx.Response:
        token.cancel()
        return httpx.Response(
            200,
            json=_auto_accept_payload("Alpha", rec_id="rec-0", rel_id="rel-0"),
        )

    router.add(query_has("Alpha"), _search_alpha)
    router.json(
        path_is("/ws/2/release/rel-0"),
        _release_payload("rel-0", "Alpha", "rec-0"),
    )
    router.json(
        query_has("Beta"),
        _auto_accept_payload("Beta", rec_id="rec-1", rel_id="rel-1"),
    )

    resolver = TrackResolver(make_mb(router), make_itunes(router))
    started: list[int] = []
    items = [
        BatchItem(ParsedHints(artist="Artist", title="Alpha"), duration_seconds=200.0),
        BatchItem(ParsedHints(artist="Artist", title="Beta"), duration_seconds=200.0),
    ]
    with pytest.raises(OperationCancelled):
        resolver.resolve_batch(items, cancel=token, on_item_started=started.append)

    assert started == [0]
    assert router.count() == 1
    assert resolver.providers_unavailable is False


def test_resolve_batch_cancel_inside_on_item_started_stops_before_request() -> None:
    router = _mb_router_for_tracks(["Alpha", "Beta", "Gamma"])
    resolver = TrackResolver(make_mb(router), None)
    token = CancelToken()
    started: list[int] = []

    def _on_started(idx: int) -> None:
        started.append(idx)
        if idx == 1:
            token.cancel()

    items = [
        BatchItem(ParsedHints(artist="Artist", title=name), duration_seconds=200.0)
        for name in ("Alpha", "Beta", "Gamma")
    ]
    with pytest.raises(OperationCancelled):
        resolver.resolve_batch(items, cancel=token, on_item_started=_on_started)

    assert started == [0, 1]
    assert not any("Beta" in req.url.params.get("query", "") for req in router.log)
    assert not any("Gamma" in req.url.params.get("query", "") for req in router.log)


def test_resolve_batch_cancel_inside_grouped_on_item_started() -> None:
    router = _mb_router_for_tracks(["Track A", "Track B", "Track C"])
    resolver = TrackResolver(make_mb(router), None)
    token = CancelToken()
    started: list[int] = []

    def _on_started(idx: int) -> None:
        started.append(idx)
        if idx == 1:
            token.cancel()

    items = [
        BatchItem(
            ParsedHints(artist="Artist", title=title),
            duration_seconds=200.0,
            group_key="album-group",
        )
        for title in ("Track A", "Track B", "Track C")
    ]
    with pytest.raises(OperationCancelled):
        resolver.resolve_batch(items, cancel=token, on_item_started=_on_started)

    assert started == [0, 1]
    assert router.count() == 1
    assert any("Track A" in req.url.params.get("query", "") for req in router.log)
    assert not any("/release/" in req.url.path for req in router.log)


def test_resolve_batch_cancel_stops_unfiltered_folder_fallback_in_search() -> None:
    token = CancelToken()
    router = Router()

    def _folder_filtered(_req: httpx.Request) -> httpx.Response:
        token.cancel()
        # Return a hit whose album does NOT match parent_album_hint, which would
        # otherwise trigger an unfiltered MusicBrainz fallback query.
        return httpx.Response(
            200,
            json=_auto_accept_payload(
                "Target",
                rec_id="rec-other",
                rel_id="rel-other",
                album="Live Elsewhere",
            ),
        )

    router.add(lambda req: req.url.path.endswith("/recording/"), _folder_filtered)
    resolver = TrackResolver(make_mb(router), None)
    hints = ParsedHints(
        artist="Artist",
        title="Target",
        parent_artist_hint="Artist",
        parent_album_hint="Wrong Folder",
        track_number=1,
    )
    with pytest.raises(OperationCancelled):
        resolver.resolve_batch([BatchItem(hints, duration_seconds=200.0)], cancel=token)

    assert router.count() == 1
    assert resolver.providers_unavailable is False


def test_resolve_batch_cancel_mid_group_gather_stops_later_members() -> None:
    token = CancelToken()
    router = Router()
    router.json(
        query_has("Track A"),
        _auto_accept_payload("Track A", rec_id="rec-a", rel_id="rel-a"),
    )

    def _track_b(_req: httpx.Request) -> httpx.Response:
        token.cancel()
        return httpx.Response(
            200,
            json=_auto_accept_payload("Track B", rec_id="rec-b", rel_id="rel-b"),
        )

    router.add(query_has("Track B"), _track_b)
    router.json(
        query_has("Track C"),
        _auto_accept_payload("Track C", rec_id="rec-c", rel_id="rel-c"),
    )

    resolver = TrackResolver(make_mb(router), None)
    items = [
        BatchItem(
            ParsedHints(artist="Artist", title=title, parent_album_hint="Studio Album"),
            duration_seconds=200.0,
            source_path=Path(f"album/{idx:02d}.mp4"),
        )
        for idx, title in enumerate(("Track A", "Track B", "Track C"), start=1)
    ]
    started: list[int] = []
    with pytest.raises(OperationCancelled):
        resolver.resolve_batch(items, cancel=token, on_item_started=started.append)

    assert started == [0, 1]
    assert router.count() == 2
    assert not any("/release/" in req.url.path for req in router.log)
    assert resolver.providers_unavailable is False


def test_resolve_batch_cancel_mid_group_finalize_stops_later_enrichments() -> None:
    token = CancelToken()
    router = Router()
    for idx, title in enumerate(("Track A", "Track B", "Track C")):
        router.json(
            query_has(title),
            _auto_accept_payload(title, rec_id=f"rec-{idx}", rel_id=f"rel-{idx}"),
        )

    def _rel_0(_req: httpx.Request) -> httpx.Response:
        token.cancel()
        return httpx.Response(200, json=_release_payload("rel-0", "Track A", "rec-0"))

    router.add(path_is("/ws/2/release/rel-0"), _rel_0)
    router.json(
        path_is("/ws/2/release/rel-1"),
        _release_payload("rel-1", "Track B", "rec-1"),
    )
    router.json(
        path_is("/ws/2/release/rel-2"),
        _release_payload("rel-2", "Track C", "rec-2"),
    )

    resolver = TrackResolver(make_mb(router), None)
    items = [
        BatchItem(
            ParsedHints(artist="Artist", title=title),
            duration_seconds=200.0,
            group_key="album-group",
        )
        for title in ("Track A", "Track B", "Track C")
    ]
    started: list[int] = []
    with pytest.raises(OperationCancelled):
        resolver.resolve_batch(items, cancel=token, on_item_started=started.append)

    # All 3 members were gathered (started once each), then only rel-0 was enriched.
    assert started == [0, 1, 2]
    release_paths = [req.url.path for req in router.log if "/release/" in req.url.path]
    assert release_paths == ["/ws/2/release/rel-0"]
    assert resolver.providers_unavailable is False


def test_resolve_batch_cancel_stops_fallback_queries_within_single_item() -> None:
    token = CancelToken()
    router = Router()

    def _low_score_hit(_req: httpx.Request) -> httpx.Response:
        token.cancel()
        return httpx.Response(
            200,
            json=search_payload(
                recording(
                    "rec-other",
                    "Unrelated Song",
                    credit(("Other Artist", "")),
                    [release("rel-other", "Other Album", "rg-other")],
                )
            ),
        )

    router.add(lambda req: req.url.path.endswith("/recording/"), _low_score_hit)
    resolver = TrackResolver(make_mb(router), make_itunes(router))
    hints = ParsedHints(
        artist="Artist",
        title="Target Song",
        swapped_candidate=("Target Song", "Artist"),
    )
    with pytest.raises(OperationCancelled):
        resolver.resolve_batch([BatchItem(hints)], cancel=token)

    # Stopped immediately after the primary recording search: no broad, swapped, or iTunes calls.
    assert router.count() == 1
    assert router.count("itunes.apple.com") == 0
    assert resolver.providers_unavailable is False


def test_resolve_batch_cancel_with_transient_error_does_not_trip_breaker() -> None:
    token = CancelToken()

    def _handler(_req: httpx.Request) -> httpx.Response:
        token.cancel()
        return httpx.Response(503)

    f = HttpFetcher(
        transport=httpx.MockTransport(_handler),
        limiter=RateLimiter(0.0),
        max_retries=0,
        sleep=lambda _s: None,
    )
    mb = MusicBrainzClient(TEST_UA, fetcher=f, art_fetcher=f)
    resolver = TrackResolver(mb, None, circuit_breaker_threshold=1)

    with pytest.raises(OperationCancelled):
        resolver.resolve_batch(
            [BatchItem(ParsedHints(artist="Artist", title="Song"))],
            cancel=token,
        )

    assert resolver.providers_unavailable is False


def test_resolve_batch_restores_cancel_context_after_cancel() -> None:
    router = _mb_router_for_tracks(["Alpha", "Beta"])
    resolver = TrackResolver(make_mb(router), None)
    cancelled = CancelToken()
    cancelled.cancel()

    with pytest.raises(OperationCancelled):
        resolver.resolve_batch(
            [BatchItem(ParsedHints(artist="Artist", title="Alpha"), duration_seconds=200.0)],
            cancel=cancelled,
        )

    # A subsequent uncancelled lookup on the same resolver succeeds normally.
    results = resolver.resolve_batch(
        [BatchItem(ParsedHints(artist="Artist", title="Beta"), duration_seconds=200.0)]
    )
    assert len(results) == 1
    assert results[0].decision == MatchDecision.AUTO_ACCEPT


# -- C4: RateLimiter.set_sleep, HttpFetcher.set_sleep, and sleep cancellation ------


def test_set_sleep_replaces_limiter_and_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_time_sleep(monkeypatch)

    now = 100.0
    limiter_sleeps: list[float] = []
    limiter = RateLimiter(1.5, clock=lambda: now)

    def _custom_limiter_sleep(secs: float) -> None:
        nonlocal now
        limiter_sleeps.append(secs)
        now += secs

    limiter.set_sleep(_custom_limiter_sleep)
    assert limiter.acquire() == 0.0
    assert limiter.acquire() == pytest.approx(1.5)
    assert limiter_sleeps == [pytest.approx(1.5)]

    # Verify HttpFetcher.set_sleep updates both the retry backoff sleep AND its limiter sleep.
    base_time = 200.0
    recorded: list[float] = []
    attempts = 0
    wall_now = datetime(2030, 1, 1, tzinfo=UTC).timestamp()

    def _handler(_req: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, headers={"Retry-After": "2"})
        if attempts == 2:
            when = datetime(2030, 1, 1, 0, 0, 4, tzinfo=UTC)
            return httpx.Response(429, headers={"Retry-After": format_datetime(when, usegmt=True)})
        return httpx.Response(200, json={"ok": True})

    def _combined_sleep(secs: float) -> None:
        nonlocal base_time
        recorded.append(round(secs, 4))
        base_time += secs

    http = HttpFetcher(
        transport=httpx.MockTransport(_handler),
        limiter=RateLimiter(1.0, clock=lambda: base_time),
        wall_clock=lambda: wall_now,
        max_retries=3,
    )
    http.set_sleep(_combined_sleep)
    assert http.get_json("https://musicbrainz.org/ws/2/test") == {"ok": True}
    # First 503 slept 2s (>= 1s limiter), second 429 slept 4s (>= 1s limiter).
    assert recorded == [2.0, 4.0]
    # Next immediate request must hit the replaced limiter sleep (1.0s).
    assert http.get_json("https://musicbrainz.org/ws/2/test") == {"ok": True}
    assert recorded == [2.0, 4.0, 1.0]


def test_cancel_during_rate_limit_sleep_in_resolve_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_time_sleep(monkeypatch)

    router = _mb_router_for_tracks(["Alpha", "Beta"])
    token = CancelToken()
    mb_fetcher = HttpFetcher(
        transport=router.transport,
        limiter=RateLimiter(30.0),
    )
    mb_fetcher.set_sleep(token.sleep)
    mb = MusicBrainzClient(TEST_UA, fetcher=mb_fetcher, art_fetcher=fetcher(router))
    resolver = TrackResolver(mb, None, circuit_breaker_threshold=1)

    items = [
        BatchItem(ParsedHints(artist="Artist", title="Alpha"), duration_seconds=200.0),
        BatchItem(ParsedHints(artist="Artist", title="Beta"), duration_seconds=200.0),
    ]
    # First recording search for Alpha finishes immediately; enrichment (/release/rel-0) enters
    # the 30-second rate-limit wait on token.sleep. Cancel promptly from another thread.
    timer = threading.Timer(0.05, token.cancel)
    start = time.monotonic()
    timer.start()
    try:
        with pytest.raises(OperationCancelled):
            resolver.resolve_batch(items, cancel=token)
    finally:
        timer.cancel()
        timer.join(timeout=1.0)
    elapsed = time.monotonic() - start

    assert elapsed < 5.0
    assert mb_fetcher.request_count == 1
    assert router.count() == 1
    assert resolver.providers_unavailable is False

    # The rate limiter's lock must have been released cleanly on cancel.
    assert mb_fetcher.limiter._lock.acquire(blocking=False)
    mb_fetcher.limiter._lock.release()


def test_cancel_during_rate_limit_sleep_preserves_prior_failure_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_time_sleep(monkeypatch)

    router = Router()
    calls = {"n": 0}

    def _handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=search_payload())

    token = CancelToken()
    sleep_entered = threading.Event()

    def _tracked_sleep(seconds: float) -> None:
        sleep_entered.set()
        token.sleep(seconds)

    mb_fetcher = HttpFetcher(
        transport=httpx.MockTransport(_handler),
        limiter=RateLimiter(30.0),
        max_retries=0,
    )
    mb_fetcher.set_sleep(_tracked_sleep)
    mb = MusicBrainzClient(TEST_UA, fetcher=mb_fetcher, art_fetcher=fetcher(router))
    # Threshold of 2: request 1 hits 503 (count=1, not tripped). Request 2 sleeps on rate limit
    # and is cancelled; cancelling must NOT increment consecutive failures or trip the breaker.
    resolver = TrackResolver(mb, None, circuit_breaker_threshold=2)

    def _canceller() -> None:
        if sleep_entered.wait(timeout=5.0):
            token.cancel()

    worker = threading.Thread(target=_canceller, daemon=True)
    worker.start()
    try:
        start = time.monotonic()
        with pytest.raises(OperationCancelled):
            resolver.resolve_batch(
                [
                    BatchItem(ParsedHints(artist="Artist", title="Track 1")),
                    BatchItem(ParsedHints(artist="Artist", title="Track 2")),
                ],
                cancel=token,
            )
        elapsed = time.monotonic() - start
    finally:
        worker.join(timeout=2.0)

    assert sleep_entered.is_set()
    assert elapsed < 5.0
    assert mb_fetcher.request_count == 1
    assert resolver.providers_unavailable is False


def test_cancel_during_itunes_rate_limit_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_time_sleep(monkeypatch)

    mb_router = Router()
    mb_router.json(lambda _r: True, search_payload())

    itunes_router = Router()
    itunes_router.json(lambda _r: True, {"resultCount": 0, "results": []})

    token = CancelToken()
    itunes_sleep_entered = threading.Event()

    def _tracked_sleep(seconds: float) -> None:
        itunes_sleep_entered.set()
        token.sleep(seconds)

    itunes_fetcher = HttpFetcher(
        transport=itunes_router.transport,
        limiter=RateLimiter(30.0),
    )
    itunes_fetcher.set_sleep(_tracked_sleep)
    itunes = ITunesClient(fetcher=itunes_fetcher, art_fetcher=fetcher(itunes_router))
    resolver = TrackResolver(make_mb(mb_router), itunes, circuit_breaker_threshold=1)

    def _canceller() -> None:
        if itunes_sleep_entered.wait(timeout=5.0):
            token.cancel()

    worker = threading.Thread(target=_canceller, daemon=True)
    worker.start()
    try:
        start = time.monotonic()
        with pytest.raises(OperationCancelled):
            resolver.resolve_batch(
                [
                    BatchItem(ParsedHints(artist="Artist", title="Miss A")),
                    BatchItem(ParsedHints(artist="Artist", title="Miss B")),
                ],
                cancel=token,
            )
        elapsed = time.monotonic() - start
    finally:
        worker.join(timeout=2.0)

    assert itunes_sleep_entered.is_set()
    assert elapsed < 5.0
    assert itunes_fetcher.request_count == 1
    assert resolver.providers_unavailable is False


@pytest.mark.parametrize(
    ("status", "headers"),
    [
        (503, {}),
        (503, {"Retry-After": "30"}),
        (429, {"Retry-After": "30"}),
    ],
)
def test_cancel_during_retry_backoff_returns_promptly(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    headers: dict[str, str],
) -> None:
    _block_time_sleep(monkeypatch)

    router = Router().add(lambda _r: True, lambda _r: httpx.Response(status, headers=headers))
    token = CancelToken()
    sleep_entered = threading.Event()
    observed_delays: list[float] = []

    def _tracked_sleep(seconds: float) -> None:
        observed_delays.append(seconds)
        sleep_entered.set()
        token.sleep(seconds)

    mb_fetcher = HttpFetcher(
        transport=router.transport,
        limiter=RateLimiter(0.0),
        backoff_base=30.0,
        max_retries=4,
        jitter=lambda: 0.0,
    )
    mb_fetcher.set_sleep(_tracked_sleep)
    mb = MusicBrainzClient(TEST_UA, fetcher=mb_fetcher, art_fetcher=fetcher(router))
    resolver = TrackResolver(mb, None, circuit_breaker_threshold=1)

    def _canceller() -> None:
        if sleep_entered.wait(timeout=5.0):
            token.cancel()

    worker = threading.Thread(target=_canceller, daemon=True)
    worker.start()
    try:
        start = time.monotonic()
        with pytest.raises(OperationCancelled):
            resolver.resolve_batch(
                [BatchItem(ParsedHints(artist="Artist", title="Song"))],
                cancel=token,
            )
        elapsed = time.monotonic() - start
    finally:
        worker.join(timeout=2.0)

    assert sleep_entered.is_set()
    assert observed_delays == [pytest.approx(30.0)]
    assert elapsed < 5.0
    assert mb_fetcher.request_count == 1
    assert router.count() == 1
    assert resolver.providers_unavailable is False


def test_cancel_during_503_backoff_with_direct_token_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_time_sleep(monkeypatch)

    router = Router().add(lambda _r: True, lambda _r: httpx.Response(503))
    token = CancelToken()
    mb_fetcher = HttpFetcher(
        transport=router.transport,
        limiter=RateLimiter(0.0),
        backoff_base=30.0,
        max_retries=3,
    )
    mb_fetcher.set_sleep(token.sleep)
    mb = MusicBrainzClient(TEST_UA, fetcher=mb_fetcher, art_fetcher=fetcher(router))
    resolver = TrackResolver(mb, None, circuit_breaker_threshold=1)

    timer = threading.Timer(0.05, token.cancel)
    start = time.monotonic()
    timer.start()
    try:
        with pytest.raises(OperationCancelled):
            resolver.resolve_batch(
                [BatchItem(ParsedHints(artist="Artist", title="Song"))],
                cancel=token,
            )
    finally:
        timer.cancel()
        timer.join(timeout=1.0)

    assert time.monotonic() - start < 5.0
    assert mb_fetcher.request_count == 1
    assert resolver.providers_unavailable is False


def test_cancel_during_artwork_stream_503_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_time_sleep(monkeypatch)

    router = Router().add(
        lambda _r: True, lambda _r: httpx.Response(503, headers={"Retry-After": "30"})
    )
    token = CancelToken()
    art_fetcher = HttpFetcher(
        transport=router.transport,
        limiter=RateLimiter(0.0),
        max_retries=1,
        backoff_base=30.0,
    )
    art_fetcher.set_sleep(token.sleep)
    mb = MusicBrainzClient(TEST_UA, fetcher=fetcher(router), art_fetcher=art_fetcher)

    timer = threading.Timer(0.05, token.cancel)
    start = time.monotonic()
    timer.start()
    try:
        with pytest.raises(OperationCancelled):
            mb.fetch_cover_art("rel-1", "rg-1")
    finally:
        timer.cancel()
        timer.join(timeout=1.0)
    assert time.monotonic() - start < 5.0
    assert art_fetcher.request_count == 1

    # iTunes artwork cascade behaves identically.
    it_router = Router().add(
        lambda _r: True, lambda _r: httpx.Response(503, headers={"Retry-After": "30"})
    )
    it_token = CancelToken()
    it_art = HttpFetcher(
        transport=it_router.transport,
        limiter=RateLimiter(0.0),
        max_retries=1,
        backoff_base=30.0,
    )
    it_art.set_sleep(it_token.sleep)
    itunes = ITunesClient(fetcher=fetcher(it_router), art_fetcher=it_art)

    timer2 = threading.Timer(0.05, it_token.cancel)
    start2 = time.monotonic()
    timer2.start()
    try:
        with pytest.raises(OperationCancelled):
            itunes.fetch_artwork("https://is1-ssl.mzstatic.com/image/thumb/x/100x100bb.jpg")
    finally:
        timer2.cancel()
        timer2.join(timeout=1.0)
    assert time.monotonic() - start2 < 5.0
    assert it_art.request_count == 1


def test_cancel_during_transport_error_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_time_sleep(monkeypatch)

    def _timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated slow read", request=request)

    token = CancelToken()
    f = HttpFetcher(
        transport=httpx.MockTransport(_timeout),
        limiter=RateLimiter(0.0),
        max_retries=2,
        transport_max_retries=1,
        backoff_base=30.0,
    )
    f.set_sleep(token.sleep)
    mb = MusicBrainzClient(TEST_UA, fetcher=f, art_fetcher=f)
    resolver = TrackResolver(mb, None, circuit_breaker_threshold=1)

    timer = threading.Timer(0.05, token.cancel)
    start = time.monotonic()
    timer.start()
    try:
        with pytest.raises(OperationCancelled):
            resolver.resolve_batch(
                [BatchItem(ParsedHints(artist="Artist", title="Song"))],
                cancel=token,
            )
    finally:
        timer.cancel()
        timer.join(timeout=1.0)

    assert time.monotonic() - start < 5.0
    assert f.request_count == 1
    assert resolver.providers_unavailable is False


def test_cancelled_request_does_not_poison_cache(tmp_path: Path) -> None:
    token = CancelToken()
    token.cancel()
    cache = TTLCache(tmp_path / "cache.sqlite")
    try:
        mb_router = Router().add(lambda _r: True, lambda _r: httpx.Response(503))
        mb_f = HttpFetcher(transport=mb_router.transport, limiter=RateLimiter(0.0), max_retries=1)
        mb_f.set_sleep(token.sleep)
        mb = MusicBrainzClient(TEST_UA, fetcher=mb_f, art_fetcher=mb_f, cache=cache)
        with pytest.raises(OperationCancelled):
            mb.search_recordings("Song", "Artist")

        it_router = Router().add(lambda _r: True, lambda _r: httpx.Response(503))
        it_f = HttpFetcher(transport=it_router.transport, limiter=RateLimiter(0.0), max_retries=1)
        it_f.set_sleep(token.sleep)
        itunes = ITunesClient(fetcher=it_f, art_fetcher=it_f, cache=cache)
        with pytest.raises(OperationCancelled):
            itunes.search("Song", "Artist")

        assert sum(stat.rows for stat in cache.stats()) == 0
    finally:
        cache.close()


# -- Propagation out of broad `except Exception` handlers -------------------------


def test_operation_cancelled_propagates_out_of_safe() -> None:
    router = Router()
    resolver = TrackResolver(make_mb(router), None, circuit_breaker_threshold=1)
    warnings: list[str] = []

    def _raise_cancel() -> None:
        raise OperationCancelled("cancelled in _safe")

    with pytest.raises(OperationCancelled):
        resolver._safe(warnings, "MusicBrainz lookup", _raise_cancel, provider="musicbrainz")

    assert warnings == []
    assert resolver.providers_unavailable is False


def test_operation_cancelled_propagates_out_of_manual_search() -> None:
    token = CancelToken()
    token.cancel()
    router = Router().add(lambda _r: True, lambda _r: httpx.Response(503))
    mb_fetcher = HttpFetcher(
        transport=router.transport,
        limiter=RateLimiter(0.0),
        max_retries=2,
        backoff_base=30.0,
    )
    mb_fetcher.set_sleep(token.sleep)
    mb = MusicBrainzClient(TEST_UA, fetcher=mb_fetcher, art_fetcher=mb_fetcher)
    resolver = TrackResolver(mb, None, circuit_breaker_threshold=1)

    with pytest.raises(OperationCancelled):
        resolver.manual_search(title="Song", artist="Artist")

    assert mb_fetcher.request_count == 1
    assert resolver.providers_unavailable is False


def test_operation_cancelled_propagates_out_of_enrich_and_enrich_candidate() -> None:
    token = CancelToken()
    token.cancel()
    router = Router().add(lambda _r: True, lambda _r: httpx.Response(503))
    mb_fetcher = HttpFetcher(
        transport=router.transport,
        limiter=RateLimiter(0.0),
        max_retries=2,
        backoff_base=30.0,
    )
    mb_fetcher.set_sleep(token.sleep)
    mb = MusicBrainzClient(TEST_UA, fetcher=mb_fetcher, art_fetcher=mb_fetcher)
    resolver = TrackResolver(mb, None, circuit_breaker_threshold=1)

    ref = ReleaseRef(release_id="rel-1", title="Discovery")
    cand = Candidate(
        source="musicbrainz",
        title="One More Time",
        artist="Daft Punk",
        recording_id="rec-1",
        release_group_id="rg-1",
        releases=[ref],
        chosen_release=ref,
    )
    warnings: list[str] = []

    with pytest.raises(OperationCancelled):
        resolver.enrich_candidate(cand, warnings=warnings)
    assert warnings == []
    assert resolver.providers_unavailable is False

    with pytest.raises(OperationCancelled):
        resolver.enrich(cand, warnings=warnings)
    assert warnings == []
    assert resolver.providers_unavailable is False


def test_operation_cancelled_propagates_from_itunes_genre_fallback_in_enrich() -> None:
    mb_router = Router()
    mb_router.json(
        path_is("/ws/2/release/rel-1"),
        release_lookup(
            "rel-1",
            "Discovery",
            artist_credits=credit(("Daft Punk", "")),
            rg_id="rg-1",
            rg_first_date="2001-03-12",
            date="2001-03-12",
            media=[medium(1, [("rec-1", "One More Time")])],
        ),
    )
    mb_router.json(path_is("/ws/2/release-group/rg-1"), {"id": "rg-1", "genres": []})

    token = CancelToken()
    token.cancel()
    itunes_router = Router().add(lambda _r: True, lambda _r: httpx.Response(503))
    it_fetcher = HttpFetcher(
        transport=itunes_router.transport,
        limiter=RateLimiter(0.0),
        max_retries=2,
        backoff_base=30.0,
    )
    it_fetcher.set_sleep(token.sleep)
    itunes = ITunesClient(fetcher=it_fetcher, art_fetcher=it_fetcher)
    resolver = TrackResolver(make_mb(mb_router), itunes, circuit_breaker_threshold=1)

    ref = ReleaseRef(release_id="rel-1", title="Discovery")
    cand = Candidate(
        source="musicbrainz",
        title="One More Time",
        artist="Daft Punk",
        recording_id="rec-1",
        release_group_id="rg-1",
        releases=[ref],
        chosen_release=ref,
    )
    warnings: list[str] = []
    with pytest.raises(OperationCancelled):
        resolver.enrich_candidate(cand, warnings=warnings)
    assert warnings == []
    assert it_fetcher.request_count == 1
    assert resolver.providers_unavailable is False


def test_operation_cancelled_propagates_out_of_resolve() -> None:
    token = CancelToken()
    token.cancel()
    router = Router().add(lambda _r: True, lambda _r: httpx.Response(503))
    mb_fetcher = HttpFetcher(
        transport=router.transport,
        limiter=RateLimiter(0.0),
        max_retries=2,
        backoff_base=30.0,
    )
    mb_fetcher.set_sleep(token.sleep)
    mb = MusicBrainzClient(TEST_UA, fetcher=mb_fetcher, art_fetcher=mb_fetcher)
    resolver = TrackResolver(mb, None, circuit_breaker_threshold=1)

    with pytest.raises(OperationCancelled):
        resolver.resolve(ParsedHints(artist="Artist", title="Song"))

    assert mb_fetcher.request_count == 1
    assert resolver.providers_unavailable is False


def test_ast_audit_lookup_and_providers_never_catch_base_exception_or_sleep() -> None:
    from audio_from_video import lookup

    for mod in (lookup, providers_base, providers_itunes, providers_musicbrainz):
        src = inspect.getsource(mod)
        tree = ast.parse(src, filename=mod.__file__ or "<module>")
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                assert node.type is not None, f"Bare except in {mod.__name__}:{node.lineno}"
                caught = ast.unparse(node.type)
                assert "BaseException" not in caught, (
                    f"BaseException handler in {mod.__name__}:{node.lineno}"
                )
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "time"
                and any(alias.name == "sleep" for alias in node.names)
            ):
                raise AssertionError(
                    f"Direct 'from time import sleep' in {mod.__name__}:{node.lineno}"
                )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "time"
                and node.func.attr == "sleep"
            ):
                raise AssertionError(f"Direct time.sleep() call in {mod.__name__}:{node.lineno}")
