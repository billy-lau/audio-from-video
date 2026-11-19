"""Tests for provider plumbing, MusicBrainz, Cover Art Archive, iTunes, and artwork."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest

from audio_from_video.artwork import ArtworkFetcher
from audio_from_video.cache import TTLCache
from audio_from_video.config import ConfigError, MatchingConfig
from audio_from_video.models import MatchDecision
from audio_from_video.providers.base import (
    MAX_DOWNLOAD_BYTES,
    HttpFetcher,
    ProviderError,
    RateLimiter,
    canonical_url,
    parse_retry_after,
)
from audio_from_video.providers.itunes import (
    ITUNES_MIN_INTERVAL,
    ITunesClient,
    artwork_url_variants,
    clean_album_name,
    results_to_candidates,
    split_itunes_artist,
)
from audio_from_video.providers.musicbrainz import (
    MB_MIN_INTERVAL,
    MusicBrainzClient,
    build_recording_query,
    flip_sort_name,
    pick_english_alias,
    recordings_to_candidates,
    split_artist_credit,
)
from audio_from_video.ranking import (
    Query,
    choose_release_group,
    evaluate_candidates,
    identity_key,
    rank_candidates,
)
from provider_fixtures import (
    TEST_UA,
    Router,
    credit,
    fetcher,
    host_is,
    jpeg_bytes,
    make_itunes,
    make_mb,
    path_is,
    query_has,
    recording,
    release,
    search_payload,
)


class FakeTime:
    """Fake monotonic clock whose sleep advances time and records durations."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 6))
        self.now += seconds


# -- Base plumbing ---------------------------------------------------------------


def test_canonical_url_sorts_and_merges_params() -> None:
    a = canonical_url("HTTPS://MusicBrainz.org/ws/2/recording/?fmt=json&query=x", {"limit": 25})
    b = canonical_url(
        "https://musicbrainz.org/ws/2/recording/", {"query": "x", "limit": 25, "fmt": "json"}
    )
    assert a == b == "https://musicbrainz.org/ws/2/recording/?fmt=json&limit=25&query=x"
    assert canonical_url("https://h/p#frag") == "https://h/p"


def test_rate_limiter_enforces_spacing_with_fake_clock() -> None:
    t = FakeTime()
    limiter = RateLimiter(1.1, clock=t.clock, sleep=t.sleep)
    limiter.acquire()
    limiter.acquire()
    t.now += 0.5
    limiter.acquire()
    t.now += 5.0
    limiter.acquire()
    assert t.sleeps == [1.1, 0.6]


def test_rate_limiter_is_thread_safe() -> None:
    t = FakeTime()
    lock = threading.Lock()

    def sleep(seconds: float) -> None:
        with lock:
            t.sleep(seconds)

    limiter = RateLimiter(1.0, clock=t.clock, sleep=sleep)
    threads = [threading.Thread(target=limiter.acquire) for _ in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert t.now == pytest.approx(105.0)


def test_default_provider_spacing() -> None:
    assert MB_MIN_INTERVAL >= 1.1
    assert ITUNES_MIN_INTERVAL >= 3.0
    mb = MusicBrainzClient(TEST_UA)
    assert mb.fetcher.limiter.min_interval == pytest.approx(1.1)
    itunes = ITunesClient()
    assert itunes.fetcher.limiter.min_interval == pytest.approx(3.0)


def test_parse_retry_after_seconds_and_http_date() -> None:
    assert parse_retry_after("3", now=0.0) == 3.0
    when = datetime(2030, 1, 1, 0, 0, 10, tzinfo=UTC)
    now = datetime(2030, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp()
    assert parse_retry_after(format_datetime(when, usegmt=True), now=now) == pytest.approx(10.0)
    assert parse_retry_after("garbage", now=0.0) is None
    assert parse_retry_after(None, now=0.0) is None


def test_fetcher_retries_with_retry_after_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, headers={"Retry-After": "2"})
        if calls["n"] == 2:
            when = datetime(2030, 1, 1, 0, 0, 7, tzinfo=UTC)
            return httpx.Response(429, headers={"Retry-After": format_datetime(when, usegmt=True)})
        return httpx.Response(200, json={"ok": True})

    sleeps: list[float] = []
    f = HttpFetcher(
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        wall_clock=lambda: datetime(2030, 1, 1, tzinfo=UTC).timestamp(),
    )
    assert f.get_json("https://musicbrainz.org/ws/2/x") == {"ok": True}
    assert sleeps == [2.0, pytest.approx(7.0)]
    assert f.request_count == 3


@pytest.mark.parametrize("status", [500, 502, 504])
def test_fetcher_exponential_backoff_with_jitter_then_error(status: int) -> None:
    sleeps: list[float] = []
    f = HttpFetcher(
        transport=httpx.MockTransport(lambda r: httpx.Response(status)),
        sleep=sleeps.append,
        jitter=lambda: 0.5,
        backoff_base=1.0,
        max_retries=3,
    )
    with pytest.raises(ProviderError):
        f.get_json("https://musicbrainz.org/ws/2/x")
    assert sleeps == [1.25, 2.5, 5.0]
    assert f.request_count == 4


def test_fetcher_404_returns_none_and_400_raises() -> None:
    router = Router().json(path_is("/bad"), {"error": "x"}, status=400)
    f = fetcher(router)
    assert f.get_json("https://musicbrainz.org/missing") is None
    assert f.get_bytes("https://coverartarchive.org/missing") is None
    with pytest.raises(ProviderError):
        f.get_json("https://musicbrainz.org/bad")


def test_fetcher_transport_errors_retry_then_raise() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    f = HttpFetcher(transport=httpx.MockTransport(handler), sleep=lambda s: None, max_retries=1)
    with pytest.raises(ProviderError):
        f.get_json("https://musicbrainz.org/x")
    assert f.request_count == 2


def test_byte_download_cap() -> None:
    big = b"x" * (MAX_DOWNLOAD_BYTES + 1)
    router = Router().add(path_is("/big"), lambda r: httpx.Response(200, content=big))
    router.add(path_is("/small"), lambda r: httpx.Response(200, content=b"abc"))
    f = fetcher(router)
    with pytest.raises(ProviderError):
        f.get_bytes("https://coverartarchive.org/big")
    assert f.get_bytes("https://coverartarchive.org/small", max_bytes=3) == b"abc"
    with pytest.raises(ProviderError):
        f.get_bytes("https://coverartarchive.org/small", max_bytes=2)


# -- MusicBrainz -----------------------------------------------------------------------


def _one_more_time() -> dict:
    return search_payload(
        recording(
            "rec-omt",
            "One More Time",
            credit(("Daft Punk", "", "artist-dp", "Daft Punk")),
            [
                release("rel-disc", "Discovery", "rg-disc", track_number=1),
                release(
                    "rel-single",
                    "One More Time",
                    "rg-single",
                    primary="Single",
                    date="2000-11-13",
                    medium_track_count=3,
                ),
            ],
            length_ms=320_000,
        )
    )


def test_mb_requires_valid_user_agent() -> None:
    with pytest.raises(ConfigError):
        MusicBrainzClient("Mozilla/5.0 (Macintosh)")
    with pytest.raises(ConfigError):
        MusicBrainzClient("audio-from-video/0.1.0 ( http://localhost/contact )")


def test_mb_search_sends_user_agent_and_quoted_query() -> None:
    router = Router().json(query_has('recording:"One More Time"'), _one_more_time())
    mb = make_mb(router)
    payload = mb.search_recordings("One More Time", "Daft Punk")
    assert payload["recordings"][0]["id"] == "rec-omt"
    req = router.log[0]
    assert req.headers["User-Agent"] == TEST_UA
    assert req.url.params["query"] == 'recording:"One More Time" AND artist:"Daft Punk"'
    assert req.url.params["fmt"] == "json"


def test_lucene_escaping_and_unquoted_fallback() -> None:
    assert build_recording_query('Say "Hi" \\o/', None) == 'recording:"Say \\"Hi\\" \\\\o/"'
    assert build_recording_query("AC/DC: Live!", "AC/DC", quoted=False) == (
        "recording:(AC\\/DC\\: Live\\!) AND artist:(AC\\/DC)"
    )
    router = Router()
    router.json(query_has('recording:"'), search_payload())
    router.json(query_has("recording:("), _one_more_time())
    mb = make_mb(router)
    cands = mb.search_candidates("One More Time", "Daft Punk")
    assert len(router.log) == 2
    assert router.log[1].url.params["query"].startswith("recording:(One More Time)")
    assert {c.release_group_id for c in cands} == {"rg-disc", "rg-single"}


def test_mb_cache_hit_avoids_requests(tmp_path: Path) -> None:
    cache = TTLCache(tmp_path / "c.sqlite3")
    router = Router().json(query_has("One More Time"), _one_more_time())
    make_mb(router, cache=cache).search_recordings("One More Time", "Daft Punk")
    assert router.count() == 1
    second = make_mb(router, cache=cache)
    second.search_recordings("One More Time", "Daft Punk")
    assert router.count() == 1
    assert second.request_count == 0
    cache.close()


def test_mb_negative_404_is_cached(tmp_path: Path) -> None:
    cache = TTLCache(tmp_path / "c.sqlite3")
    router = Router()
    mb = make_mb(router, cache=cache)
    assert mb.get_release("nope") is None
    assert make_mb(router, cache=cache).get_release("nope") is None
    assert router.count() == 1
    cache.close()


def test_get_release_and_artist_inc_params() -> None:
    router = Router()
    router.json(path_is("/ws/2/release/rel-1"), {"id": "rel-1"})
    router.json(path_is("/ws/2/artist/art-1"), {"id": "art-1", "aliases": []})
    mb = make_mb(router)
    assert mb.get_release("rel-1") == {"id": "rel-1"}
    assert mb.get_artist("art-1")["id"] == "art-1"
    assert router.log[0].url.params["inc"] == (
        "artist-credits+recordings+release-groups+genres+tags+media+release-rels"
    )
    assert router.log[1].url.params["inc"] == "aliases"


def test_cover_art_archive_cascade_follows_redirects() -> None:
    img = jpeg_bytes(500, 500)
    router = Router()
    router.add(
        path_is("/release-group/rg-1/front-1200"),
        lambda r: httpx.Response(307, headers={"Location": "https://archive.org/img/front.jpg"}),
    )
    router.add(host_is("archive.org"), lambda r: httpx.Response(200, content=img))
    mb = make_mb(router)
    data, url = mb.fetch_cover_art("rel-1", "rg-1")
    assert data == img
    assert url.endswith("/release-group/rg-1/front-1200")
    paths = [r.url.path for r in router.log]
    assert paths == [
        "/release/rel-1/front-1200",
        "/release/rel-1/front-500",
        "/release-group/rg-1/front-1200",
        "/img/front.jpg",
    ]
    assert all(r.headers["User-Agent"] == TEST_UA for r in router.log)


def test_cover_art_cascade_exhausted_returns_none() -> None:
    router = Router()
    mb = make_mb(router)
    assert mb.fetch_cover_art("rel-1", "rg-1") == (None, None)
    assert [r.url.path for r in router.log][-1] == "/release/rel-1/front"
    assert router.count() == 5


def test_split_artist_credit_featuring_and_coheadliners() -> None:
    get_lucky = credit(
        ("Daft Punk", " feat. "),
        ("Pharrell Williams", " & "),
        ("Nile Rodgers", ""),
    )
    primary, feats, _aliases, lead, lead_id, feat_ids, full = split_artist_credit(get_lucky)
    assert primary == "Daft Punk"
    assert feats == ["Pharrell Williams", "Nile Rodgers"]
    assert lead == "Daft Punk"
    assert lead_id == "artist-daft-punk"
    assert len(feat_ids) == 2
    assert full == "Daft Punk feat. Pharrell Williams & Nile Rodgers"

    duo = credit(("Beyoncé", " & "), ("JAY-Z", ""))
    primary, feats, *_ = split_artist_credit(duo)
    assert primary == "Beyoncé & JAY-Z"
    assert feats == []

    ft = credit(("Mark Ronson", " ft. "), ("Bruno Mars", ""))
    assert split_artist_credit(ft)[:2] == ("Mark Ronson", ["Bruno Mars"])
    featuring = credit(("Gotye", " featuring "), ("Kimbra", ""))
    assert split_artist_credit(featuring)[:2] == ("Gotye", ["Kimbra"])


def test_sort_name_flip_and_aliases_kept_for_matching() -> None:
    assert flip_sort_name("Cheung, Jacky") == "Jacky Cheung"
    assert flip_sort_name("Beatles, The") == "The Beatles"
    assert flip_sort_name("Daft Punk") == "Daft Punk"
    credits = [
        {
            "name": "張學友",
            "joinphrase": "",
            "artist": {
                "id": "a-jc",
                "name": "張學友",
                "sort-name": "Cheung, Jacky",
                "aliases": [{"name": "Jacky Cheung Hok-yau"}],
            },
        }
    ]
    _p, _f, aliases, *_ = split_artist_credit(credits)
    assert "Jacky Cheung" in aliases
    assert "Jacky Cheung Hok-yau" in aliases


def test_recordings_to_candidates_groups_by_release_group() -> None:
    cands = recordings_to_candidates(_one_more_time())
    assert len(cands) == 2
    album = next(c for c in cands if c.release_group_id == "rg-disc")
    assert album.album == "Discovery"
    assert album.primary_type == "Album"
    assert album.duration_seconds == 320.0
    assert album.releases[0].track_number == 1
    assert album.releases[0].formats == ("CD",)
    assert album.source == "musicbrainz"
    assert recordings_to_candidates(None) == []


def test_recordings_to_candidates_reads_live_release_count_as_disc_count() -> None:
    payload = search_payload(
        recording(
            "rec-1",
            "Mr. Brightside",
            credit(("The Killers", "")),
            [
                {
                    "id": "rel-std",
                    "title": "Hot Fuss",
                    "status": "Official",
                    "date": "2004-06-07",
                    "country": "GB",
                    "count": 1,
                    "track-count": 11,
                    "release-group": {
                        "id": "rg-hf",
                        "title": "Hot Fuss",
                        "primary-type": "Album",
                    },
                    "media": [
                        {
                            "position": 1,
                            "format": "CD",
                            "track-count": 11,
                            "track": [{"number": "2"}],
                        }
                    ],
                },
                {
                    "id": "rel-box",
                    "title": "Hot Fuss",
                    "status": "Official",
                    "date": "2004-06-07",
                    "country": "GB",
                    "count": 11,
                    "track-count": 22,
                    "release-group": {
                        "id": "rg-hf",
                        "title": "Hot Fuss",
                        "primary-type": "Album",
                    },
                    "media": [
                        {
                            "position": 2,
                            "format": '7" Vinyl',
                            "track-count": 2,
                            "track": [{"number": "1"}],
                        }
                    ],
                },
            ],
        )
    )
    cands = recordings_to_candidates(payload)
    assert len(cands) == 1
    by_id = {ref.release_id: ref for ref in cands[0].releases}
    assert by_id["rel-std"].disc_count == 1
    assert by_id["rel-box"].disc_count == 11


def test_pick_english_alias_prefers_en_locale() -> None:
    payload = {
        "name": "張學友",
        "sort-name": "Cheung, Jacky",
        "aliases": [
            {"name": "Zhang Xueyou", "locale": None, "type": "Artist name"},
            {"name": "Jacky Cheung", "locale": "en", "primary": True, "type": "Artist name"},
        ],
    }
    assert pick_english_alias(payload) == "Jacky Cheung"
    assert pick_english_alias({"sort-name": "Cheung, Jacky", "aliases": []}) == "Jacky Cheung"
    assert pick_english_alias({"sort-name": "王菲", "aliases": []}) is None


# -- iTunes ----------------------------------------------------------------------------


def _itunes_payload() -> dict:
    return {
        "resultCount": 2,
        "results": [
            {
                "kind": "song",
                "trackName": "Uptown Funk (feat. Bruno Mars)",
                "artistName": "Mark Ronson",
                "collectionName": "Uptown Funk - Single",
                "trackTimeMillis": 270_000,
                "trackNumber": 1,
                "trackCount": 1,
                "discNumber": 1,
                "discCount": 1,
                "releaseDate": "2014-11-10T08:00:00Z",
                "primaryGenreName": "Pop",
                "artworkUrl100": "https://is1-ssl.mzstatic.com/image/thumb/x/100x100bb.jpg",
            },
            {
                "kind": "song",
                "trackName": "Lose Yourself to Dance",
                "artistName": "Daft Punk",
                "collectionName": "Random Access Memories",
                "primaryGenreName": "Dance",
            },
        ],
    }


def test_itunes_search_params_and_conversion() -> None:
    router = Router().json(host_is("itunes.apple.com"), _itunes_payload())
    itunes = make_itunes(router)
    cands = itunes.search_candidates("Uptown Funk", "Mark Ronson")
    params = router.log[0].url.params
    assert params["entity"] == "song"
    assert params["country"] == "US"
    assert params["term"] == "Mark Ronson Uptown Funk"
    first = cands[0]
    assert first.title == "Uptown Funk"
    assert first.featuring == ["Bruno Mars"]
    assert first.album == "Uptown Funk"
    assert first.primary_type == "Single"
    assert first.duration_seconds == 270.0
    assert first.genre == "Pop"
    assert first.release_date == "2014-11-10"
    assert first.source == "itunes"
    assert cands[1].primary_type == "Album"


def test_itunes_cache_hit_avoids_requests(tmp_path: Path) -> None:
    cache = TTLCache(tmp_path / "c.sqlite3")
    router = Router().json(host_is("itunes.apple.com"), _itunes_payload())
    make_itunes(router, cache=cache).search("Uptown Funk", "Mark Ronson")
    make_itunes(router, cache=cache).search("Uptown Funk", "Mark Ronson")
    assert router.count() == 1
    cache.close()


def test_clean_album_name_and_artwork_variants() -> None:
    assert clean_album_name("Shake It Off - Single") == ("Shake It Off", "Single")
    assert clean_album_name("Some EP Name - EP") == ("Some EP Name", "EP")
    assert clean_album_name("1989") == ("1989", "Album")
    url = "https://is1-ssl.mzstatic.com/image/thumb/a/b/100x100bb.jpg"
    assert artwork_url_variants(url) == [
        "https://is1-ssl.mzstatic.com/image/thumb/a/b/1000x1000bb.jpg",
        "https://is1-ssl.mzstatic.com/image/thumb/a/b/600x600bb.jpg",
    ]
    assert artwork_url_variants("https://is1-ssl.mzstatic.com/image/cover.jpg") == []
    assert results_to_candidates({"results": [{"kind": "music-video", "trackName": "x"}]}) == []


def test_itunes_artwork_rewrite_verifies_min_side() -> None:
    router = Router()
    # The 1000x1000bb variant silently returns a tiny 100x100 image; it must be rejected.
    router.add(
        lambda r: "1000x1000bb" in r.url.path,
        lambda r: httpx.Response(200, content=jpeg_bytes(100, 100)),
    )
    router.add(
        lambda r: "600x600bb" in r.url.path,
        lambda r: httpx.Response(200, content=jpeg_bytes(600, 600)),
    )
    itunes = make_itunes(router)
    data, url = itunes.fetch_artwork("https://is1-ssl.mzstatic.com/image/thumb/z/100x100bb.jpg")
    assert url.endswith("600x600bb.jpg")
    assert data is not None
    assert all("100x100bb" not in r.url.path for r in router.log)


def test_itunes_artwork_never_accepts_small_images() -> None:
    router = Router().add(
        lambda r: True, lambda r: httpx.Response(200, content=jpeg_bytes(299, 400))
    )
    itunes = make_itunes(router)
    assert itunes.fetch_artwork("https://x.mzstatic.com/100x100bb.jpg") == (None, None)
    assert itunes.fetch_artwork(None) == (None, None)


# -- ArtworkFetcher ----------------------------------------------------------------------


def test_artwork_fetcher_caa_success_and_cache(tmp_path: Path) -> None:
    cache = TTLCache(tmp_path / "c.sqlite3")
    img = jpeg_bytes(1200, 1200)
    router = Router().add(
        path_is("/release/rel-1/front-1200"), lambda r: httpx.Response(200, content=img)
    )
    art = ArtworkFetcher(make_mb(router), None, cache=cache)
    outcome = art.fetch(release_id="rel-1", release_group_id="rg-1")
    assert outcome.source == "coverartarchive"
    assert outcome.warning is None
    assert outcome.data is not None and outcome.data[:2] == b"\xff\xd8"
    assert outcome.data != img  # Resized from 1200 to 1000.
    again = ArtworkFetcher(make_mb(router), None, cache=cache)
    assert again.fetch(release_id="rel-1").data == outcome.data
    assert router.count() == 1
    cache.close()


def test_artwork_fetcher_negative_sentinel_then_itunes_fallback(tmp_path: Path) -> None:
    cache = TTLCache(tmp_path / "c.sqlite3")
    router = Router()
    router.add(
        host_is("is1-ssl.mzstatic.com"),
        lambda r: httpx.Response(200, content=jpeg_bytes(1000, 1000)),
    )
    mb, itunes = make_mb(router), make_itunes(router)
    art = ArtworkFetcher(mb, itunes, cache=cache)
    url = "https://is1-ssl.mzstatic.com/image/thumb/q/100x100bb.jpg"
    outcome = art.fetch(release_id="rel-x", release_group_id="rg-x", itunes_artwork_url=url)
    assert outcome.source == "itunes"
    assert cache.get_bytes("artwork", "caa:rel-x") == b""
    caa_requests = router.count("coverartarchive.org")
    assert caa_requests == 5
    # A fresh fetcher honors the cached negative and makes no CAA requests.
    ArtworkFetcher(make_mb(router), make_itunes(router), cache=cache).fetch(
        release_id="rel-x", itunes_artwork_url=url
    )
    assert router.count("coverartarchive.org") == caa_requests
    cache.close()


def test_artwork_fetcher_failures_warn_and_never_raise() -> None:
    router = Router().add(lambda r: True, lambda r: httpx.Response(200, content=b"not an image"))
    art = ArtworkFetcher(make_mb(router), make_itunes(router))
    outcome = art.fetch(release_id="rel-bad", itunes_artwork_url="https://x/100x100bb.jpg")
    assert outcome.data is None
    assert outcome.warning

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    router2 = Router().add(lambda r: True, boom)
    mb = MusicBrainzClient(
        TEST_UA, fetcher=fetcher(router2), art_fetcher=fetcher(router2, max_retries=0)
    )
    outcome2 = ArtworkFetcher(mb, None).fetch(release_id="rel-down")
    assert outcome2.data is None
    assert "failed" in (outcome2.warning or "")


def test_artwork_fetcher_dedupes_concurrent_requests() -> None:
    img = jpeg_bytes(400, 400)
    gate = threading.Event()

    def slow(request: httpx.Request) -> httpx.Response:
        gate.wait(1.0)
        return httpx.Response(200, content=img)

    router = Router().add(path_is("/release/rel-1/front-1200"), slow)
    art = ArtworkFetcher(make_mb(router), None)
    results: list[bytes | None] = []

    def worker() -> None:
        results.append(art.fetch(release_id="rel-1").data)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for th in threads:
        th.start()
    gate.set()
    for th in threads:
        th.join()
    assert router.count() == 1
    assert len(results) == 6 and all(r == results[0] and r for r in results)


# -- Review regressions ---------------------------------------------------------------


def test_redirect_loop_becomes_provider_error() -> None:
    router = Router().add(
        path_is("/loop"),
        lambda r: httpx.Response(302, headers={"Location": "https://musicbrainz.org/loop"}),
    )
    f = fetcher(router)
    with pytest.raises(ProviderError, match="TooManyRedirects"):
        f.get_json("https://musicbrainz.org/loop")
    with pytest.raises(ProviderError):
        f.get_bytes("https://musicbrainz.org/loop")


def test_non_object_json_becomes_provider_error() -> None:
    router = Router().json(path_is("/list"), [1, 2, 3])
    with pytest.raises(ProviderError, match="expected an object"):
        fetcher(router).get_json("https://musicbrainz.org/list")


def test_search_limit_is_musicbrainz_maximum() -> None:
    router = Router()
    make_mb(router).search_recordings("Song", "Artist")
    assert router.log[0].url.params["limit"] == "100"


def test_missing_medium_count_stays_unknown() -> None:
    payload = search_payload(
        recording("r", "Song", credit(("Artist", "")), [release("rel", "Album", "rg", disc=2)])
    )
    ref = recordings_to_candidates(payload)[0].releases[0]
    assert ref.disc_count is None
    assert ref.disc_number == 2


def test_release_group_lookup_uses_release_group_namespace(tmp_path: Path) -> None:
    cache = TTLCache(tmp_path / "c.sqlite3")
    router = Router().json(path_is("/ws/2/release-group/rg-1"), {"id": "rg-1", "genres": []})
    make_mb(router, cache=cache).get_release_group("rg-1")
    assert router.log[0].url.params["inc"] == "genres"
    assert {s.namespace: s.rows for s in cache.stats()}["release_group"] == 1
    make_mb(router, cache=cache).get_release_group("rg-1")
    assert router.count() == 1
    cache.close()


@pytest.mark.parametrize(
    ("raw", "query", "track_title", "lead", "rest"),
    [
        (
            "Daft Punk, Pharrell Williams & Nile Rodgers",
            None,
            None,
            "Daft Punk, Pharrell Williams & Nile Rodgers",
            [],
        ),
        (
            "Daft Punk, Pharrell Williams & Nile Rodgers",
            None,
            "Get Lucky (feat. Pharrell Williams & Nile Rodgers)",
            "Daft Punk",
            ["Pharrell Williams", "Nile Rodgers"],
        ),
        (
            "Daft Punk, Pharrell Williams & Nile Rodgers",
            "Daft Punk",
            None,
            "Daft Punk",
            ["Pharrell Williams", "Nile Rodgers"],
        ),
        ("Simon & Garfunkel", None, None, "Simon & Garfunkel", []),
        ("Simon & Garfunkel", "Simon & Garfunkel", None, "Simon & Garfunkel", []),
        ("Bob Marley & The Wailers", "Bob Marley", None, "Bob Marley & The Wailers", []),
        ("Prince & The Revolution", "Prince", None, "Prince & The Revolution", []),
        ("Tom Petty & The Heartbreakers", "Tom Petty", None, "Tom Petty & The Heartbreakers", []),
        ("Tyler, The Creator", None, "See You Again", "Tyler, The Creator", []),
        ("Earth, Wind & Fire", None, "Shining Star", "Earth, Wind & Fire", []),
        ("Crosby, Stills, Nash & Young", None, "Ohio", "Crosby, Stills, Nash & Young", []),
        ("Peter, Bjorn and John", None, "Young Folks", "Peter, Bjorn and John", []),
        ("Beyoncé & JAY-Z", "Beyoncé", None, "Beyoncé & JAY-Z", []),
        ("Macklemore & Ryan Lewis", "Macklemore", None, "Macklemore & Ryan Lewis", []),
        ("Daryl Hall & John Oates", "Daryl Hall", None, "Daryl Hall & John Oates", []),
        (
            "Macklemore & Ryan Lewis",
            "Macklemore",
            "Thrift Shop (feat. Wanz)",
            "Macklemore & Ryan Lewis",
            [],
        ),
        ("Beyoncé & JAY-Z", "Beyoncé", "Crazy in Love (feat. JAY-Z)", "Beyoncé", ["JAY-Z"]),
        ("Earth, Wind & Fire", "Earth Wind and Fire", None, "Earth, Wind & Fire", []),
        ("Adele", "Adele", None, "Adele", []),
    ],
)
def test_split_itunes_artist(
    raw: str, query: str | None, track_title: str | None, lead: str, rest: list[str]
) -> None:
    assert split_itunes_artist(raw, query, track_title=track_title) == (lead, rest)


def test_itunes_collaboration_credits_share_identity_with_lead() -> None:
    payload = {
        "results": [
            {
                "kind": "song",
                "trackName": "Get Lucky",
                "artistName": "Daft Punk, Pharrell Williams & Nile Rodgers",
                "collectionName": "Random Access Memories",
            },
            {
                "kind": "song",
                "trackName": "Get Lucky (Radio Edit)",
                "artistName": "Daft Punk",
                "collectionName": "Get Lucky - Single",
            },
        ]
    }
    first, second = results_to_candidates(payload, query_artist="Daft Punk")
    assert first.artist == "Daft Punk"
    assert first.featuring == ["Pharrell Williams", "Nile Rodgers"]
    assert identity_key(first) == identity_key(second)
    q = Query(title="Get Lucky", artist="Daft Punk")
    assert evaluate_candidates(q, [first, second], MatchingConfig()).decision == (
        MatchDecision.AUTO_ACCEPT
    )


def test_itunes_compilations_and_placeholder_dates() -> None:
    payload = {
        "results": [
            {
                "kind": "song",
                "trackName": "吻別",
                "artistName": "Jacky Cheung",
                "collectionName": "張學友4合1珍藏集",
                "discCount": 4,
                "trackNumber": 13,
                "releaseDate": "1993-01-01T08:00:00Z",
            },
            {
                "kind": "song",
                "trackName": "吻別",
                "artistName": "Jacky Cheung",
                "collectionName": "吻別",
                "discCount": 1,
                "trackNumber": 1,
                "releaseDate": "1993-03-05T08:00:00Z",
            },
            {
                "kind": "song",
                "trackName": "X",
                "artistName": "Y",
                "collectionName": "The Very Best of Y",
                "discCount": 1,
            },
        ]
    }
    comp, album, best_of = results_to_candidates(payload)
    assert comp.secondary_types == ["Compilation"]
    assert comp.release_group_date == "1993"
    assert comp.release_date == "1993"
    assert album.secondary_types == []
    assert album.release_group_date == "1993-03-05"
    assert album.release_date == "1993-03-05"
    assert best_of.secondary_types == ["Compilation"]
    # A 3-disc studio album is NOT mislabeled as a compilation, while a 1-track collection
    # without ' - Single' is classified as Single and year-range collections as Compilation.
    extra = results_to_candidates(
        {
            "results": [
                {
                    "kind": "song",
                    "trackName": "Song",
                    "artistName": "Band",
                    "collectionName": "Triple Studio Album",
                    "discCount": 3,
                    "trackCount": 12,
                },
                {
                    "kind": "song",
                    "trackName": "Get Lucky",
                    "artistName": "Daft Punk",
                    "collectionName": "Get Lucky [Daft Punk Remix]",
                    "trackCount": 1,
                },
                {
                    "kind": "song",
                    "trackName": "One More Time",
                    "artistName": "Daft Punk",
                    "collectionName": "Musique, Vol. 1 (1993-2005)",
                    "trackCount": 15,
                },
            ]
        }
    )
    assert extra[0].secondary_types == [] and extra[0].primary_type == "Album"
    assert extra[1].primary_type == "Single"
    assert extra[2].secondary_types == ["Compilation"]
    q = Query(title="吻別", artist="張學友", artist_aliases=("Jacky Cheung",))
    ranked = rank_candidates(q, [comp, album])
    assert len(ranked) == 1
    assert ranked[0].score >= 0.95
    chosen = choose_release_group(q, ranked[0].candidates)
    assert (chosen.album, chosen.track_number) == ("吻別", 1)


def test_itunes_earliest_studio_album_beats_later_eponymous_and_remix_singles() -> None:
    # Earlier studio album beats a later eponymous album; title-named breaks same-date ties.
    payload = {
        "results": [
            {
                "kind": "song",
                "trackName": "Song",
                "artistName": "Artist",
                "collectionName": "Other Album",
                "trackCount": 10,
                "releaseDate": "1990-02-02T08:00:00Z",
            },
            {
                "kind": "song",
                "trackName": "Song",
                "artistName": "Artist",
                "collectionName": "Song",
                "trackCount": 10,
                "releaseDate": "1991-02-02T08:00:00Z",
            },
        ]
    }
    q = Query(title="Song", artist="Artist")
    group = rank_candidates(q, results_to_candidates(payload))[0].candidates
    assert choose_release_group(q, group).album == "Other Album"

    same_day = {
        "results": [
            {
                "kind": "song",
                "trackName": "Song",
                "artistName": "Artist",
                "collectionName": "Other Album",
                "trackCount": 10,
                "releaseDate": "1990-02-02T08:00:00Z",
            },
            {
                "kind": "song",
                "trackName": "Song",
                "artistName": "Artist",
                "collectionName": "Song",
                "trackCount": 10,
                "releaseDate": "1990-02-02T08:00:00Z",
            },
        ]
    }
    group_tie = rank_candidates(q, results_to_candidates(same_day))[0].candidates
    assert choose_release_group(q, group_tie).album == "Song"

    # Live-shaped Get Lucky: 1-track remix single and 3-track single vs. Random Access Memories.
    get_lucky = {
        "results": [
            {
                "kind": "song",
                "trackName": "Get Lucky (Daft Punk Remix)",
                "artistName": "Daft Punk, Pharrell Williams & Nile Rodgers",
                "collectionName": "Get Lucky [Daft Punk Remix]",
                "trackNumber": 1,
                "trackCount": 1,
                "releaseDate": "2013-07-03T07:00:00Z",
            },
            {
                "kind": "song",
                "trackName": "Get Lucky (Radio Edit)",
                "artistName": "Daft Punk, Pharrell Williams & Nile Rodgers",
                "collectionName": "Get Lucky",
                "trackNumber": 1,
                "trackCount": 3,
                "releaseDate": "2013-04-19T07:00:00Z",
            },
            {
                "kind": "song",
                "trackName": "Get Lucky (feat. Pharrell Williams & Nile Rodgers)",
                "artistName": "Daft Punk",
                "collectionName": "Random Access Memories",
                "trackNumber": 8,
                "trackCount": 13,
                "releaseDate": "2013-05-17T07:00:00Z",
            },
        ]
    }
    gl_cands = results_to_candidates(get_lucky, query_artist="Daft Punk")
    assert gl_cands[0].primary_type == "Single"
    assert gl_cands[1].primary_type == "Single"
    gl_out = evaluate_candidates(
        Query(title="Get Lucky", artist="Daft Punk"), gl_cands, MatchingConfig()
    )
    assert gl_out.decision == MatchDecision.AUTO_ACCEPT
    assert gl_out.best is not None
    assert (gl_out.best.album, gl_out.best.track_number) == ("Random Access Memories", 8)

    # Live-shaped One More Time: earlier Radio Edit on compilation & earlier untagged Radio Edit
    # must never beat the clean studio track on Discovery.
    omt = {
        "results": [
            {
                "kind": "song",
                "trackName": "One More Time (Short Radio Edit)",
                "artistName": "Daft Punk",
                "collectionName": "Musique, Vol. 1 (1993-2005)",
                "trackNumber": 6,
                "trackCount": 15,
                "releaseDate": "2000-11-13T08:00:00Z",
            },
            {
                "kind": "song",
                "trackName": "One More Time (Radio Edit)",
                "artistName": "Daft Punk",
                "collectionName": "Early Promo Album",
                "trackNumber": 1,
                "trackCount": 10,
                "releaseDate": "2000-11-13T08:00:00Z",
            },
            {
                "kind": "song",
                "trackName": "One More Time",
                "artistName": "Daft Punk",
                "collectionName": "Discovery",
                "trackNumber": 1,
                "trackCount": 14,
                "releaseDate": "2001-03-12T08:00:00Z",
            },
        ]
    }
    omt_cands = results_to_candidates(omt, query_artist="Daft Punk")
    omt_out = evaluate_candidates(
        Query(title="One More Time", artist="Daft Punk"), omt_cands, MatchingConfig()
    )
    assert omt_out.decision == MatchDecision.AUTO_ACCEPT
    assert omt_out.best is not None
    assert (omt_out.best.album, omt_out.best.track_number) == ("Discovery", 1)


def test_artwork_fetcher_falls_back_to_itunes_lookup_for_mb_candidate(tmp_path: Path) -> None:
    from audio_from_video.models import Candidate, ReleaseRef

    cache = TTLCache(tmp_path / "c.sqlite3")
    router = Router()
    router.json(
        host_is("itunes.apple.com"),
        {
            "results": [
                {
                    "kind": "song",
                    "trackName": "One More Time",
                    "artistName": "Daft Punk",
                    "collectionName": "Discovery",
                    "artworkUrl100": "https://is1-ssl.mzstatic.com/image/thumb/a/100x100bb.jpg",
                }
            ]
        },
    )
    router.add(
        host_is("is1-ssl.mzstatic.com"),
        lambda r: httpx.Response(200, content=jpeg_bytes(1000, 1000)),
    )
    art = ArtworkFetcher(make_mb(router), make_itunes(router, cache=cache), cache=cache)
    rel = ReleaseRef(release_id="rel-no-caa", title="Discovery")
    cand = Candidate(
        source="musicbrainz",
        title="One More Time",
        artist="Daft Punk",
        lead_artist="Daft Punk",
        album="Discovery",
        chosen_release=rel,
        release_group_id="rg-no-caa",
        artwork_url=None,
    )
    outcome = art.fetch_for_candidate(cand)
    assert outcome.source == "itunes"
    assert outcome.data is not None and outcome.data[:2] == b"\xff\xd8"
    assert router.count("itunes.apple.com") == 1
    # Second track from the same album reuses the cached iTunes lookup URL and artwork.
    cand2 = Candidate(
        source="musicbrainz",
        title="Aerodynamic",
        artist="Daft Punk",
        lead_artist="Daft Punk",
        album="Discovery",
        chosen_release=rel,
        release_group_id="rg-no-caa",
        artwork_url=None,
    )
    outcome2 = art.fetch_for_candidate(cand2)
    assert outcome2.source == "itunes"
    assert router.count("itunes.apple.com") == 1
    cache.close()


def test_split_artist_credit_does_not_leak_member_aliases_on_multi_artist_credits() -> None:
    from audio_from_video.providers.musicbrainz import split_artist_credit

    mashup = [
        {"name": "Daughter", "joinphrase": " / ", "artist": {"id": "a1", "name": "Daughter"}},
        {
            "name": "Daft Punk",
            "joinphrase": "",
            "artist": {
                "id": "a2",
                "name": "Daft Punk",
                "aliases": [{"name": "daft punk"}, {"name": "ダフト・パンク"}],
            },
        },
    ]
    primary, feats, aliases, lead, _lid, _fids, _full = split_artist_credit(mashup)
    assert primary == "Daughter / Daft Punk"
    assert lead == "Daughter"
    assert feats == []
    assert aliases == []

    solo_feat = [
        {
            "name": "Daft Punk",
            "joinphrase": " feat. ",
            "artist": {
                "id": "a2",
                "name": "Daft Punk",
                "aliases": [{"name": "ダフト・パンク"}],
            },
        },
        {"name": "Pharrell Williams", "joinphrase": "", "artist": {"id": "a3", "name": "Pharrell"}},
    ]
    primary2, feats2, aliases2, *_ = split_artist_credit(solo_feat)
    assert primary2 == "Daft Punk"
    assert feats2 == ["Pharrell Williams"]
    assert "ダフト・パンク" in aliases2


def test_search_artist_names_filters_out_unrelated_substring_acts() -> None:
    router = Router().json(
        path_is("/ws/2/artist/"),
        {
            "artists": [
                {"id": "a-symph", "name": "Симфоническое Кино", "aliases": []},
                {"id": "a-frg", "name": "Кино ФРГ", "aliases": []},
                {
                    "id": "a-kino",
                    "name": "Кино",
                    "aliases": [{"name": "Kino"}],
                },
            ]
        },
    )
    assert make_mb(router).search_artist_names("Кино") == ["Кино"]

    router2 = Router().json(
        path_is("/ws/2/artist/"),
        {
            "artists": [
                {"id": "a-iu", "name": "IU", "aliases": [{"name": "아이유"}]},
                {"id": "a-other", "name": "IU Tribute", "aliases": []},
            ]
        },
    )
    assert make_mb(router2).search_artist_names("아이유") == ["IU"]


def test_scrub_error_urls_and_artwork_breaker_warns_once() -> None:
    from audio_from_video.providers.base import scrub_error_urls

    for raw, expected in (
        (
            "HTTP 400 from https://musicbrainz.org/ws/2/recording/.",
            "HTTP 400.",
        ),
        (
            "Request to https://musicbrainz.org/ws/2/recording/ failed after retries (HTTP 503).",
            "Request failed after retries (HTTP 503).",
        ),
        (
            "Unexpected JSON from https://itunes.apple.com/search: expected an object, got list.",
            "Unexpected JSON: expected an object, got list.",
        ),
    ):
        cleaned = scrub_error_urls(ProviderError(raw))
        assert cleaned == expected
        assert ".org" not in cleaned and ".com" not in cleaned and "https://" not in cleaned

    # Artwork circuit breaker emits the skip warning once per host and strips URLs.
    router = Router().add(host_is("coverartarchive.org"), lambda r: httpx.Response(503))
    mb = MusicBrainzClient(
        TEST_UA, fetcher=fetcher(router), art_fetcher=fetcher(router, max_retries=0)
    )
    art = ArtworkFetcher(mb, None)
    r1 = art.fetch(release_id="rel-1")
    r2 = art.fetch(release_id="rel-2")
    r3 = art.fetch(release_id="rel-3")
    r4 = art.fetch(release_id="rel-4")
    assert r1.warning and "https://" not in r1.warning and ".org" not in r1.warning
    assert r2.warning and "https://" not in r2.warning
    assert r3.warning and "skipped after earlier failures" in r3.warning
    assert r4.warning and "skipped after earlier failures" not in r4.warning


def test_transport_errors_cap_retries_below_http_max_retries() -> None:
    calls = {"n": 0}

    def boom(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("offline", request=request)

    f = HttpFetcher(
        transport=httpx.MockTransport(boom),
        sleep=lambda _s: None,
        max_retries=4,
    )
    with pytest.raises(ProviderError):
        f.get_json("https://musicbrainz.org/ws/2/recording/")
    assert calls["n"] == 2  # Initial attempt + 1 transport retry, not 5 attempts.


def test_client_memos_are_bounded_lru() -> None:
    from audio_from_video.providers.musicbrainz import SEARCH_MEMO_MAX

    router = Router().json(host_is("musicbrainz.org"), search_payload())
    mb = make_mb(router)
    for i in range(SEARCH_MEMO_MAX + 10):
        mb.search_recordings(f"Song {i}", "Artist")
    assert len(mb._search_memo) == SEARCH_MEMO_MAX


def test_artwork_fetchers_retry_once() -> None:
    assert MusicBrainzClient(TEST_UA).art_fetcher.max_retries == 1
    assert ITunesClient().art_fetcher.max_retries == 1


def test_itunes_artwork_server_errors_raise_for_circuit_breaker() -> None:
    router = Router().add(lambda r: True, lambda r: httpx.Response(503))
    itunes = make_itunes(router)
    with pytest.raises(ProviderError):
        itunes.fetch_artwork("https://is1-ssl.mzstatic.com/image/thumb/a/100x100bb.jpg")


def test_artwork_circuit_breaker_skips_failed_host_for_the_run() -> None:
    router = Router()
    router.add(host_is("coverartarchive.org"), lambda r: httpx.Response(503))
    router.add(
        host_is("is1-ssl.mzstatic.com"), lambda r: httpx.Response(200, content=jpeg_bytes(600, 600))
    )
    mb = MusicBrainzClient(
        TEST_UA, fetcher=fetcher(router), art_fetcher=fetcher(router, max_retries=1)
    )
    art = ArtworkFetcher(mb, make_itunes(router))
    url = "https://is1-ssl.mzstatic.com/image/thumb/a/100x100bb.jpg"
    first = art.fetch(release_id="rel-1", itunes_artwork_url=url)
    assert first.source == "itunes"
    second = art.fetch(release_id="rel-2", itunes_artwork_url=url)
    assert second.source == "itunes"
    caa_after_two = router.count("coverartarchive.org")
    assert caa_after_two == 4  # Two consecutive transient host failures trip the breaker.
    third = art.fetch(release_id="rel-3", itunes_artwork_url=url)
    assert third.source == "itunes"
    assert router.count("coverartarchive.org") == caa_after_two


def test_artwork_per_url_4xx_and_size_cap_continue_cascade_without_tripping() -> None:
    # 1000x1000bb returns 403 while 600x600bb returns 200 -> cascade succeeds and host stays up.
    router = Router()
    router.add(lambda r: "1000x1000bb" in r.url.path, lambda r: httpx.Response(403))
    router.add(
        lambda r: "600x600bb" in r.url.path,
        lambda r: httpx.Response(200, content=jpeg_bytes(600, 600)),
    )
    itunes = make_itunes(router)
    data, url = itunes.fetch_artwork("https://is1-ssl.mzstatic.com/image/thumb/x/100x100bb.jpg")
    assert data is not None and url and url.endswith("600x600bb.jpg")

    # CAA front-1200 exceeds size cap (Content-Length > 15 MiB); front-500 succeeds.
    router2 = Router()
    router2.add(
        path_is("/release/rel-1/front-1200"),
        lambda r: httpx.Response(200, headers={"Content-Length": str(20 * 1024 * 1024)}),
    )
    router2.add(
        path_is("/release/rel-1/front-500"),
        lambda r: httpx.Response(200, content=jpeg_bytes(500, 500)),
    )
    art = ArtworkFetcher(make_mb(router2), None)
    outcome = art.fetch(release_id="rel-1")
    assert outcome.source == "coverartarchive"
    assert outcome.data is not None


def test_unquoted_lucene_escapes_boolean_operators() -> None:
    q = build_recording_query("BELIEVE IT OR NOT", "NOT LIKE US", quoted=False)
    assert " OR " not in q and " NOT " not in q
    assert "or not)" in q and "not LIKE US" in q


def test_itunes_album_genre_is_cached_per_album(tmp_path: Path) -> None:
    payload = {
        "results": [
            {
                "kind": "song",
                "trackName": "One More Time",
                "artistName": "Daft Punk",
                "collectionName": "Discovery",
                "primaryGenreName": "Electronic",
            },
        ]
    }
    cache = TTLCache(tmp_path / "c.sqlite3")
    router = Router().json(host_is("itunes.apple.com"), payload)
    itunes = make_itunes(router, cache=cache)
    assert itunes.album_genre("Daft Punk", "Discovery", "One More Time") == "Electronic"
    assert itunes.album_genre("Daft Punk", "Discovery", "Aerodynamic") == "Electronic"
    assert router.count() == 1
    fresh = make_itunes(router, cache=cache)
    assert fresh.album_genre("Daft Punk", "Discovery", "Digital Love") == "Electronic"
    assert router.count() == 1
    assert {s.namespace: s.rows for s in cache.stats()}["itunes_album"] == 1
    cache.close()
