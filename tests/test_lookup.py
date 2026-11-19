"""Tests for TrackResolver query planning, retries, fallbacks, enrichment, and batches."""

from __future__ import annotations

import httpx

from audio_from_video.lookup import BatchItem, TrackResolver, title_case_genre
from audio_from_video.models import ManualOverrides, MatchDecision, MatchStatus, ParsedHints
from audio_from_video.providers.musicbrainz import MusicBrainzClient
from provider_fixtures import (
    TEST_UA,
    Router,
    credit,
    fetcher,
    host_is,
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

DP = credit(("Daft Punk", "", "artist-dp", "Daft Punk"))


def _omt_search() -> dict:
    return search_payload(
        recording(
            "rec-omt",
            "One More Time",
            DP,
            [
                release("rel-disc", "Discovery", "rg-disc", track_number=1),
                release(
                    "rel-disc-jp",
                    "Discovery",
                    "rg-disc",
                    track_number=1,
                    country="JP",
                    medium_track_count=15,
                    date="2001-03-20",
                ),
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


def _disc_lookup(genres: list | None = None, rg_genres: list | None = None) -> dict:
    return release_lookup(
        "rel-disc",
        "Discovery",
        artist_credits=DP,
        rg_id="rg-disc",
        rg_first_date="2001-03-07",
        date="2001-03-12",
        media=[medium(1, [("rec-omt", "One More Time"), ("rec-aero", "Aerodynamic")])],
        genres=genres,
        rg_genres=rg_genres,
    )


def _resolver(router: Router, *, with_itunes: bool = True) -> TrackResolver:
    return TrackResolver(make_mb(router), make_itunes(router) if with_itunes else None)


def test_hintless_input_makes_zero_requests() -> None:
    router = Router()
    resolver = _resolver(router)
    result = resolver.resolve(ParsedHints(is_hintless=True, clean_stem="VID_0042"))
    assert result.decision == MatchDecision.SKIPPED
    assert router.count() == 0
    assert result.queries_tried == []
    assert resolver.resolve(ParsedHints(artist="Someone")).decision == MatchDecision.SKIPPED
    assert router.count() == 0


def test_auto_accept_enriches_from_single_release_lookup() -> None:
    router = Router()
    router.json(query_has('recording:"One More Time" AND artist:"Daft Punk"'), _omt_search())
    router.json(
        path_is("/ws/2/release/rel-disc"),
        _disc_lookup(genres=[{"name": "house", "count": 5}, {"name": "electronic", "count": 3}]),
    )
    resolver = _resolver(router)
    result = resolver.resolve(
        ParsedHints(
            artist="Daft Punk", title="One More Time", trusted_artist=True, trusted_title=True
        )
    )
    assert result.decision == MatchDecision.AUTO_ACCEPT
    md = result.metadata
    assert (md.artist, md.title, md.album, md.album_artist) == (
        "Daft Punk",
        "One More Time",
        "Discovery",
        "Daft Punk",
    )
    assert (md.track_number, md.track_total, md.disc_number, md.disc_total) == (1, 2, 1, 1)
    assert md.date == "2001-03-12"
    assert md.original_year == "2001"
    assert md.genre == "House"
    assert md.mb_recording_id == "rec-omt"
    assert md.mb_release_id == "rel-disc"
    assert md.mb_release_group_id == "rg-disc"
    assert md.mb_artist_id == "artist-dp"
    assert md.match_status == MatchStatus.EXACT
    paths = [r.url.path for r in router.log]
    assert paths.count("/ws/2/release/rel-disc") == 1
    assert not any(p.startswith("/ws/2/artist/") for p in paths)
    assert router.count("itunes.apple.com") == 0
    assert result.queries_tried == [
        {"provider": "musicbrainz", "artist": "Daft Punk", "title": "One More Time"}
    ]


def test_genre_falls_back_to_release_group_then_itunes() -> None:
    router = Router()
    router.json(query_has("One More Time"), _omt_search())
    router.json(
        path_is("/ws/2/release/rel-disc"),
        _disc_lookup(rg_genres=[{"name": "french house", "count": 2}]),
    )
    md = _resolver(router).resolve(ParsedHints(artist="Daft Punk", title="One More Time")).metadata
    assert md.genre == "French House"

    router2 = Router()
    router2.json(query_has("One More Time"), _omt_search())
    router2.json(path_is("/ws/2/release/rel-disc"), _disc_lookup())
    router2.json(
        host_is("itunes.apple.com"),
        {
            "results": [
                {
                    "kind": "song",
                    "trackName": "Other Song",
                    "artistName": "Daft Punk",
                    "primaryGenreName": "Rock",
                },
                {
                    "kind": "song",
                    "trackName": "One More Time",
                    "artistName": "Daft Punk",
                    "collectionName": "Discovery",
                    "primaryGenreName": "Electronic",
                },
            ]
        },
    )
    md2 = (
        _resolver(router2).resolve(ParsedHints(artist="Daft Punk", title="One More Time")).metadata
    )
    assert md2.genre == "Electronic"
    assert router2.count("itunes.apple.com") == 1


def test_title_case_genre() -> None:
    assert title_case_genre("hip hop") == "Hip Hop"
    assert title_case_genre("r&b") == "R&B"
    assert title_case_genre("drum and bass") == "Drum and Bass"
    assert title_case_genre("synth-pop") == "Synth-Pop"
    assert title_case_genre("edm") == "EDM"
    assert title_case_genre("uk garage") == "UK Garage"
    assert title_case_genre("idm") == "IDM"
    assert title_case_genre("dnb") == "DnB"
    assert title_case_genre("ost") == "OST"
    assert title_case_genre("hi-nrg") == "Hi-NRG"


def test_disc_two_track_position() -> None:
    pf = credit(("Pink Floyd", "", "artist-pf"))
    search = search_payload(
        recording(
            "rec-cn",
            "Comfortably Numb",
            pf,
            [
                release(
                    "rel-wall",
                    "The Wall",
                    "rg-wall",
                    disc=2,
                    track_number=6,
                    medium_track_count=13,
                    track_count=26,
                    media_count=2,
                    date="1979-11-30",
                ),
            ],
        )
    )
    lookup = release_lookup(
        "rel-wall",
        "The Wall",
        artist_credits=pf,
        rg_id="rg-wall",
        rg_first_date="1979-11-30",
        date="1979-11-30",
        media=[
            medium(1, [(f"rec-d1-{i}", f"Song {i}") for i in range(1, 14)]),
            medium(
                2,
                [
                    *[(f"rec-d2-{i}", f"Song {i}") for i in range(1, 6)],
                    ("rec-cn", "Comfortably Numb"),
                    *[(f"rec-d2-{i}", f"Song {i}") for i in range(7, 14)],
                ],
            ),
        ],
    )
    router = (
        Router()
        .json(query_has("Comfortably Numb"), search)
        .json(path_is("/ws/2/release/rel-wall"), lookup)
    )
    hints = ParsedHints(
        title="Comfortably Numb",
        track_number=6,
        disc_number=2,
        parent_artist_hint="Pink Floyd",
        parent_album_hint="The Wall",
    )
    result = _resolver(router).resolve(hints)
    assert result.decision == MatchDecision.AUTO_ACCEPT
    md = result.metadata
    assert (md.disc_number, md.track_number, md.disc_total, md.track_total) == (2, 6, 2, 13)


def test_folder_artist_used_only_for_full_shape_with_track_number() -> None:
    router = Router()
    resolver = _resolver(router, with_itunes=False)
    full = ParsedHints(
        title="Bohemian Rhapsody",
        track_number=11,
        parent_artist_hint="Queen",
        parent_album_hint="A Night at the Opera",
    )
    resolver.resolve(full)
    assert router.log[0].url.params["query"] == (
        'recording:"Bohemian Rhapsody" AND artist:"Queen"'
        ' AND release:"A Night at the Opera" AND status:official'
    )
    no_track = ParsedHints(
        title="Bohemian Rhapsody",
        parent_artist_hint="Queen",
        parent_album_hint="A Night at the Opera",
    )
    assert TrackResolver.query_artist(no_track) is None
    noisy = ParsedHints(
        title="X",
        track_number=1,
        parent_artist_hint="A",
        parent_album_hint="B",
        downloader_signals=["url"],
    )
    assert TrackResolver.query_artist(noisy) is None


def test_title_only_query_never_auto_accepts() -> None:
    search = search_payload(
        recording(
            "rec-y",
            "Yesterday",
            credit(("The Beatles", "")),
            [release("rel-help", "Help!", "rg-help", date="1965-08-06")],
        )
    )
    router = Router().json(query_has("Yesterday"), search)
    result = _resolver(router, with_itunes=False).resolve(ParsedHints(title="Yesterday"))
    assert result.decision == MatchDecision.AMBIGUOUS
    assert not any("/release/" in r.url.path for r in router.log)


def test_edition_tails_stripped_from_query_title() -> None:
    router = Router()
    _resolver(router, with_itunes=False).resolve(
        ParsedHints(artist="Roxy Music", title="More Than This - 2012 Remaster")
    )
    assert router.log[0].url.params["query"] == (
        'recording:"More Than This" AND artist:"Roxy Music"'
        " AND status:official AND primarytype:album"
    )


def test_swapped_retry_after_low_primary_score() -> None:
    e65 = credit(("Eiffel 65", "", "artist-e65"))
    search = search_payload(
        recording(
            "rec-blue",
            "Blue (Da Ba Dee)",
            e65,
            [release("rel-euro", "Europop", "rg-euro", date="1999-01-01")],
        )
    )
    router = Router().json(query_has('artist:"Eiffel 65"'), search)
    hints = ParsedHints(
        artist="Blue (Da Ba Dee)",
        title="Eiffel 65",
        swapped_candidate=("Eiffel 65", "Blue (Da Ba Dee)"),
        original_embedded_pair=("Night Drive Mixes", "Blue (Da Ba Dee) - Eiffel 65"),
        downloader_signals=["title_mismatch"],
    )
    result = _resolver(router, with_itunes=False).resolve(hints)
    assert result.decision == MatchDecision.AUTO_ACCEPT
    assert result.metadata.artist == "Eiffel 65"
    assert [q["artist"] for q in result.queries_tried] == ["Blue (Da Ba Dee)", "Eiffel 65"]


def test_original_embedded_pair_tried_after_swapped() -> None:
    router = Router()
    hints = ParsedHints(
        artist="Left",
        title="Right",
        swapped_candidate=("Right", "Left"),
        original_embedded_pair=("Uploader", "Left - Right"),
    )
    result = _resolver(router, with_itunes=False).resolve(hints)
    assert [q["artist"] for q in result.queries_tried] == ["Left", "Right", "Uploader"]
    assert result.decision == MatchDecision.UNMATCHED


def test_itunes_fallback_when_musicbrainz_has_nothing() -> None:
    router = Router()
    router.json(
        host_is("itunes.apple.com"),
        {
            "results": [
                {
                    "kind": "song",
                    "trackName": "Uptown Funk (feat. Bruno Mars)",
                    "artistName": "Mark Ronson",
                    "collectionName": "Uptown Special",
                    "trackNumber": 4,
                    "trackCount": 11,
                    "discNumber": 1,
                    "discCount": 1,
                    "releaseDate": "2015-01-12T08:00:00Z",
                    "primaryGenreName": "Pop",
                    "artworkUrl100": "https://is1-ssl.mzstatic.com/image/thumb/u/100x100bb.jpg",
                }
            ]
        },
    )
    result = _resolver(router).resolve(
        ParsedHints(artist="Mark Ronson", title="Uptown Funk", featuring=["Bruno Mars"])
    )
    assert result.decision == MatchDecision.AUTO_ACCEPT
    md = result.metadata
    assert md.source_description == "itunes"
    assert (md.album, md.track_number, md.track_total, md.genre) == ("Uptown Special", 4, 11, "Pop")
    assert md.featuring == ["Bruno Mars"]
    assert md.mb_release_id is None
    assert result.queries_tried[-1]["provider"] == "itunes"


def test_non_latin_aliases_from_artist_and_pseudo_release() -> None:
    jc = [
        {
            "name": "張學友",
            "joinphrase": "",
            "artist": {"id": "a-jc", "name": "張學友", "sort-name": "Cheung, Jacky"},
        }
    ]
    search = search_payload(
        recording(
            "rec-wb",
            "吻別",
            jc,
            [release("rel-wb", "吻別", "rg-wb", date="1993-03-01", country="HK")],
        )
    )
    lookup = release_lookup(
        "rel-wb",
        "吻別",
        artist_credits=jc,
        rg_id="rg-wb",
        rg_first_date="1993-03-01",
        date="1993-03-01",
        media=[medium(1, [("rec-wb", "吻別"), ("rec-2", "其他")])],
        relations=[
            {
                "type": "transl-tracklisting",
                "direction": "forward",
                "release": {"id": "rel-wb-hans", "text-representation": {"script": "Hans"}},
            },
            {
                "type": "transl-tracklisting",
                "direction": "forward",
                "release": {"id": "rel-wb-latn", "text-representation": {"script": "Latn"}},
            },
        ],
    )
    pseudo = release_lookup(
        "rel-wb-latn",
        "The Goodbye Kiss",
        artist_credits=jc,
        rg_id="rg-wb",
        rg_first_date="1993-03-01",
        date="1993-03-01",
        media=[medium(1, [("rec-wb", "The Goodbye Kiss"), ("rec-2", "Other")])],
    )
    artist = {
        "id": "a-jc",
        "name": "張學友",
        "sort-name": "Cheung, Jacky",
        "aliases": [
            {"name": "Jacky Cheung", "locale": "en", "primary": True, "type": "Artist name"}
        ],
    }
    router = (
        Router()
        .json(query_has("吻別"), search)
        .json(path_is("/ws/2/release/rel-wb"), lookup)
        .json(path_is("/ws/2/release/rel-wb-latn"), pseudo)
        .json(path_is("/ws/2/artist/a-jc"), artist)
    )
    result = _resolver(router, with_itunes=False).resolve(
        ParsedHints(artist="張學友", title="吻別")
    )
    assert result.decision == MatchDecision.AUTO_ACCEPT
    md = result.metadata
    assert md.artist == "Jacky Cheung {張學友}"
    assert md.album_artist == "Jacky Cheung {張學友}"
    assert md.title == "The Goodbye Kiss {吻別}"
    assert md.album == "The Goodbye Kiss {吻別}"
    assert not any(r.url.path == "/ws/2/release/rel-wb-hans" for r in router.log)


def test_ambiguous_result_is_not_enriched() -> None:
    search = search_payload(
        recording("r1", "Hurt", credit(("Johnny Cash", "")), [release("x1", "A", "rg1")]),
        recording("r2", "Hurt", credit(("Rosanne Cash", "")), [release("x2", "B", "rg2")]),
    )
    router = Router().json(query_has("Hurt"), search)
    result = _resolver(router, with_itunes=False).resolve(ParsedHints(artist="Cash", title="Hurt"))
    assert result.decision == MatchDecision.AMBIGUOUS
    assert len(result.top_candidates) == 2
    assert result.metadata.match_status != MatchStatus.EXACT
    assert not any("/release/" in r.url.path for r in router.log)


def test_provider_errors_become_warnings() -> None:
    router = Router().add(lambda r: True, lambda r: httpx.Response(503))
    mb = MusicBrainzClient(
        TEST_UA, fetcher=fetcher(router, max_retries=0), art_fetcher=fetcher(router)
    )
    result = TrackResolver(mb, None).resolve(ParsedHints(artist="A", title="B"))
    assert result.decision == MatchDecision.UNMATCHED
    assert result.warnings and "MusicBrainz search failed" in result.warnings[0]
    assert "https://" not in result.warnings[0]


def test_batch_coherence_prefers_dominant_release_group() -> None:
    artist = credit(("Artist", "", "artist-x"))

    def rec(rid: str, title: str, rgs: list[tuple[str, str, str]]) -> dict:
        return recording(
            rid,
            title,
            artist,
            [release(f"rel-{rg}-{rid}", album, rg, date=date) for rg, album, date in rgs],
        )

    later = ("rg-later", "Later", "1999-06-01")
    first = ("rg-first", "First", "1985-06-01")
    router = Router()
    router.json(query_has('"Song A"'), search_payload(rec("ra", "Song A", [later])))
    router.json(query_has('"Song B"'), search_payload(rec("rb", "Song B", [later])))
    router.json(query_has('"Song C"'), search_payload(rec("rc", "Song C", [first, later])))
    resolver = _resolver(router, with_itunes=False)

    single = resolver.resolve(ParsedHints(artist="Artist", title="Song C"))
    assert single.metadata.mb_release_group_id == "rg-first"

    items = [
        BatchItem(ParsedHints(artist="Artist", title=t), group_key="folder-1")
        for t in ("Song A", "Song B", "Song C")
    ]
    results = resolver.resolve_batch(items)
    assert [r.decision for r in results] == [MatchDecision.AUTO_ACCEPT] * 3
    assert [r.metadata.mb_release_group_id for r in results] == ["rg-later"] * 3
    assert results[2].metadata.album == "Later"


def test_batch_without_group_keys_matches_single_resolution() -> None:
    router = Router().json(query_has("One More Time"), _omt_search())
    router.json(path_is("/ws/2/release/rel-disc"), _disc_lookup())
    resolver = _resolver(router, with_itunes=False)
    results = resolver.resolve_batch(
        [
            BatchItem(ParsedHints(artist="Daft Punk", title="One More Time")),
            BatchItem(ParsedHints(is_hintless=True)),
        ]
    )
    assert results[0].metadata.mb_release_id == "rel-disc"
    assert results[1].decision == MatchDecision.SKIPPED


# -- Review regressions ---------------------------------------------------------------


def test_lookup_does_not_import_planner() -> None:
    import subprocess
    import sys

    code = "import sys, audio_from_video.lookup; print('audio_from_video.planner' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
    from audio_from_video import planner
    from audio_from_video.offline_metadata import build_offline_metadata

    assert planner.build_offline_metadata is build_offline_metadata


def test_manual_overrides_are_search_hints_and_pinned_values() -> None:
    router = Router()
    router.json(query_has('recording:"One More Time" AND artist:"Daft Punk"'), _omt_search())
    router.json(path_is("/ws/2/release/rel-disc"), _disc_lookup())
    overrides = ManualOverrides(
        artist="Daft Punk", title="One More Time", album="My Album", track=7
    )
    hintless = ParsedHints(is_hintless=True, clean_stem="VID_0042")
    result = _resolver(router, with_itunes=False).resolve(hintless, overrides=overrides)
    assert result.decision == MatchDecision.AUTO_ACCEPT
    assert router.log[0].url.params["query"] == (
        'recording:"One More Time" AND artist:"Daft Punk"'
        ' AND release:"My Album" AND status:official'
    )
    md = result.metadata
    assert (md.artist, md.title, md.album, md.track_number) == (
        "Daft Punk",
        "One More Time",
        "My Album",
        7,
    )
    assert md.mb_release_id == "rel-disc"


def test_manual_overrides_apply_to_unmatched_fallback_and_batches() -> None:
    router = Router()
    resolver = _resolver(router, with_itunes=False)
    ov = ManualOverrides(artist="Pinned Artist", album_artist="Pinned AA")
    result = resolver.resolve(ParsedHints(title="Some Song", clean_stem="Some Song"), overrides=ov)
    assert result.decision == MatchDecision.UNMATCHED
    assert result.metadata.artist == "Pinned Artist"
    assert result.metadata.album_artist == "Pinned AA"
    batch = resolver.resolve_batch([BatchItem(ParsedHints(title="Some Song"), overrides=ov)])
    assert batch[0].metadata.artist == "Pinned Artist"
    assert router.log[-2].url.params["query"].endswith('artist:"Pinned Artist"')


def test_unexpected_payloads_never_escape_resolve_or_enrich() -> None:
    router = Router().json(query_has("Song"), [1, 2, 3])
    result = _resolver(router).resolve(ParsedHints(artist="A", title="Song"))
    assert result.decision == MatchDecision.UNMATCHED
    assert any("expected an object" in w for w in result.warnings)

    weird = Router().json(query_has("One More Time"), _omt_search())
    weird.json(
        path_is("/ws/2/release/rel-disc"),
        {"title": 5, "media": "nope", "release-group": [], "relations": "x"},
    )
    res2 = _resolver(weird, with_itunes=False).resolve(
        ParsedHints(artist="Daft Punk", title="One More Time")
    )
    assert res2.decision == MatchDecision.AUTO_ACCEPT
    assert res2.metadata.title == "One More Time"


def test_enrich_survives_non_provider_exceptions(monkeypatch) -> None:
    router = Router().json(query_has("One More Time"), _omt_search())
    resolver = _resolver(router, with_itunes=False)

    def boom(_mbid: str) -> dict:
        raise RuntimeError("database exploded")

    monkeypatch.setattr(resolver.mb, "get_release", boom)
    result = resolver.resolve(ParsedHints(artist="Daft Punk", title="One More Time"))
    assert result.decision == MatchDecision.AUTO_ACCEPT
    assert result.metadata.album == "Discovery"
    assert any("database exploded" in w for w in result.warnings)

    def search_boom(*_a: object, **_kw: object) -> list:
        raise ValueError("bad data")

    monkeypatch.setattr(resolver.mb, "search_candidates", search_boom)
    res2 = resolver.resolve(ParsedHints(artist="X", title="Y"))
    assert res2.decision == MatchDecision.UNMATCHED
    assert any("bad data" in w for w in res2.warnings)


def test_release_group_genre_lookup_before_itunes() -> None:
    router = Router()
    router.json(query_has("One More Time"), _omt_search())
    lookup = _disc_lookup()
    del lookup["release-group"]["genres"]
    router.json(path_is("/ws/2/release/rel-disc"), lookup)
    router.json(
        path_is("/ws/2/release-group/rg-disc"),
        {"id": "rg-disc", "genres": [{"name": "french house", "count": 4}]},
    )
    md = _resolver(router).resolve(ParsedHints(artist="Daft Punk", title="One More Time")).metadata
    assert md.genre == "French House"
    assert router.count("itunes.apple.com") == 0
    assert any(r.url.path == "/ws/2/release-group/rg-disc" for r in router.log)


def test_edition_rules_reapplied_with_full_media_count() -> None:
    artist = credit(("Artist", "", "artist-x"))
    search = search_payload(
        recording(
            "rec-s",
            "Song",
            artist,
            [
                release("rel-2cd", "Album", "rg-a", date="2001-01-01", track_number=3),
                release("rel-1cd", "Album", "rg-a", date="2001-06-01", track_number=3),
            ],
        )
    )

    def lookup(rid: str, discs: int) -> dict:
        media = [medium(1, [("rec-a", "A"), ("rec-b", "B"), ("rec-s", "Song")])]
        media += [medium(i, [(f"rec-x{i}", "X")]) for i in range(2, discs + 1)]
        return release_lookup(
            rid,
            "Album",
            artist_credits=artist,
            rg_id="rg-a",
            rg_first_date="2001-01-01",
            date="2001-01-01",
            media=media,
            genres=[{"name": "rock", "count": 1}],
        )

    router = Router().json(query_has("Song"), search)
    router.json(path_is("/ws/2/release/rel-2cd"), lookup("rel-2cd", 2))
    router.json(path_is("/ws/2/release/rel-1cd"), lookup("rel-1cd", 1))
    md = (
        _resolver(router, with_itunes=False)
        .resolve(ParsedHints(artist="Artist", title="Song"))
        .metadata
    )
    assert md.mb_release_id == "rel-1cd"
    assert md.disc_total == 1


def test_itunes_fallback_uses_harvested_musicbrainz_aliases() -> None:
    jc = [
        {
            "name": "張學友",
            "joinphrase": "",
            "artist": {"id": "a-jc", "name": "張學友", "sort-name": "Cheung, Jacky"},
        }
    ]
    other_song = search_payload(
        recording("rec-o", "完全不同的歌", jc, [release("rel-o", "另一張", "rg-o")])
    )
    router = Router().json(query_has("吻別"), other_song)
    router.json(
        host_is("itunes.apple.com"),
        {
            "results": [
                {
                    "kind": "song",
                    "trackName": "吻別",
                    "artistName": "Jacky Cheung",
                    "collectionName": "吻別",
                    "trackNumber": 1,
                    "releaseDate": "1993-03-05T08:00:00Z",
                }
            ]
        },
    )
    result = _resolver(router).resolve(ParsedHints(artist="張學友", title="吻別"))
    assert result.decision == MatchDecision.AUTO_ACCEPT
    assert result.metadata.source_description == "itunes"
    assert (result.metadata.album, result.metadata.track_number) == ("吻別", 1)


def test_record_all_runs_every_query_and_enriches_top_identity() -> None:
    router = Router()
    router.json(query_has('recording:"One More Time" AND artist:"Daft Punk"'), _omt_search())
    router.json(path_is("/ws/2/release/rel-disc"), _disc_lookup())
    hints = ParsedHints(
        artist="Daft Punk", title="One More Time", swapped_candidate=("One More Time", "Daft Punk")
    )
    normal = _resolver(router).resolve(hints)
    assert len(normal.queries_tried) == 1
    router.log.clear()
    resolver = TrackResolver(make_mb(router), make_itunes(router), record_all=True)
    result = resolver.resolve(hints)
    assert [q["provider"] for q in result.queries_tried] == ["musicbrainz", "musicbrainz", "itunes"]
    paths = {r.url.path for r in router.log}
    assert "/ws/2/release/rel-disc" in paths
    assert "/ws/2/release/rel-single" in paths
    assert result.decision == MatchDecision.AUTO_ACCEPT


def test_album_override_and_topic_album_stay_active_with_downloader_signals() -> None:
    router = Router()
    res = _resolver(router, with_itunes=False)
    url_hints = ParsedHints(
        artist="Pink Floyd",
        title="Comfortably Numb",
        downloader_signals=["url"],
    )
    q_override = res._base_query(url_hints, 383.0, ManualOverrides(album="Pulse"))
    assert q_override.album_hint == "Pulse"

    topic_hints = ParsedHints(
        artist="Fleetwood Mac",
        title="Go Your Own Way",
        album="Rumours",
        downloader_signals=["topic_suffix"],
        trusted_artist=True,
    )
    assert res._base_query(topic_hints, 218.0).album_hint == "Rumours"


def test_batch_coherence_respects_majority_and_explicit_remix() -> None:
    router = Router()
    router.json(
        query_has('recording:"One More Time"'),
        search_payload(
            recording(
                "rec-omt",
                "One More Time",
                DP,
                [release("rel-disc", "Discovery", "rg-disc", track_number=1)],
            )
        ),
    )
    router.json(
        query_has('recording:"Aerodynamic (Daft Punk Remix)"'),
        search_payload(
            recording(
                "rec-aero-rmx",
                "Aerodynamic (Daft Punk Remix)",
                DP,
                [
                    release(
                        "rel-club",
                        "Daft Club",
                        "rg-club",
                        secondary=["Remix"],
                        date="2003-12-01",
                    )
                ],
                length_ms=300_000,
            ),
            recording(
                "rec-aero-std",
                "Aerodynamic",
                DP,
                [release("rel-disc", "Discovery", "rg-disc", date="2001-03-12")],
                length_ms=212_000,
            ),
        ),
    )
    router.json(path_is("/ws/2/release/rel-disc"), _disc_lookup())
    router.json(
        path_is("/ws/2/release/rel-club"),
        release_lookup(
            "rel-club",
            "Daft Club",
            artist_credits=DP,
            rg_id="rg-club",
            rg_first_date="2003-12-01",
            date="2003-12-01",
            media=[medium(1, [("rec-aero-rmx", "Aerodynamic (Daft Punk Remix)")])],
            genres=[{"name": "electronic", "count": 1}],
        ),
    )
    items = [
        BatchItem(
            ParsedHints(artist="Daft Punk", title="One More Time", parent_album_hint="Favorites"),
            duration_seconds=320.0,
        ),
        BatchItem(
            ParsedHints(artist="Daft Punk", title="One More Time", parent_album_hint="Favorites"),
            duration_seconds=320.0,
        ),
        BatchItem(
            ParsedHints(
                artist="Daft Punk",
                title="Aerodynamic (Daft Punk Remix)",
                edition_flags=["remix"],
                parent_album_hint="Favorites",
            ),
            duration_seconds=300.0,
        ),
    ]
    results = _resolver(router, with_itunes=False).resolve_batch(items)
    assert results[2].metadata.album == "Daft Club"


def test_enrichment_strips_doubled_feat_and_resolves_nonlatin_featuring_before_genre() -> None:
    cd = credit(("Celine Dion", " feat. ", "a-cd"), ("張學友", "", "a-jc"))
    search = search_payload(
        recording(
            "rec-duet",
            "Duet Song [feat. 張學友]",
            cd,
            [release("rel-duet", "Album", "rg-duet")],
        )
    )
    router = Router()
    router.json(query_has("Duet Song"), search)
    router.json(
        path_is("/ws/2/release/rel-duet"),
        release_lookup(
            "rel-duet",
            "Album",
            artist_credits=credit(("Various Artists", "", "a-va")),
            rg_id="rg-duet",
            rg_first_date="2000-01-01",
            date="2000-01-01",
            media=[medium(1, [("rec-duet", "Duet Song [feat. 張學友]")])],
        ),
    )
    router.json(
        path_is("/ws/2/artist/a-jc"),
        {
            "id": "a-jc",
            "name": "張學友",
            "aliases": [{"name": "Jacky Cheung", "locale": "en", "primary": True}],
        },
    )
    router.json(
        host_is("itunes.apple.com"),
        {
            "results": [
                {
                    "kind": "song",
                    "trackName": "Duet Song",
                    "artistName": "Celine Dion",
                    "collectionName": "Album",
                    "primaryGenreName": "Pop",
                }
            ]
        },
    )
    md = _resolver(router).resolve(ParsedHints(artist="Celine Dion", title="Duet Song")).metadata
    assert md.title == "Duet Song"
    assert md.featuring == ["Jacky Cheung {張學友}"]
    assert md.genre == "Pop"
    itunes_req = next(r for r in router.log if r.url.host == "itunes.apple.com")
    assert "Various Artists" not in itunes_req.url.params["term"]


def test_album_artist_override_preserves_track_artist_and_swapped_retry() -> None:
    from pathlib import Path

    from audio_from_video.models import ProbeResult
    from audio_from_video.parser import parse_hints

    tff = credit(("Tears for Fears", "", "a-tff"))
    search = search_payload(
        recording(
            "rec-shout",
            "Shout",
            tff,
            [release("rel-sftbc", "Songs From the Big Chair", "rg-sftbc", track_number=1)],
        )
    )
    lookup = release_lookup(
        "rel-sftbc",
        "Songs From the Big Chair",
        artist_credits=tff,
        rg_id="rg-sftbc",
        rg_first_date="1985-02-25",
        date="1985-02-25",
        media=[medium(1, [("rec-shout", "Shout")])],
        genres=[{"name": "synth-pop", "count": 5}],
    )
    router = Router()
    router.json(query_has('recording:"Shout" AND artist:"Tears for Fears"'), search)
    router.json(path_is("/ws/2/release/rel-sftbc"), lookup)
    ov = ManualOverrides(album_artist="Various Artists")
    hints = parse_hints(
        ProbeResult(
            path=Path("/Music/NOW 80s/Tears for Fears - Shout.mp4"),
            duration_seconds=393.0,
            format_name="mp4",
        )
    )
    res = _resolver(router, with_itunes=False).resolve(hints, overrides=ov)
    assert res.decision == MatchDecision.AUTO_ACCEPT
    assert res.metadata.artist == "Tears for Fears"
    assert res.metadata.album_artist == "Various Artists"
    assert all("Various Artists" not in r.url.params.get("query", "") for r in router.log)

    # Unmatched track still preserves parsed artist and sets album_artist (matches offline).
    empty_router = Router()
    unmatched = _resolver(empty_router, with_itunes=False).resolve(hints, overrides=ov)
    assert unmatched.decision == MatchDecision.UNMATCHED
    assert unmatched.metadata.artist == "Tears for Fears"
    assert unmatched.metadata.album_artist == "Various Artists"

    # Swapped retry still runs when only --album-artist is given.
    swapped_hints = ParsedHints(
        artist="Shout",
        title="Tears for Fears",
        swapped_candidate=("Tears for Fears", "Shout"),
        clean_stem="Shout - Tears for Fears",
    )
    swapped_res = _resolver(router, with_itunes=False).resolve(swapped_hints, overrides=ov)
    assert swapped_res.decision == MatchDecision.AUTO_ACCEPT
    assert swapped_res.metadata.artist == "Tears for Fears"
    assert swapped_res.metadata.album_artist == "Various Artists"

    # When no parsed artist exists, a non-VA --album-artist assists search, while VA does not.
    title_only = ParsedHints(title="Shout", clean_stem="Shout")
    r_artist = Router().json(query_has('recording:"Shout" AND artist:"Tears for Fears"'), search)
    r_artist.json(path_is("/ws/2/release/rel-sftbc"), lookup)
    assisted = _resolver(r_artist, with_itunes=False).resolve(
        title_only, overrides=ManualOverrides(album_artist="Tears for Fears")
    )
    assert assisted.decision == MatchDecision.AUTO_ACCEPT
    r_va = Router()
    _resolver(r_va, with_itunes=False).resolve(title_only, overrides=ov)
    assert all("Various Artists" not in r.url.params.get("query", "") for r in r_va.log)


def test_edition_recheck_handles_five_unknown_disc_releases() -> None:
    artist = credit(("Artist", "", "artist-x"))
    search = search_payload(
        recording(
            "rec-s",
            "Song",
            artist,
            [
                release("rel-0", "Album", "rg-a", date="2001-01-01", track_number=3),
                release("rel-1", "Album", "rg-a", date="2001-02-01", track_number=3),
                release("rel-2", "Album", "rg-a", date="2001-03-01", track_number=3),
                release("rel-3", "Album", "rg-a", date="2001-04-01", track_number=3),
                release("rel-4", "Album", "rg-a", date="2001-05-01", track_number=3),
            ],
        )
    )

    def lookup(rid: str, discs: int) -> dict:
        media = [medium(1, [("rec-a", "A"), ("rec-b", "B"), ("rec-s", "Song")])]
        media += [medium(i, [(f"rec-x{i}", "X")]) for i in range(2, discs + 1)]
        return release_lookup(
            rid,
            "Album",
            artist_credits=artist,
            rg_id="rg-a",
            rg_first_date="2001-01-01",
            date="2001-01-01",
            media=media,
            genres=[{"name": "rock", "count": 1}],
        )

    router = Router().json(query_has("Song"), search)
    router.json(path_is("/ws/2/release/rel-0"), lookup("rel-0", 2))
    router.json(path_is("/ws/2/release/rel-1"), lookup("rel-1", 1))
    router.json(path_is("/ws/2/release/rel-2"), lookup("rel-2", 2))
    router.json(path_is("/ws/2/release/rel-3"), lookup("rel-3", 2))
    md = (
        _resolver(router, with_itunes=False)
        .resolve(ParsedHints(artist="Artist", title="Song"))
        .metadata
    )
    assert md.mb_release_id == "rel-1"
    assert md.disc_total == 1


def test_lookup_circuit_breaker_skips_unreachable_provider_across_batch() -> None:
    router = Router().add(host_is("musicbrainz.org"), lambda r: httpx.Response(503))
    router.add(host_is("itunes.apple.com"), lambda r: httpx.Response(503))
    mb = MusicBrainzClient(
        TEST_UA, fetcher=fetcher(router, max_retries=0), art_fetcher=fetcher(router)
    )
    from audio_from_video.providers.itunes import ITunesClient

    itunes = ITunesClient(fetcher=fetcher(router, max_retries=0))
    resolver = TrackResolver(mb, itunes, circuit_breaker_threshold=2)
    items = [BatchItem(ParsedHints(artist="Artist", title=f"Song {i}")) for i in range(5)]
    results = resolver.resolve_batch(items)
    assert all(r.decision == MatchDecision.UNMATCHED for r in results)
    # Each provider fails twice (tripping its breaker) and skips tracks 3..5 completely.
    assert router.count("musicbrainz.org") == 2
    assert router.count("itunes.apple.com") == 2
    all_warnings = [w for r in results for w in r.warnings]
    assert any("MusicBrainz appears unreachable" in w for w in all_warnings)
    assert any("iTunes appears unreachable" in w for w in all_warnings)
    assert any("Skipped" in w for w in results[-1].warnings)

    # A single failure followed by a success resets the breaker counter and does not trip.
    calls = {"n": 0}
    artist_c = credit(("Artist", "", "a-art"))

    def flaky(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json=search_payload(
                recording(
                    f"rec-{calls['n']}",
                    "Track",
                    artist_c,
                    [release("rel-ok", "Album", "rg-ok")],
                )
            ),
        )

    flaky_router = Router().add(host_is("musicbrainz.org"), flaky)
    mb_flaky = MusicBrainzClient(
        TEST_UA, fetcher=fetcher(flaky_router, max_retries=0), art_fetcher=fetcher(flaky_router)
    )
    flaky_resolver = TrackResolver(mb_flaky, None, circuit_breaker_threshold=2)
    for i in range(4):
        flaky_resolver.resolve(ParsedHints(artist="Artist", title=f"Track {i}"))
    assert calls["n"] >= 4
    assert "musicbrainz" not in flaky_resolver._tripped_providers


def test_enrichment_preserves_full_recording_featuring_over_abbreviated_track_credit() -> None:
    full_credit = credit(
        ("Daft Punk", " feat. ", "a-dp"),
        ("Pharrell Williams", " & ", "a-pw"),
        ("Nile Rodgers", "", "a-nr"),
    )
    short_credit = credit(("Daft Punk", " feat. ", "a-dp"), ("Pharrell Williams", "", "a-pw"))
    search = search_payload(
        recording(
            "rec-gl",
            "Get Lucky",
            full_credit,
            [release("rel-ram", "Random Access Memories", "rg-ram", track_number=8)],
        )
    )
    rel_data = release_lookup(
        "rel-ram",
        "Random Access Memories",
        artist_credits=DP,
        rg_id="rg-ram",
        rg_first_date="2013-05-17",
        date="2013-05-17",
        media=[
            {
                "position": 1,
                "track-count": 13,
                "tracks": [
                    {
                        "position": 8,
                        "number": "8",
                        "title": "Get Lucky",
                        "artist-credit": short_credit,
                        "recording": {
                            "id": "rec-gl",
                            "title": "Get Lucky",
                            "artist-credit": full_credit,
                        },
                    }
                ],
            }
        ],
        genres=[{"name": "electro", "count": 5}],
    )
    router = (
        Router()
        .json(query_has("Get Lucky"), search)
        .json(path_is("/ws/2/release/rel-ram"), rel_data)
    )
    res = _resolver(router, with_itunes=False).resolve(
        ParsedHints(artist="Daft Punk", title="Get Lucky")
    )
    assert res.decision == MatchDecision.AUTO_ACCEPT
    assert res.metadata.featuring == ["Pharrell Williams", "Nile Rodgers"]


def test_enrichment_does_not_duplicate_featuring_when_track_joins_with_ampersand() -> None:
    rec_credit = credit(("Estelle", " feat. ", "a-est"), ("Kanye West", "", "a-kw"))
    track_credit = credit(("Estelle", " & ", "a-est"), ("Kanye West", "", "a-kw"))
    search = search_payload(
        recording(
            "rec-ab",
            "American Boy",
            rec_credit,
            [release("rel-shine", "Shine", "rg-shine", track_number=3)],
        )
    )
    rel_data = release_lookup(
        "rel-shine",
        "Shine",
        artist_credits=credit(("Estelle", "", "a-est")),
        rg_id="rg-shine",
        rg_first_date="2008-03-28",
        date="2008-03-28",
        media=[
            {
                "position": 1,
                "track-count": 12,
                "tracks": [
                    {
                        "position": 3,
                        "number": "3",
                        "title": "American Boy",
                        "artist-credit": track_credit,
                        "recording": {
                            "id": "rec-ab",
                            "title": "American Boy",
                            "artist-credit": rec_credit,
                        },
                    }
                ],
            }
        ],
        genres=[{"name": "r&b", "count": 3}],
    )
    router = (
        Router()
        .json(query_has("American Boy"), search)
        .json(path_is("/ws/2/release/rel-shine"), rel_data)
    )
    res = _resolver(router, with_itunes=False).resolve(
        ParsedHints(artist="Estelle", title="American Boy", featuring=["Kanye West"])
    )
    assert res.decision == MatchDecision.AUTO_ACCEPT
    assert res.metadata.artist == "Estelle"
    assert res.metadata.featuring == ["Kanye West"]


def test_recheck_edition_preserves_disc_hint_and_keeps_best_in_sync() -> None:
    pf = credit(("Pink Floyd", "", "a-pf"))
    rel_2cd = release(
        "rel-2cd",
        "The Wall",
        "rg-wall",
        date="1979-11-30",
        disc=2,
        track_number=6,
        medium_track_count=13,
    )
    rel_1cd = release(
        "rel-digital",
        "The Wall",
        "rg-wall",
        date="2011-09-26",
        disc=1,
        track_number=19,
        medium_track_count=26,
    )
    search = search_payload(
        recording("rec-cn", "Comfortably Numb", pf, [rel_2cd, rel_1cd], length_ms=383_000)
    )
    lookup_2cd = release_lookup(
        "rel-2cd",
        "The Wall",
        artist_credits=pf,
        rg_id="rg-wall",
        rg_first_date="1979-11-30",
        date="1979-11-30",
        media=[
            medium(1, [(f"rec-d1-{i}", f"T{i}") for i in range(1, 14)]),
            medium(
                2,
                [(f"rec-d2-{i}", f"D2-{i}") for i in range(1, 6)]
                + [("rec-cn", "Comfortably Numb")]
                + [(f"rec-d2-{i}", f"D2-{i}") for i in range(7, 14)],
            ),
        ],
        genres=[{"name": "progressive rock", "count": 10}],
    )
    lookup_1cd = release_lookup(
        "rel-digital",
        "The Wall",
        artist_credits=pf,
        rg_id="rg-wall",
        rg_first_date="1979-11-30",
        date="2011-09-26",
        media=[
            medium(
                1,
                [(f"rec-x-{i}", f"X{i}") for i in range(1, 19)] + [("rec-cn", "Comfortably Numb")],
            )
        ],
        genres=[{"name": "progressive rock", "count": 10}],
    )
    router = (
        Router()
        .json(query_has("Comfortably Numb"), search)
        .json(path_is("/ws/2/release/rel-2cd"), lookup_2cd)
        .json(path_is("/ws/2/release/rel-digital"), lookup_1cd)
    )
    # With disc_hint=2, recheck must keep the 2-CD edition (disc 2 track 6), not switch to 1-disc.
    res_hint = _resolver(router, with_itunes=False).resolve(
        ParsedHints(artist="Pink Floyd", title="Comfortably Numb", disc_number=2, track_number=6)
    )
    assert res_hint.decision == MatchDecision.AUTO_ACCEPT
    assert res_hint.metadata.mb_release_id == "rel-2cd"
    assert res_hint.best is not None and res_hint.best.release_id == "rel-2cd"
    assert (res_hint.metadata.disc_number, res_hint.metadata.track_number) == (2, 6)

    # Without disc_hint, recheck switches to the 1-disc edition AND updates result.best in sync.
    res_no_hint = _resolver(router, with_itunes=False).resolve(
        ParsedHints(artist="Pink Floyd", title="Comfortably Numb")
    )
    assert res_no_hint.decision == MatchDecision.AUTO_ACCEPT
    assert res_no_hint.metadata.mb_release_id == "rel-digital"
    assert res_no_hint.best is not None and res_no_hint.best.release_id == "rel-digital"


def test_enrichment_prefers_release_group_genre_votes_over_sparse_release_ties() -> None:
    router = Router().json(query_has("One More Time"), _omt_search())
    lookup = _disc_lookup(
        genres=[{"name": "ambient", "count": 1}, {"name": "british blues", "count": 1}],
        rg_genres=[{"name": "french house", "count": 23}, {"name": "pop", "count": 10}],
    )
    router.json(path_is("/ws/2/release/rel-disc"), lookup)
    res = _resolver(router, with_itunes=False).resolve(
        ParsedHints(artist="Daft Punk", title="One More Time")
    )
    assert res.metadata.genre == "French House"


def test_warning_scrubbing_removes_domain_fragments() -> None:
    router = Router().add(lambda r: True, lambda r: httpx.Response(400))
    mb = MusicBrainzClient(
        TEST_UA, fetcher=fetcher(router, max_retries=0), art_fetcher=fetcher(router)
    )
    res = TrackResolver(mb, None).resolve(ParsedHints(artist="A", title="B"))
    assert res.warnings
    assert res.warnings[0] == "MusicBrainz search failed: HTTP 400."
    assert ".org" not in res.warnings[0] and "https://" not in res.warnings[0]


def test_cross_attempt_margin_prevents_alias_attempt_overriding_primary_artist() -> None:
    kino = credit(("Кино", "", "a-kino"))
    cover = credit(("Симфоническое Кино", "", "a-symph"))
    router = Router()
    router.json(
        query_has('artist:"Кино"'),
        search_payload(
            recording(
                "rec-orig",
                "Группа крови",
                kino,
                [release("rel-orig", "Группа крови", "rg-orig", date="1988-01-01")],
                length_ms=285_000,
            )
        ),
    )
    router.json(
        path_is("/ws/2/artist/"),
        {
            "artists": [
                {"id": "a-symph", "name": "Симфоническое Кино", "aliases": []},
                {"id": "a-kino", "name": "Кино", "aliases": [{"name": "Kino"}]},
            ]
        },
    )
    router.json(
        query_has('artist:"Симфоническое Кино"'),
        search_payload(
            recording(
                "rec-cov",
                "Группа крови",
                cover,
                [release("rel-cov", "СимфоКино", "rg-cov", date="2012-01-01")],
                length_ms=240_000,
            )
        ),
    )
    router.json(
        path_is("/ws/2/release/rel-orig"),
        release_lookup(
            "rel-orig",
            "Группа крови",
            artist_credits=kino,
            rg_id="rg-orig",
            rg_first_date="1988-01-01",
            date="1988-01-01",
            media=[medium(1, [("rec-orig", "Группа крови")])],
            genres=[{"name": "post-punk", "count": 5}],
        ),
    )
    # Even with a short duration that triggers the cross-script alias lookup, the unrelated
    # cover act "Симфоническое Кино" is never treated as an alias of "Кино".
    res = _resolver(router, with_itunes=False).resolve(
        ParsedHints(artist="Кино", title="Группа крови"),
        duration_seconds=120.0,
    )
    assert res.best is not None and res.best.artist == "Кино"
    assert all("Симфоническое" not in r.url.params.get("query", "") for r in router.log)
