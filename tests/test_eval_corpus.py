"""Tests for eval seed labels, record and replay transports, and the gate metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from audio_from_video.config import MatchingConfig
from audio_from_video.eval_corpus import (
    CATEGORIES,
    RecordingTransport,
    ReplayMissError,
    ReplayTransport,
    SeedError,
    build_frozen_corpus,
    check_disjoint,
    evaluate,
    file_sha256,
    identity_matches,
    load_corpus,
    load_seeds,
    parse_seed_hints,
    seed_to_probe,
    validate_seed,
    write_corpus,
)
from audio_from_video.lookup import TrackResolver, default_group_key
from audio_from_video.parser import parse_hints
from audio_from_video.providers.base import HttpFetcher, ProviderError, RateLimiter
from audio_from_video.providers.itunes import ITunesClient
from audio_from_video.providers.musicbrainz import MusicBrainzClient
from provider_fixtures import (
    Router,
    credit,
    medium,
    path_is,
    query_has,
    recording,
    release,
    release_lookup,
    search_payload,
)

FIXTURES = Path(__file__).parent / "fixtures"
TUNING_SEEDS = FIXTURES / "eval_seed_tuning.json"
HOLDOUT_SEEDS = FIXTURES / "eval_seed_holdout.json"
REPLAY_UA = "audio-from-video/0.1.0 ( https://github.com/org/audio-from-video )"


@pytest.fixture(scope="module")
def tuning() -> list[dict[str, Any]]:
    return load_seeds(TUNING_SEEDS)


@pytest.fixture(scope="module")
def holdout() -> list[dict[str, Any]]:
    return load_seeds(HOLDOUT_SEEDS)


# -- Seed files ----------------------------------------------------------------------


def test_seed_counts(tuning: list, holdout: list) -> None:
    assert len(tuning) >= 30
    assert len(holdout) >= 50
    assert len(tuning) + len(holdout) >= 80


def test_seed_splits_are_disjoint(tuning: list, holdout: list) -> None:
    assert check_disjoint(tuning, holdout) == []


def test_check_disjoint_rejects_shared_artist_across_different_songs() -> None:
    a = [
        _seed(
            "t1",
            "clear",
            "Daft Punk - One More Time.mp4",
            {"outcome": "auto_accept", "artist": "Daft Punk", "title": "One More Time"},
        )
    ]
    b = [
        _seed(
            "h1",
            "clear",
            "Daft Punk - Digital Love.mp4",
            {"outcome": "auto_accept", "artist": "Daft Punk", "title": "Digital Love"},
        )
    ]
    problems = check_disjoint(a, b)
    assert any("shared artist 'daft punk'" in p for p in problems)


def test_check_disjoint_rejects_shared_title_only_query() -> None:
    a = [
        _seed(
            "t1",
            "ambiguous",
            "Creep.mp4",
            {"outcome": "top3", "artist": "Radiohead", "title": "Creep"},
        )
    ]
    b = [
        _seed(
            "h1",
            "ambiguous",
            "Creep (Official Audio).mp4",
            {"outcome": "top3", "artist": "TLC", "title": "Creep"},
        )
    ]
    problems = check_disjoint(a, b)
    assert any("shared query (None, 'creep')" in p for p in problems)


def test_every_category_is_covered_in_both_splits(tuning: list, holdout: list) -> None:
    assert {s["category"] for s in tuning} == CATEGORIES
    assert {s["category"] for s in holdout} == CATEGORIES


def test_all_seeds_have_realistic_durations_and_hard_adversarials(
    tuning: list, holdout: list
) -> None:
    for split in (tuning, holdout):
        parsed = parse_seed_hints(split)
        for seed, hints in zip(split, parsed, strict=True):
            dur = seed["input"]["duration_seconds"]
            assert isinstance(dur, (int, float)) and dur > 0, seed["id"]
            if seed["category"] == "adversarial":
                assert hints.artist, f"adversarial seed {seed['id']} needs an artist query"


def test_both_splits_cover_batch_live_remix_and_swapped_filenames(
    tuning: list, holdout: list
) -> None:
    for label, split in (("tuning", tuning), ("holdout", holdout)):
        parsed = parse_seed_hints(split)
        groups: dict[object, int] = {}
        has_live = False
        has_remix = False
        has_swapped_filename = False
        for seed, hints in zip(split, parsed, strict=True):
            key = default_group_key(hints)
            if key is not None:
                groups[key] = groups.get(key, 0) + 1
            if "live" in hints.edition_flags:
                has_live = True
            if "remix" in hints.edition_flags:
                has_remix = True
                exp = seed["expected"]
                assert exp.get("title_contains") or exp.get("album") or exp.get("album_any_of"), (
                    f"remix seed {seed['id']} must assert title_contains or album"
                )
            if (
                not seed["input"]["tags"]
                and seed["expected"]["outcome"] == "auto_accept"
                and hints.swapped_candidate is not None
                and hints.artist is not None
                and hints.artist.casefold() != (seed["expected"].get("artist") or "").casefold()
            ):
                has_swapped_filename = True
        assert any(count >= 2 for count in groups.values()), f"{label} missing folder batch"
        assert has_live, f"{label} missing live edition seed"
        assert has_remix, f"{label} missing remix edition seed"
        assert has_swapped_filename, f"{label} missing swapped Title - Artist filename seed"


def test_required_holdout_positive_controls_present(holdout: list) -> None:
    tags = [s["input"]["tags"] for s in holdout if s["category"] == "positive_control"]
    pairs = {(t.get("artist"), t.get("title")) for t in tags}
    assert ("Roxy Music", "More Than This - 2012 Remaster") in pairs
    assert ("Berliner Philharmoniker", "Symphony No. 5 - Allegro con brio") in pairs
    artists = {a for a, _ in pairs}
    assert {"Channel Tres", "Official髭男dism", "TV on the Radio"} <= artists


def test_seeds_contain_no_contacts_or_local_paths(tuning: list, holdout: list) -> None:
    text = TUNING_SEEDS.read_text(encoding="utf-8") + HOLDOUT_SEEDS.read_text(encoding="utf-8")
    assert "@" not in text
    assert "/Users/" not in text and "\\Users\\" not in text
    assert "example." not in text


def test_parser_matches_downloader_and_trust_labels(tuning: list, holdout: list) -> None:
    for split in (tuning, holdout):
        for seed, hints in zip(split, parse_seed_hints(split), strict=True):
            exp = seed["expected"]
            if "downloader_signals" in exp:
                assert sorted(hints.downloader_signals) == sorted(exp["downloader_signals"]), seed[
                    "id"
                ]
            if "trusted_artist" in exp:
                assert hints.trusted_artist == exp["trusted_artist"], seed["id"]
            if exp["outcome"] == "skip_lookup":
                assert hints.is_hintless, seed["id"]


def test_seed_validation_rejects_bad_schema() -> None:
    good = {
        "id": "x",
        "category": "clear",
        "input": {"filename": "A - B.mp4", "parent_dirs": [], "tags": {}, "duration_seconds": None},
        "expected": {"outcome": "auto_accept", "artist": "A", "title": "B"},
        "notes": "",
    }
    validate_seed(good)
    for mutate in (
        lambda s: s.update(category="nope"),
        lambda s: s["expected"].update(outcome="maybe"),
        lambda s: s["expected"].pop("artist"),
        lambda s: s["expected"].update(track_number=0),
        lambda s: s["input"].update(tags={"artist": 5}),
        lambda s: s["input"].update(filename="bad/name.mp4"),
        lambda s: s["expected"].update(title_contains="  "),
        lambda s: s["expected"].update(title_contains=42),
        lambda s: s.update(category="downloader"),
        lambda s: s.update(category="hintless"),
    ):
        bad = json.loads(json.dumps(good))
        mutate(bad)
        with pytest.raises(SeedError):
            validate_seed(bad)


def test_seed_to_probe_uses_stop_listed_roots() -> None:
    loose = {"input": {"filename": "Daft Punk - One More Time.mp4", "parent_dirs": []}}
    hints = parse_hints(seed_to_probe(loose))
    assert hints.parent_album_hint is None and hints.parent_artist_hint is None
    shaped = {
        "input": {"filename": "06 Karma Police.mp4", "parent_dirs": ["Radiohead", "OK Computer"]}
    }
    probe = seed_to_probe(shaped)
    assert probe.path.as_posix().endswith("Music/Radiohead/OK Computer/06 Karma Police.mp4")
    hints = parse_hints(probe)
    assert (hints.parent_artist_hint, hints.parent_album_hint) == ("Radiohead", "OK Computer")


def test_identity_matching_rules() -> None:
    exp = {
        "artist_any_of": ["Jacky Cheung {張學友}", "張學友"],
        "title": "吻別",
        "album": "1989",
        "track_number": 6,
        "featuring": ["Jay-Z"],
    }
    assert identity_matches(exp, artist="Jacky Cheung {張學友}", title="The Goodbye Kiss {吻別}")
    assert identity_matches(exp, artist="張學友", title="吻別")
    assert not identity_matches(exp, artist="王菲", title="吻別")
    assert identity_matches(
        exp,
        artist="張學友",
        title="吻別",
        album="1989",
        track_number=6,
        featuring=["JAY\u2010Z"],
        check_details=True,
    )
    assert not identity_matches(
        exp,
        artist="張學友",
        title="吻別",
        album="1989 (Taylor's Version)",
        track_number=6,
        featuring=["Jay-Z"],
        check_details=True,
    )
    assert not identity_matches(
        exp,
        artist="張學友",
        title="吻別",
        album="1989",
        track_number=7,
        featuring=["Jay-Z"],
        check_details=True,
    )
    assert not identity_matches(
        exp,
        artist="張學友",
        title="吻別",
        album="1989",
        track_number=6,
        featuring=[],
        check_details=True,
    )
    remix_exp = {
        "artist": "Mr. Probz",
        "title": "Waves",
        "title_contains": "robin schulz",
    }
    assert identity_matches(
        remix_exp,
        artist="Mr. Probz",
        title="Waves (Robin Schulz Radio Edit)",
        check_details=True,
    )
    assert not identity_matches(
        remix_exp,
        artist="Mr. Probz",
        title="Waves",
        check_details=True,
    )


# -- Transports ------------------------------------------------------------------------


def test_recording_transport_stores_status_and_json_but_never_headers() -> None:
    router = Router().json(path_is("/ws/2/release/r1"), {"id": "r1"})
    recorder = RecordingTransport(router.transport)
    client = httpx.Client(transport=recorder)
    resp = client.get(
        "https://musicbrainz.org/ws/2/release/r1?inc=media&fmt=json",
        headers={"User-Agent": REPLAY_UA},
    )
    assert resp.json() == {"id": "r1"}
    client.get("https://musicbrainz.org/ws/2/release/missing")
    key = "https://musicbrainz.org/ws/2/release/r1?fmt=json&inc=media"
    assert recorder.responses[key] == {"status": 200, "json": {"id": "r1"}}
    assert recorder.responses["https://musicbrainz.org/ws/2/release/missing"]["status"] == 404
    assert "github.com" not in json.dumps(recorder.responses)


def test_replay_transport_serves_and_raises_clear_miss() -> None:
    replay = ReplayTransport(
        {
            "https://h/p?a=1&b=2": {"status": 200, "json": {"ok": 1}},
            "https://h/gone": {"status": 404, "json": None},
        }
    )
    client = httpx.Client(transport=replay)
    assert client.get("https://h/p?b=2&a=1").json() == {"ok": 1}
    assert client.get("https://h/gone").status_code == 404
    with pytest.raises(ReplayMissError, match="No recorded response"):
        client.get("https://h/unknown")


# -- Metrics on a tiny synthetic replay corpus --------------------------------------------


def _seed(sid: str, cat: str, filename: str, expected: dict, **inp: Any) -> dict:
    return {
        "id": sid,
        "category": cat,
        "input": {
            "filename": filename,
            "parent_dirs": inp.get("dirs", []),
            "tags": inp.get("tags", {}),
            "duration_seconds": None,
        },
        "expected": expected,
        "notes": "synthetic",
    }


SYNTH_SEEDS = [
    _seed(
        "s1",
        "clear",
        "Daft Punk - One More Time.mp4",
        {
            "outcome": "auto_accept",
            "artist": "Daft Punk",
            "title": "One More Time",
            "album": "Discovery",
            "track_number": 1,
        },
    ),
    _seed("s2", "hintless", "VID_0042.mp4", {"outcome": "skip_lookup"}),
    _seed(
        "s3",
        "adversarial",
        "Daft Punk - Digital Love (Karaoke Version).mp4",
        {"outcome": "no_auto_accept"},
    ),
    _seed(
        "s4",
        "ambiguous",
        "Cash - Hurt.mp4",
        {"outcome": "top3", "artist": "Johnny Cash", "title": "Hurt"},
    ),
    _seed(
        "s5",
        "positive_control",
        "x.mp4",
        {"outcome": "parse_only", "downloader_signals": [], "trusted_artist": True},
        tags={"artist": "Roxy Music", "title": "More Than This - 2012 Remaster"},
    ),
    _seed(
        "s6",
        "downloader",
        "Go Your Own Way.webm",
        {"outcome": "parse_only", "downloader_signals": ["topic_suffix"], "trusted_artist": True},
        tags={"artist": "Fleetwood Mac - Topic", "title": "Go Your Own Way"},
    ),
]


def _synthetic_router() -> Router:
    dp = credit(("Daft Punk", "", "artist-dp"))
    router = Router()
    router.json(
        query_has('recording:"One More Time"'),
        search_payload(
            recording(
                "rec-omt",
                "One More Time",
                dp,
                [release("rel-disc", "Discovery", "rg-disc", track_number=1)],
            )
        ),
    )
    router.json(
        query_has('recording:"Digital Love"'),
        search_payload(
            recording(
                "rec-dl",
                "Digital Love",
                dp,
                [release("rel-disc", "Discovery", "rg-disc", track_number=3)],
            )
        ),
    )
    router.json(
        query_has('recording:"Hurt"'),
        search_payload(
            recording(
                "r1", "Hurt", credit(("Johnny Cash", "")), [release("x1", "American IV", "rg1")]
            ),
            recording("r2", "Hurt", credit(("Rosanne Cash", "")), [release("x2", "Other", "rg2")]),
        ),
    )
    router.json(
        path_is("/ws/2/release/rel-disc"),
        release_lookup(
            "rel-disc",
            "Discovery",
            artist_credits=dp,
            rg_id="rg-disc",
            rg_first_date="2001-03-07",
            date="2001-03-12",
            media=[
                medium(
                    1,
                    [
                        ("rec-omt", "One More Time"),
                        ("rec-aero", "Aerodynamic"),
                        ("rec-dl", "Digital Love"),
                    ],
                )
            ],
            genres=[{"name": "house", "count": 1}],
        ),
    )
    return router


def _record(seeds: list[dict]) -> dict[str, Any]:
    recorder = RecordingTransport(_synthetic_router().transport)

    def f() -> HttpFetcher:
        return HttpFetcher(transport=recorder, limiter=RateLimiter(0.0), sleep=lambda s: None)

    mb = MusicBrainzClient(REPLAY_UA, fetcher=f(), art_fetcher=f())
    itunes = ITunesClient(fetcher=f(), art_fetcher=f())
    resolver = TrackResolver(mb, itunes, record_all=True)
    for seed in seeds:
        if seed["expected"]["outcome"] != "parse_only":
            resolver.resolve(parse_hints(seed_to_probe(seed)))
    return recorder.responses


def test_evaluate_metrics_on_synthetic_replay_corpus() -> None:
    responses = _record(SYNTH_SEEDS)
    assert all("headers" not in v for v in responses.values())
    metrics = evaluate(SYNTH_SEEDS, responses)
    by_id = {o.seed_id: o for o in metrics.outcomes}
    assert by_id["s1"].decision == "auto_accept" and by_id["s1"].correct
    assert by_id["s2"].requests == 0 and by_id["s2"].decision == "skipped"
    assert by_id["s3"].decision != "auto_accept"
    assert by_id["s4"].in_top3
    assert metrics.auto_accept_precision == 1.0
    assert metrics.adversarial_false_accepts == 0
    assert metrics.auto_accept_recall == 1.0
    assert metrics.top3_recall == 1.0
    assert metrics.safety == 1.0


def test_evaluate_runs_batch_coherence_for_folder_siblings() -> None:
    dp = credit(("Daft Punk", "", "artist-dp"))
    router = Router()
    router.json(
        query_has('recording:"One More Time"'),
        search_payload(
            recording(
                "rec-omt",
                "One More Time",
                dp,
                [release("rel-disc", "Discovery", "rg-disc", date="2001-03-12", track_number=1)],
            )
        ),
    )
    router.json(
        query_has('recording:"Aerodynamic"'),
        search_payload(
            recording(
                "rec-aero",
                "Aerodynamic",
                dp,
                [release("rel-disc", "Discovery", "rg-disc", date="2001-03-12", track_number=2)],
            )
        ),
    )
    # Track 3 appears on an earlier single ("rg-early") AND on the folder's album ("rg-disc").
    # Without batch coherence, "rg-early" would tie or win; with batch coherence, "rg-disc" wins.
    router.json(
        query_has('recording:"Digital Love"'),
        search_payload(
            recording(
                "rec-dl",
                "Digital Love",
                dp,
                [
                    release(
                        "rel-early",
                        "Earlier LP",
                        "rg-early",
                        primary="Album",
                        date="2000-01-01",
                        track_number=9,
                    ),
                    release(
                        "rel-disc",
                        "Discovery",
                        "rg-disc",
                        primary="Album",
                        date="2001-03-12",
                        track_number=3,
                    ),
                ],
            )
        ),
    )
    router.json(
        path_is("/ws/2/release/rel-disc"),
        release_lookup(
            "rel-disc",
            "Discovery",
            artist_credits=dp,
            rg_id="rg-disc",
            rg_first_date="2001-03-07",
            date="2001-03-12",
            media=[
                medium(
                    1,
                    [
                        ("rec-omt", "One More Time"),
                        ("rec-aero", "Aerodynamic"),
                        ("rec-dl", "Digital Love"),
                    ],
                )
            ],
            genres=[{"name": "house", "count": 1}],
        ),
    )
    # Use a non-matching folder album name so folder_album_boost alone does not pick Discovery;
    # only two-pass batch coherence from siblings 1 and 2 can pull track 3 onto Discovery.
    batch_seeds = [
        _seed(
            "b1",
            "clear",
            "01 One More Time.mp4",
            {
                "outcome": "auto_accept",
                "artist": "Daft Punk",
                "title": "One More Time",
                "album": "Discovery",
                "track_number": 1,
            },
            dirs=["Daft Punk", "French House Rip"],
        ),
        _seed(
            "b2",
            "clear",
            "02 Aerodynamic.mp4",
            {
                "outcome": "auto_accept",
                "artist": "Daft Punk",
                "title": "Aerodynamic",
                "album": "Discovery",
                "track_number": 2,
            },
            dirs=["Daft Punk", "French House Rip"],
        ),
        _seed(
            "b3",
            "clear",
            "03 Digital Love.mp4",
            {
                "outcome": "auto_accept",
                "artist": "Daft Punk",
                "title": "Digital Love",
                "album": "Discovery",
                "track_number": 3,
            },
            dirs=["Daft Punk", "French House Rip"],
        ),
    ]
    recorder = RecordingTransport(router.transport)

    def f() -> HttpFetcher:
        return HttpFetcher(transport=recorder, limiter=RateLimiter(0.0), sleep=lambda s: None)

    mb = MusicBrainzClient(REPLAY_UA, fetcher=f(), art_fetcher=f())
    itunes = ITunesClient(fetcher=f(), art_fetcher=f())
    resolver = TrackResolver(mb, itunes, record_all=True)
    for seed in batch_seeds:
        resolver.resolve(parse_hints(seed_to_probe(seed)))

    # Isolated single-seed evaluation would pick "Earlier LP" (earlier date, 2000 vs 2001).
    single = evaluate([batch_seeds[2]], recorder.responses)
    assert not single.outcomes[0].correct
    # Batch evaluation resolves the 3 folder siblings together and pulls track 3 onto Discovery.
    batched = evaluate(batch_seeds, recorder.responses)
    assert all(o.correct for o in batched.outcomes)
    assert batched.auto_accept_precision == 1.0
    assert batched.auto_accept_recall == 1.0


def test_evaluate_counts_false_accepts_and_parser_failures() -> None:
    seeds = json.loads(json.dumps(SYNTH_SEEDS))
    seeds[0]["expected"]["track_number"] = 2  # Deliberately wrong label.
    seeds[4]["expected"]["trusted_artist"] = False
    responses = _record(SYNTH_SEEDS)
    metrics = evaluate(seeds, responses)
    assert metrics.false_auto_accepts == 1
    assert metrics.auto_accept_precision == 0.0
    assert metrics.auto_accept_recall == 0.0
    assert metrics.safety < 1.0


def test_evaluate_reports_replay_miss_per_seed_without_raising() -> None:
    metrics = evaluate(SYNTH_SEEDS[:2], {})
    by_id = {o.seed_id: o for o in metrics.outcomes}
    assert metrics.replay_miss_seeds == ["s1"]
    assert by_id["s1"].replay_misses
    assert by_id["s1"].replay_misses[0].startswith("https://musicbrainz.org/ws/2/recording/")
    assert not by_id["s1"].correct
    assert by_id["s2"].replay_misses == []
    assert metrics.auto_accept_recall == 0.0


def test_replay_miss_is_contained_by_the_resolver() -> None:
    from audio_from_video.eval_corpus import build_replay_clients
    from audio_from_video.models import MatchDecision, ParsedHints

    assert issubclass(ReplayMissError, ProviderError)
    mb, itunes, transport = build_replay_clients({})
    result = TrackResolver(mb, itunes).resolve(ParsedHints(artist="A", title="B"))
    assert result.decision == MatchDecision.UNMATCHED
    assert result.warnings
    assert transport.misses


def test_record_all_corpus_supports_threshold_sweeps() -> None:
    responses = _record(SYNTH_SEEDS)
    # The recorder enriched release groups and queried iTunes even though the
    # MusicBrainz scores were high, so any threshold replays without misses.
    assert any("itunes.apple.com" in url for url in responses)
    for cfg in (
        MatchingConfig(auto_accept_score=0.99, min_margin=0.3, ambiguous_floor=0.95),
        MatchingConfig(auto_accept_score=0.6, min_margin=0.01, ambiguous_floor=0.2),
    ):
        metrics = evaluate(SYNTH_SEEDS, responses, matching_config=cfg)
        assert metrics.replay_miss_seeds == []


def test_holdout_has_at_least_fifty_auto_accept_seeds(holdout: list) -> None:
    assert sum(1 for s in holdout if s["expected"]["outcome"] == "auto_accept") >= 50


def test_build_frozen_corpus_copies_seeds_verbatim(tmp_path: Path) -> None:
    seed_file = tmp_path / "seeds.json"
    seed_file.write_text(json.dumps(SYNTH_SEEDS), encoding="utf-8")
    corpus = build_frozen_corpus(SYNTH_SEEDS, file_sha256(seed_file), {"b": {}, "a": {}})
    assert corpus["seeds"] == SYNTH_SEEDS
    assert list(corpus["responses"]) == ["a", "b"]
    assert len(corpus["seed_file_sha256"]) == 64
    assert corpus["afv_version"]


# -- Frozen corpus replay (skipped until the user approves labels and records) ------------


def _load_frozen(name: str, seed_path: Path) -> dict[str, Any]:
    path = FIXTURES / name
    if not path.is_file() and name.endswith(".json.gz"):
        alt = FIXTURES / name.removesuffix(".gz")
        if alt.is_file():
            path = alt
    assert path.is_file(), (
        f"Committed frozen corpus {name} is missing. Re-record with scripts/record_eval_corpus.py."
    )
    corpus = load_corpus(path)
    assert corpus["seed_file_sha256"] == file_sha256(seed_path), (
        f"{seed_path.name} changed after {name} was recorded; re-record the corpus."
    )
    assert all("headers" not in v for v in corpus["responses"].values())
    return corpus


def test_frozen_tuning_replay_reports_metrics() -> None:
    corpus = _load_frozen("eval_tuning.json.gz", TUNING_SEEDS)
    metrics = evaluate(corpus["seeds"], corpus["responses"])
    assert metrics.safety == 1.0
    assert metrics.replay_miss_seeds == []


def test_frozen_holdout_meets_phase2_exit_gate() -> None:
    corpus = _load_frozen("eval_holdout.json.gz", HOLDOUT_SEEDS)
    metrics = evaluate(corpus["seeds"], corpus["responses"])
    assert metrics.auto_accept_precision >= 0.95
    assert metrics.adversarial_false_accepts == 0
    assert metrics.auto_accept_recall >= 0.85
    assert metrics.top3_recall >= 0.95
    assert metrics.safety == 1.0
    assert metrics.replay_miss_seeds == []


def test_record_script_exits_2_without_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without a configured contact or --maintainer-contact, the real default is never used:
    the script stops before any network access and writes nothing."""
    module = _load_script("record_eval_corpus")
    assert module.DEFAULT_MB_CONTACT  # The real maintainer default stays in place.
    monkeypatch.delenv("AFV_MB_CONTACT", raising=False)
    monkeypatch.delenv("AFV_MB_USER_AGENT", raising=False)
    monkeypatch.setenv("AFV_CONFIG_PATH", str(tmp_path / "missing.toml"))
    (tmp_path / "eval_seed_tuning.json").write_text(json.dumps(SYNTH_SEEDS), encoding="utf-8")
    hits: list[str] = []

    def _record_hit(request: httpx.Request) -> httpx.Response:
        hits.append(str(request.url))
        return httpx.Response(503)

    before = {p: file_sha256(p) for p in (TUNING_SEEDS, HOLDOUT_SEEDS)}
    code = module.main(
        ["--fixtures", str(tmp_path), "--split", "tuning"],
        transport=httpx.MockTransport(_record_hit),
        sleep=lambda _s: None,
    )
    assert code == 2
    assert hits == []
    assert "--maintainer-contact" in capsys.readouterr().err
    assert {p: file_sha256(p) for p in before} == before
    assert not list(tmp_path.glob("eval_tuning*"))  # No corpus and no checkpoint.


def test_record_script_uses_the_maintainer_contact_only_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("record_eval_corpus")
    monkeypatch.delenv("AFV_MB_CONTACT", raising=False)
    monkeypatch.delenv("AFV_MB_USER_AGENT", raising=False)
    monkeypatch.setenv("AFV_CONFIG_PATH", str(tmp_path / "missing.toml"))
    user_agent, source = module.resolve_recording_user_agent(maintainer_contact=True)
    assert module.DEFAULT_MB_CONTACT in user_agent
    assert source == "maintainer default"
    with pytest.raises(module.ConfigError, match="--maintainer-contact"):
        module.resolve_recording_user_agent()
    # An explicit contact always wins over the flag.
    monkeypatch.setenv("AFV_MB_CONTACT", "relay@duck.com")
    user_agent, source = module.resolve_recording_user_agent(maintainer_contact=True)
    assert "relay@duck.com" in user_agent
    assert module.DEFAULT_MB_CONTACT not in user_agent
    assert source == "configured"


def test_record_script_prints_the_contact_in_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script("record_eval_corpus")
    monkeypatch.delenv("AFV_MB_CONTACT", raising=False)
    monkeypatch.delenv("AFV_MB_USER_AGENT", raising=False)
    monkeypatch.setenv("AFV_CONFIG_PATH", str(tmp_path / "missing.toml"))
    (tmp_path / "eval_seed_tuning.json").write_text(json.dumps(SYNTH_SEEDS), encoding="utf-8")
    code = module.main(
        ["--fixtures", str(tmp_path), "--split", "tuning", "--maintainer-contact"],
        transport=_synthetic_router().transport,
        mb_interval=0.0,
        itunes_interval=0.0,
        max_retries=0,
        sleep=lambda _s: None,
    )
    assert code == 0
    first_line = capsys.readouterr().out.splitlines()[0]
    assert first_line.startswith("MusicBrainz User-Agent: ")
    assert module.DEFAULT_MB_CONTACT in first_line
    assert first_line.endswith("(maintainer default)")


def _load_script(name: str):
    import importlib.util

    script = Path(__file__).parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_record_script_aborts_on_provider_failure_without_writing_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script("record_eval_corpus")
    seed_file = tmp_path / "eval_seed_tuning.json"
    seed_file.write_text(json.dumps(SYNTH_SEEDS), encoding="utf-8")
    monkeypatch.setenv("AFV_MB_CONTACT", "relay@duck.com")

    # e7 scenario: MusicBrainz returns 503 for s3 ("Digital Love") while recording.
    failing = Router()
    failing.add(query_has('recording:"Digital Love"'), lambda r: httpx.Response(503))
    base = _synthetic_router()
    for pred, handler in base.routes:
        failing.add(pred, handler)

    code = module.main(
        ["--fixtures", str(tmp_path), "--split", "tuning"],
        transport=failing.transport,
        mb_interval=0.0,
        itunes_interval=0.0,
        max_retries=0,
        sleep=lambda _s: None,
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "s3" in captured.err
    assert "HTTP 503" in captured.err
    assert "https://musicbrainz.org/ws/2/recording/" in captured.err
    assert "WARNING [s3]" in captured.out
    assert not (tmp_path / "eval_tuning.json.gz").exists()
    ckpt = tmp_path / "eval_tuning.partial.json"
    assert ckpt.is_file()
    saved = json.loads(ckpt.read_text(encoding="utf-8"))
    assert all(v["status"] in (200, 404) for v in saved["responses"].values())

    # Re-running after the provider recovers resumes from checkpoint and writes the corpus.
    ok_code = module.main(
        ["--fixtures", str(tmp_path), "--split", "tuning"],
        transport=_synthetic_router().transport,
        mb_interval=0.0,
        itunes_interval=0.0,
        max_retries=0,
        sleep=lambda _s: None,
    )
    assert ok_code == 0
    out_gz = tmp_path / "eval_tuning.json.gz"
    assert out_gz.is_file()
    assert load_corpus(out_gz)["seeds"] == SYNTH_SEEDS
    assert not ckpt.exists()


def test_reuse_existing_never_refetches_after_seeds_are_added(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding a seed changes the seed-file hash; existing responses must still be reused."""
    module = _load_script("record_eval_corpus")
    seed_file = tmp_path / "eval_seed_tuning.json"
    seed_file.write_text(json.dumps(SYNTH_SEEDS), encoding="utf-8")
    monkeypatch.setenv("AFV_MB_CONTACT", "relay@duck.com")
    kwargs = {"mb_interval": 0.0, "itunes_interval": 0.0, "max_retries": 0}
    assert (
        module.main(
            ["--fixtures", str(tmp_path), "--split", "tuning"],
            transport=_synthetic_router().transport,
            sleep=lambda _s: None,
            **kwargs,
        )
        == 0
    )
    out_gz = tmp_path / "eval_tuning.json.gz"
    before = load_corpus(out_gz)["responses"]

    # Add a seed whose lookups are all already recorded (same input as s1, new id).
    extra = json.loads(json.dumps(SYNTH_SEEDS[0]))
    extra["id"] = "s1-copy"
    seed_file.write_text(json.dumps([*SYNTH_SEEDS, extra]), encoding="utf-8")

    hits: list[str] = []

    def _refuse(request: httpx.Request) -> httpx.Response:
        hits.append(str(request.url))
        return httpx.Response(503)

    code = module.main(
        ["--fixtures", str(tmp_path), "--split", "tuning", "--reuse-existing"],
        transport=httpx.MockTransport(_refuse),
        sleep=lambda _s: None,
        **kwargs,
    )
    assert code == 0
    assert hits == []  # Nothing re-fetched despite the changed seed-file hash.
    after = load_corpus(out_gz)
    assert [s["id"] for s in after["seeds"]][-1] == "s1-copy"
    assert all(after["responses"][url] == resp for url, resp in before.items())


def test_evaluate_treats_frozen_5xx_and_429_as_replay_misses() -> None:
    responses = _record(SYNTH_SEEDS)
    # Poison s3 (adversarial) with a frozen 503 and s1 (auto_accept) with a frozen 429.
    corrupted = json.loads(json.dumps(responses))
    for url in list(corrupted):
        if "Digital+Love" in url or "Digital%20Love" in url:
            corrupted[url] = {"status": 503, "json": None}
        if "One+More+Time" in url or "One%20More%20Time" in url:
            corrupted[url] = {"status": 429, "json": None}
    metrics = evaluate(SYNTH_SEEDS, corrupted)
    by_id = {o.seed_id: o for o in metrics.outcomes}
    assert "s1" in metrics.replay_miss_seeds
    assert "s3" in metrics.replay_miss_seeds
    assert any("HTTP 429" in m for m in by_id["s1"].replay_misses)
    assert any("HTTP 503" in m for m in by_id["s3"].replay_misses)
    assert not by_id["s1"].correct
    assert not by_id["s1"].in_top3


def test_record_all_enriches_album_behind_thirteen_singles_for_threshold_sweeps() -> None:
    # e9 scenario: 13 singles listed ahead of the studio album (all tied at score 1.0),
    # recorded under a strict config where the seed is ambiguous during recording.
    dp = credit(("Daft Punk", "", "artist-dp"))
    alt = credit(("Daft Punk Tribute", "", "artist-alt"))
    singles_1 = [
        release(
            f"rel-single-{i}",
            f"One More Time Single {i}",
            f"rg-single-{i}",
            primary="Single",
            date=f"{2001 + i:04d}-01-01",
            track_number=1,
        )
        for i in range(13)
    ]
    album_1 = release(
        "rel-disc",
        "Discovery",
        "rg-disc",
        primary="Album",
        date="2001-03-12",
        track_number=1,
    )
    singles_2 = [
        release(
            f"rel-alt-s-{i}",
            f"Single {i}",
            f"rg-alt-s-{i}",
            primary="Single",
            date=f"{2005 + i:04d}-01-01",
        )
        for i in range(13)
    ]
    album_2 = release("rel-alt-lp", "Alt Album", "rg-alt-lp", primary="Album", date="2005-01-01")
    router = Router()
    router.json(
        query_has('recording:"One More Time"'),
        search_payload(
            recording("rec-omt", "One More Time", dp, [*singles_1, album_1], length_ms=320_000),
            recording("rec-alt", "One More Time", alt, [*singles_2, album_2], length_ms=320_000),
        ),
    )
    router.json(
        path_is("/ws/2/release/rel-disc"),
        release_lookup(
            "rel-disc",
            "Discovery",
            artist_credits=dp,
            rg_id="rg-disc",
            rg_first_date="2001-03-07",
            date="2001-03-12",
            media=[medium(1, [("rec-omt", "One More Time")])],
            genres=[{"name": "house", "count": 5}],
        ),
    )
    router.json(
        path_is("/ws/2/release/rel-alt-lp"),
        release_lookup(
            "rel-alt-lp",
            "Alt Album",
            artist_credits=alt,
            rg_id="rg-alt-lp",
            rg_first_date="2005-01-01",
            date="2005-01-01",
            media=[medium(1, [("rec-alt", "One More Time")])],
            genres=[{"name": "electronic", "count": 1}],
        ),
    )
    seed = [SYNTH_SEEDS[0]]
    recorder = RecordingTransport(router.transport)

    def f() -> HttpFetcher:
        return HttpFetcher(transport=recorder, limiter=RateLimiter(0.0), sleep=lambda _s: None)

    mb = MusicBrainzClient(REPLAY_UA, fetcher=f(), art_fetcher=f())
    itunes = ITunesClient(fetcher=f(), art_fetcher=f())
    # Record under a strict config where the seed ends ambiguous (auto_accept_score=0.99,
    # min_margin=0.99); record_all must still enrich BOTH identities' studio albums first!
    strict = MatchingConfig(auto_accept_score=0.99, min_margin=0.99)
    res = TrackResolver(mb, itunes, strict, record_all=True).resolve(
        parse_hints(seed_to_probe(seed[0]))
    )
    assert res.decision.value == "ambiguous"
    assert any("/ws/2/release/rel-disc" in u for u in recorder.responses)
    assert any("/ws/2/release/rel-alt-lp" in u for u in recorder.responses)

    # Replay under default config: zero misses and Discovery is auto-accepted!
    metrics = evaluate(seed, recorder.responses)
    assert metrics.replay_miss_seeds == []
    assert metrics.outcomes[0].decision == "auto_accept"
    assert metrics.outcomes[0].correct


def test_eval_report_prints_metrics_failures_and_sweep(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script("eval_report")
    seed_file = tmp_path / "eval_seed_tuning.json"
    seed_file.write_text(json.dumps(SYNTH_SEEDS), encoding="utf-8")
    corpus_file = tmp_path / "eval_tuning.json.gz"
    corpus = build_frozen_corpus(SYNTH_SEEDS, file_sha256(seed_file), _record(SYNTH_SEEDS))
    write_corpus(corpus_file, corpus)
    code = module.main([str(corpus_file), "--all", "--sweep"])
    out = capsys.readouterr().out
    assert code == 0
    assert "auto_accept_precision" in out
    assert "Threshold sweep" in out
    assert "s1" in out

    # --sweep must refuse holdout corpora to preserve out-of-sample discipline.
    holdout_file = tmp_path / "eval_holdout.json.gz"
    write_corpus(holdout_file, corpus)
    assert module.main([str(holdout_file), "--sweep"]) == 2
    assert "forbidden" in capsys.readouterr().err
