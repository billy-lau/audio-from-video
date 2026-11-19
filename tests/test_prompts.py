"""Regression tests for interactive prompts, alias dedup, contact save, and provider outages."""

from __future__ import annotations

import builtins
import io
from pathlib import Path
from typing import Any

import httpx
import pytest
from rich.console import Console

from audio_from_video.cli.prompts import group_editions, make_rich_prompt_callbacks
from audio_from_video.config import ConfigError, load_config, save_mb_contact
from audio_from_video.lookup import MatchResult, TrackResolver
from audio_from_video.models import (
    Candidate,
    ManualOverrides,
    MatchDecision,
    MatchStatus,
    MediaInput,
    ParsedHints,
    ReleaseRef,
    TrackMetadata,
)
from audio_from_video.planner import (
    AliasAction,
    AliasPromptResponse,
    CandidateAction,
    PromptCallbacks,
    _AliasSession,
    _resolve_non_latin_aliases,
    build_plan,
)
from audio_from_video.providers.base import HttpFetcher, ProviderError, RateLimiter
from conftest import make_video_with_audio


def _candidate(artist: str, title: str, album: str) -> Candidate:
    return Candidate(source="musicbrainz", title=title, artist=artist, album=album, score=0.7)


def test_real_console_input_shows_literal_option_keys_and_bracketed_filenames(
    monkeypatch: Any,
) -> None:
    """Prompts go through console.input without Rich markup, so '[s]' etc. stay visible."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=300)
    answers = iter(["f", "", "k"])
    monkeypatch.setattr(builtins, "input", lambda *_a, **_k: next(answers))
    callbacks = make_rich_prompt_callbacks(console=console)  # No input_fn: real console path.
    assert callbacks.resolve_match is not None
    assert callbacks.select_edition is not None
    assert callbacks.resolve_non_latin_alias is not None

    media = MediaInput(path=Path("Song [live] [b].mp4"), from_directory_scan=False)
    match = MatchResult(
        decision=MatchDecision.AMBIGUOUS,
        best=_candidate("A", "Song", "X"),
        alternatives=[],
        score=0.7,
        margin=0.0,
        queries_tried=[],
        metadata=TrackMetadata(),
    )
    resp = callbacks.resolve_match(media, ParsedHints(title="Song"), match, False)
    assert resp.action == CandidateAction.USE_FALLBACK

    ed1 = ReleaseRef(release_id="r1", title="X", status="Official", formats=("CD",))
    ed2 = ReleaseRef(release_id="r2", title="X", status="Official", formats=("Vinyl",))
    chosen = callbacks.select_edition(media, _candidate("A", "Song", "X"), [ed1, ed2], ed1)
    assert chosen == ed1

    alias = callbacks.resolve_non_latin_alias(media, "album", "[re:Stacks] 吻別", "album 'x'")
    assert alias.action == AliasAction.KEEP_ITEM

    out = buf.getvalue()
    for literal in ("[s] manual search", "[m] custom tags", "[f] fallback", "[k] skip", "[F]"):
        assert literal in out
    assert "Song [live] [b].mp4" in out
    assert "[a] enter English alias" in out
    assert "[A] keep native" in out


def test_itunes_seasonal_genre_dropped_on_non_holiday_albums() -> None:
    from audio_from_video.providers.itunes import clean_itunes_genre, results_to_candidates

    payload = {
        "results": [
            {
                "kind": "song",
                "artistName": "Jeff Buckley",
                "trackName": "Hallelujah",
                "collectionName": "Grace",
                "primaryGenreName": "Holiday",
            },
            {
                "kind": "song",
                "artistName": "Pentatonix",
                "trackName": "Hallelujah",
                "collectionName": "A Pentatonix Christmas (Deluxe)",
                "primaryGenreName": "Holiday",
            },
            {
                "kind": "song",
                "artistName": "Leonard Cohen",
                "trackName": "Hallelujah",
                "collectionName": "The Essential Leonard Cohen",
                "primaryGenreName": "Singer/Songwriter",
            },
        ]
    }
    genres = [c.genre for c in results_to_candidates(payload)]
    assert genres == [None, "Holiday", "Singer/Songwriter"]
    assert clean_itunes_genre("Christmas: Pop", "Grace") is None
    assert clean_itunes_genre("Christmas: Pop", "Merry Christmas") == "Christmas: Pop"
    assert clean_itunes_genre("", "Grace") is None


def test_artist_and_album_alias_prompts_are_asked_once_per_run_with_enter() -> None:
    session = _AliasSession()
    prompts: list[tuple[str, str]] = []

    def _keep_item(_media: MediaInput, kind: str, value: str, _scope: str) -> AliasPromptResponse:
        prompts.append((kind, value))
        return AliasPromptResponse(action=AliasAction.KEEP_ITEM)

    callbacks = PromptCallbacks(resolve_non_latin_alias=_keep_item)
    for title in ("吻別", "一路上有你", "秋意濃"):
        meta = TrackMetadata(
            artist="張學友",
            album_artist="張學友",
            album="吻別",
            title=title,
            match_status=MatchStatus.EXACT,
        )
        _resolve_non_latin_aliases(
            MediaInput(path=Path(f"{title}.mp4"), from_directory_scan=False),
            meta,
            overrides=ManualOverrides(),
            session=session,
            callbacks=callbacks,
        )
    # Artist once, album once, and three titles (titles stay per-track).
    assert [kind for kind, _ in prompts] == ["artist", "album", "title", "title", "title"]


def test_save_mb_contact_handles_array_tables_and_verifies_result(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[musicbrainz]\ncontact = "old@duck.com"\n\n[[profiles]]\nname = "x"\n',
        encoding="utf-8",
    )
    save_mb_contact("relay@duck.com", config_path=cfg)
    assert load_config(config_path=cfg, env={}).musicbrainz.contact == "relay@duck.com"

    cfg2 = tmp_path / "config2.toml"
    cfg2.write_text(
        '[musicbrainz]\nuser_agent_note = 1\n\n[[profiles]]\nname = "y"\n', encoding="utf-8"
    )
    save_mb_contact("relay@duck.com", config_path=cfg2)
    parsed = load_config(config_path=cfg2, env={})
    assert parsed.musicbrainz.contact == "relay@duck.com"

    # A multi-line string containing a fake [musicbrainz] header cannot be updated safely,
    # so it raises instead of silently reporting success (and leaves the file untouched).
    cfg3 = tmp_path / "config3.toml"
    original = '[notes]\ntext = """\n[musicbrainz]\n"""\n'
    cfg3.write_text(original, encoding="utf-8")
    with pytest.raises(ConfigError):
        save_mb_contact("relay@duck.com", config_path=cfg3)
    assert cfg3.read_text(encoding="utf-8") == original


def test_group_editions_collapses_pressings_and_keeps_default_representative() -> None:
    vinyls = [
        ReleaseRef(
            release_id=f"v{i}",
            title="Rumours",
            status="Official",
            date="1977",
            country=c,
            disc_count=1,
            track_count=11,
            formats=('12" Vinyl',),
        )
        for i, c in enumerate(["US", "GB", "DE", "FR", "JP", "CA", "AU", "NL", "IT", "ES", "SE"])
    ]
    cd = ReleaseRef(
        release_id="cd1",
        title="Rumours",
        status="Official",
        date="1983",
        disc_count=1,
        track_count=11,
        formats=("CD",),
    )
    deluxe = ReleaseRef(
        release_id="dx",
        title="Rumours",
        status="Official",
        disambiguation="Deluxe",
        disc_count=3,
        track_count=38,
        formats=("CD", "CD", "CD"),
    )
    grouped = group_editions([*vinyls, cd, deluxe], vinyls[3])
    assert len(grouped) == 3
    assert grouped[0] == (vinyls[3], 11)  # Default edition represents its group.
    assert grouped[1] == (cd, 1)
    assert grouped[2] == (deluxe, 1)


def test_connect_timeout_is_not_retried() -> None:
    calls = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectTimeout("timed out", request=request)

    fetcher = HttpFetcher(
        transport=httpx.MockTransport(_handler),
        limiter=RateLimiter(0.0),
        max_retries=4,
        sleep=lambda _s: None,
    )
    with pytest.raises(ProviderError) as exc_info:
        fetcher.get_json("https://musicbrainz.org/ws/2/recording/")
    assert exc_info.value.transient is True
    assert calls == 1
    fetcher.close()


def test_provider_outage_stops_prompting_and_notifies_once(
    tmp_path: Path, ffprobe_path: Path
) -> None:
    from audio_from_video.providers.itunes import ITunesClient
    from audio_from_video.providers.musicbrainz import MusicBrainzClient

    videos = tmp_path / "videos"
    videos.mkdir()
    for name in (
        "Daft Punk - One More Time.mp4",
        "Daft Punk - Aerodynamic.mp4",
        "Adele - Hello.mp4",
    ):
        make_video_with_audio(videos / name)

    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network unreachable", request=request)

    transport = httpx.MockTransport(_handler)

    def _fetcher() -> HttpFetcher:
        return HttpFetcher(
            transport=transport, limiter=RateLimiter(0.0), max_retries=0, sleep=lambda _s: None
        )

    mb = MusicBrainzClient(
        "audio-from-video/0.1.0 ( relay@duck.com )", fetcher=_fetcher(), art_fetcher=_fetcher()
    )
    itunes = ITunesClient(fetcher=_fetcher(), art_fetcher=_fetcher())
    resolver = TrackResolver(mb, itunes)

    notices: list[str] = []

    def _must_not_prompt(*_args: object) -> Any:
        raise AssertionError("Match prompt must not fire while providers are unreachable")

    callbacks = PromptCallbacks(resolve_match=_must_not_prompt, notify=notices.append)
    from audio_from_video.discovery import discover_inputs

    media_inputs, output_root = discover_inputs([videos], output_dir=tmp_path / "out")
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    plan = build_plan(
        media_inputs,
        output_root=output_root,
        ffprobe_path=ffprobe_path,
        config=cfg,
        resolver=resolver,
        non_interactive=False,
        keep_native_missing=True,
        prompt_callbacks=callbacks,
    )
    assert resolver.providers_unavailable is True
    assert len(notices) == 1
    assert "unreachable" in notices[0]
    assert all(item.metadata is not None for item in plan)
    assert all(item.metadata.match_status == MatchStatus.PARTIAL for item in plan if item.metadata)


def test_manual_search_without_results_keeps_previous_candidates_and_fallback_all(
    tmp_path: Path, ffprobe_path: Path
) -> None:
    from audio_from_video.discovery import discover_inputs

    videos = tmp_path / "videos"
    videos.mkdir()
    make_video_with_audio(videos / "Creep.mp4")
    make_video_with_audio(videos / "Hello.mp4")
    make_video_with_audio(videos / "Yesterday.mp4")

    original = [_candidate("Radiohead", "Creep", "Pablo Honey"), _candidate("TLC", "Creep", "CSC")]

    class _FakeResolver:
        providers_unavailable = False

        def resolve_batch(self, items: Any, *, on_progress: Any = None) -> list[MatchResult]:
            return [
                MatchResult(
                    decision=MatchDecision.AMBIGUOUS,
                    best=original[0],
                    alternatives=original[1:],
                    score=0.7,
                    margin=0.0,
                    queries_tried=[],
                    metadata=TrackMetadata(),
                )
                for _ in items
            ]

        def manual_search(self, **_kwargs: Any) -> MatchResult:
            return MatchResult(
                decision=MatchDecision.UNMATCHED,
                best=None,
                alternatives=[],
                score=0.0,
                margin=0.0,
                queries_tried=[],
                metadata=TrackMetadata(),
            )

    seen_candidate_counts: list[int] = []
    notices: list[str] = []
    responses = iter(["search", "fallback_all"])

    def _resolve_match(_media: Any, _hints: Any, match: MatchResult, _always: bool) -> Any:
        from audio_from_video.planner import CandidatePromptResponse

        seen_candidate_counts.append(len(match.top_candidates))
        step = next(responses)
        if step == "search":
            return CandidatePromptResponse(
                action=CandidateAction.MANUAL_SEARCH, search_title="Nothing Here"
            )
        return CandidatePromptResponse(action=CandidateAction.USE_FALLBACK_ALL)

    callbacks = PromptCallbacks(resolve_match=_resolve_match, notify=notices.append)
    media_inputs, output_root = discover_inputs([videos], output_dir=tmp_path / "out")
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    plan = build_plan(
        media_inputs,
        output_root=output_root,
        ffprobe_path=ffprobe_path,
        config=cfg,
        resolver=_FakeResolver(),  # type: ignore[arg-type]
        non_interactive=False,
        keep_native_missing=True,
        prompt_callbacks=callbacks,
    )
    # Prompted twice for the first file only: the empty search kept both original candidates,
    # and [F] applied the fallback to the remaining two files without prompting.
    assert seen_candidate_counts == [2, 2]
    assert any("no candidates" in n for n in notices)
    assert len(plan) == 3
    assert all(item.candidate is None for item in plan)
