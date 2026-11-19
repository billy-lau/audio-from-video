"""Phase 3 end-to-end pipeline, interactivity, artwork, normalize, and path-hardening tests."""

from __future__ import annotations

import io
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from mutagen.id3 import ID3
from PIL import Image
from rich.console import Console
from typer.testing import CliRunner

from audio_from_video.artwork import ArtworkFetcher
from audio_from_video.cache import TTLCache
from audio_from_video.cli import main as cli_main
from audio_from_video.cli.prompts import make_rich_prompt_callbacks, prompt_mb_contact
from audio_from_video.config import load_config
from audio_from_video.discovery import discover_inputs
from audio_from_video.extractor import ProcessRegistry
from audio_from_video.lookup import TrackResolver
from audio_from_video.models import (
    ConflictPolicy,
    MatchDecision,
    MatchStatus,
    PlanAction,
    RunOutcome,
)
from audio_from_video.planner import (
    build_offline_plan,
    build_plan,
)
from audio_from_video.providers.base import HttpFetcher, RateLimiter
from audio_from_video.providers.itunes import ITunesClient
from audio_from_video.providers.musicbrainz import MusicBrainzClient
from audio_from_video.runner import execute_plan, write_json_report
from conftest import (
    make_multi_track_mkv,
    make_silent_video_no_audio,
    make_video_with_audio,
)

_TEST_UA = "audio-from-video/0.1.0 ( relay@duck.com )"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _make_jpeg_bytes(
    size: tuple[int, int] = (600, 600), color: tuple[int, int, int] = (40, 80, 160)
) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, color)
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _make_online_stack(
    handler: Any,
    cache_path: Path,
    *,
    matching_config: Any = None,
) -> tuple[TTLCache, MusicBrainzClient, ITunesClient, TrackResolver, ArtworkFetcher]:
    cache = TTLCache(cache_path)
    transport = httpx.MockTransport(handler)
    mb_fetcher = HttpFetcher(
        transport=transport, limiter=RateLimiter(0.0), max_retries=0, sleep=lambda _: None
    )
    mb_art = HttpFetcher(
        transport=transport, limiter=RateLimiter(0.0), max_retries=0, sleep=lambda _: None
    )
    it_fetcher = HttpFetcher(
        transport=transport, limiter=RateLimiter(0.0), max_retries=0, sleep=lambda _: None
    )
    it_art = HttpFetcher(
        transport=transport, limiter=RateLimiter(0.0), max_retries=0, sleep=lambda _: None
    )
    mb = MusicBrainzClient(_TEST_UA, fetcher=mb_fetcher, art_fetcher=mb_art, cache=cache)
    itunes = ITunesClient(fetcher=it_fetcher, art_fetcher=it_art, cache=cache)
    resolver = TrackResolver(mb, itunes, matching_config)
    art = ArtworkFetcher(mb, itunes, cache=cache)
    return cache, mb, itunes, resolver, art


def _daft_punk_recording_search(title: str, rec_id: str, track_pos: int) -> dict[str, Any]:
    return {
        "recordings": [
            {
                "id": rec_id,
                "title": title,
                "length": 600,
                "artist-credit": [
                    {
                        "name": "Daft Punk",
                        "artist": {"id": "art-dp", "name": "Daft Punk"},
                        "joinphrase": "",
                    }
                ],
                "releases": [
                    {
                        "id": "rel-disc-orig",
                        "title": "Discovery",
                        "status": "Official",
                        "date": "2001-03-12",
                        "country": "FR",
                        "count": 1,
                        "track-count": 14,
                        "release-group": {
                            "id": "rg-disc",
                            "title": "Discovery",
                            "primary-type": "Album",
                            "secondary-types": [],
                        },
                        "media": [
                            {
                                "position": 1,
                                "format": "CD",
                                "track-count": 14,
                                "track": [{"number": str(track_pos), "position": track_pos}],
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _daft_punk_release_detail() -> dict[str, Any]:
    return {
        "id": "rel-disc-orig",
        "title": "Discovery",
        "status": "Official",
        "date": "2001-03-12",
        "country": "FR",
        "artist-credit": [
            {"name": "Daft Punk", "artist": {"id": "art-dp", "name": "Daft Punk"}, "joinphrase": ""}
        ],
        "release-group": {
            "id": "rg-disc",
            "title": "Discovery",
            "first-release-date": "2001-03-12",
            "genres": [{"name": "french house", "count": 12}],
        },
        "genres": [{"name": "electronic", "count": 5}],
        "media": [
            {
                "position": 1,
                "format": "CD",
                "track-count": 14,
                "tracks": [
                    {
                        "position": 1,
                        "number": "1",
                        "title": "One More Time",
                        "recording": {"id": "rec-omt", "title": "One More Time"},
                    },
                    {
                        "position": 2,
                        "number": "2",
                        "title": "Aerodynamic",
                        "recording": {"id": "rec-aero", "title": "Aerodynamic"},
                    },
                ],
            }
        ],
    }


def test_online_pipeline_end_to_end_artwork_dedup_dry_run_no_artwork_and_report(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    make_video_with_audio(videos / "Daft Punk - One More Time [Official Video].mp4")
    make_video_with_audio(videos / "Daft Punk - Aerodynamic [Official Audio].mp4")
    make_video_with_audio(videos / "VID_0042.mp4")
    make_silent_video_no_audio(videos / "banner.mp4")

    jpeg_bytes = _make_jpeg_bytes()
    caa_hits = 0
    mb_search_urls: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal caa_hits
        url = str(request.url)
        if request.url.host == "coverartarchive.org":
            caa_hits += 1
            if "/release/rel-disc-orig/front-1200" in request.url.path:
                return httpx.Response(
                    200, content=jpeg_bytes, headers={"Content-Type": "image/jpeg"}
                )
            return httpx.Response(404)
        if request.url.host == "musicbrainz.org":
            if "/ws/2/recording/" in request.url.path:
                mb_search_urls.append(url)
                q = request.url.params.get("query", "")
                if "One More Time" in q:
                    return httpx.Response(
                        200, json=_daft_punk_recording_search("One More Time", "rec-omt", 1)
                    )
                if "Aerodynamic" in q:
                    return httpx.Response(
                        200, json=_daft_punk_recording_search("Aerodynamic", "rec-aero", 2)
                    )
                return httpx.Response(200, json={"recordings": []})
            if "/ws/2/release/rel-disc-orig" in request.url.path:
                return httpx.Response(200, json=_daft_punk_release_detail())
        return httpx.Response(404)

    cache, mb, itunes, resolver, art = _make_online_stack(_handler, tmp_path / "cache.sqlite3")
    try:
        out_dir = tmp_path / "out"
        media_inputs, output_root = discover_inputs([videos], output_dir=out_dir)
        cfg = load_config(config_path=tmp_path / "missing.toml", env={})

        # Stage 1 planning (as used by --dry-run): ZERO artwork downloads, ZERO output files!
        plan = build_plan(
            media_inputs,
            output_root=output_root,
            ffprobe_path=ffprobe_path,
            config=cfg,
            resolver=resolver,
            non_interactive=True,
        )
        assert caa_hits == 0
        assert not out_dir.exists()
        # Hintless VID_0042.mp4 must NEVER trigger an API call!
        assert all("VID_0042" not in u for u in mb_search_urls)

        # Stage 2 execution with ArtworkFetcher: fetches cover art ONCE for both album tracks!
        summary = execute_plan(
            plan,
            ffmpeg_path=ffmpeg_path,
            output_root=output_root,
            jobs=2,
            artwork_fetcher=art,
        )
        assert summary.exit_code == 0
        assert summary.count(RunOutcome.SUCCEEDED) == 2
        assert summary.count(RunOutcome.UNMATCHED) == 1
        assert summary.count(RunOutcome.SKIPPED_NO_AUDIO) == 1
        assert caa_hits == 1  # Lock-deduplicated across both Discovery tracks!

        omt = output_root / "Daft Punk" / "Discovery" / "01 Daft Punk - One More Time.mp3"
        aero = output_root / "Daft Punk" / "Discovery" / "02 Daft Punk - Aerodynamic.mp3"
        unm = output_root / "_unmatched" / "VID_0042.mp3"
        assert omt.is_file() and aero.is_file() and unm.is_file()

        tags = ID3(str(omt), translate=False)
        assert tags.version == (2, 3, 0)
        assert tags["TPE1"].text == ["Daft Punk"]
        assert tags["TALB"].text == ["Discovery"]
        assert tags["TRCK"].text == ["1/14"]
        assert str(tags["TYER"]) == "2001"
        assert str(tags["TDAT"]) == "1203"
        assert str(tags["TORY"]) == "2001"
        assert tags["TCON"].text == ["French House"]
        assert any(f.FrameID == "APIC" for f in tags.values())
        assert any(f.FrameID == "UFID" for f in tags.values())

        # JSON report includes artwork_source, match_decision, and MBIDs.
        report_path = tmp_path / "report.json"
        write_json_report(summary, report_path)
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        assert rep["counts"]["succeeded"] == 2
        omt_entry = next(t for t in rep["tracks"] if "One More Time" in t["source"])
        assert omt_entry["artwork_source"] == "coverartarchive"
        assert omt_entry["match_decision"] == "auto_accept"
        assert omt_entry["metadata"]["mb_release_id"] == "rel-disc-orig"
        assert omt_entry["metadata"]["original_year"] == "2001"

        # --no-artwork makes zero artwork fetches and writes no APIC frame.
        caa_before = caa_hits
        out_no_art = tmp_path / "out_no_art"
        plan_na = build_plan(
            media_inputs,
            output_root=out_no_art,
            ffprobe_path=ffprobe_path,
            config=cfg,
            resolver=resolver,
            non_interactive=True,
        )
        sum_na = execute_plan(
            plan_na,
            ffmpeg_path=ffmpeg_path,
            output_root=out_no_art,
            jobs=2,
            no_artwork=True,
            artwork_fetcher=art,
        )
        assert sum_na.exit_code == 0
        assert caa_hits == caa_before
        tags_na = ID3(
            str(out_no_art / "Daft Punk" / "Discovery" / "01 Daft Punk - One More Time.mp3")
        )
        assert not any(f.FrameID == "APIC" for f in tags_na.values())
    finally:
        mb.fetcher.close()
        mb.art_fetcher.close()
        itunes.fetcher.close()
        itunes.art_fetcher.close()
        cache.close()


def test_online_artwork_failure_is_non_fatal_and_warns(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    clip = make_video_with_audio(tmp_path / "Daft Punk - One More Time.mp4")

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "coverartarchive.org":
            return httpx.Response(200, content=b"not-a-valid-jpeg-stream")
        if request.url.host == "itunes.apple.com":
            return httpx.Response(200, json={"results": []})
        if request.url.host == "musicbrainz.org":
            if "/ws/2/recording/" in request.url.path:
                return httpx.Response(
                    200, json=_daft_punk_recording_search("One More Time", "rec-omt", 1)
                )
            if "/ws/2/release/rel-disc-orig" in request.url.path:
                return httpx.Response(200, json=_daft_punk_release_detail())
        return httpx.Response(404)

    cache, mb, itunes, resolver, art = _make_online_stack(_handler, tmp_path / "cache.sqlite3")
    try:
        out_dir = tmp_path / "out"
        media_inputs, output_root = discover_inputs([clip], output_dir=out_dir)
        cfg = load_config(config_path=tmp_path / "missing.toml", env={})
        plan = build_plan(
            media_inputs,
            output_root=output_root,
            ffprobe_path=ffprobe_path,
            config=cfg,
            resolver=resolver,
            non_interactive=True,
        )
        summary = execute_plan(
            plan,
            ffmpeg_path=ffmpeg_path,
            output_root=output_root,
            artwork_fetcher=art,
        )
        assert summary.exit_code == 0
        assert summary.count(RunOutcome.SUCCEEDED) == 1
        res = summary.results[0]
        assert res.artwork_source is None
        assert any("cover art" in w.lower() or "decode" in w.lower() for w in res.warnings)
        assert res.output_path is not None and res.output_path.is_file()
        tags = ID3(str(res.output_path))
        assert tags["TPE1"].text == ["Daft Punk"]
        assert not any(f.FrameID == "APIC" for f in tags.values())
    finally:
        mb.fetcher.close()
        mb.art_fetcher.close()
        itunes.fetcher.close()
        itunes.art_fetcher.close()
        cache.close()


def test_non_interactive_never_auto_applies_ambiguous_matches(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    make_video_with_audio(videos / "Johnny Cash - Hurt.mp4")
    make_video_with_audio(videos / "Creep.mp4")

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "musicbrainz.org" and "/ws/2/recording/" in request.url.path:
            q = request.url.params.get("query", "")
            if "Hurt" in q:
                # Two distinct competing identities with close scores (margin < 0.08 -> AMBIGUOUS).
                return httpx.Response(
                    200,
                    json={
                        "recordings": [
                            {
                                "id": "rec-hurt-1",
                                "title": "Hurt",
                                "length": 320000,
                                "artist-credit": [
                                    {
                                        "name": "Johnny Cash",
                                        "artist": {"id": "a1", "name": "Johnny Cash"},
                                    }
                                ],
                                "releases": [
                                    {
                                        "id": "rel-1",
                                        "title": "American IV",
                                        "status": "Official",
                                        "date": "2002",
                                        "release-group": {
                                            "id": "rg-1",
                                            "title": "American IV",
                                            "primary-type": "Album",
                                        },
                                    }
                                ],
                            },
                            {
                                "id": "rec-hurt-2",
                                "title": "Hurt",
                                "length": 320000,
                                "artist-credit": [
                                    {
                                        "name": "Johnny Cash & Band",
                                        "artist": {"id": "a2", "name": "Johnny Cash & Band"},
                                    }
                                ],
                                "releases": [
                                    {
                                        "id": "rel-2",
                                        "title": "Live Recording",
                                        "status": "Official",
                                        "date": "2003",
                                        "release-group": {
                                            "id": "rg-2",
                                            "title": "Live Recording",
                                            "primary-type": "Album",
                                        },
                                    }
                                ],
                            },
                        ]
                    },
                )
            if "Creep" in q:
                return httpx.Response(
                    200,
                    json={
                        "recordings": [
                            {
                                "id": "rec-creep-rh",
                                "title": "Creep",
                                "length": 600,
                                "artist-credit": [
                                    {
                                        "name": "Radiohead",
                                        "artist": {"id": "a-rh", "name": "Radiohead"},
                                    }
                                ],
                                "releases": [
                                    {
                                        "id": "rel-ph",
                                        "title": "Pablo Honey",
                                        "status": "Official",
                                        "date": "1993",
                                        "release-group": {
                                            "id": "rg-ph",
                                            "title": "Pablo Honey",
                                            "primary-type": "Album",
                                        },
                                    }
                                ],
                            }
                        ]
                    },
                )
        if request.url.host == "itunes.apple.com":
            return httpx.Response(200, json={"results": []})
        return httpx.Response(404)

    cache, mb, itunes, resolver, _art = _make_online_stack(_handler, tmp_path / "cache.sqlite3")
    try:
        out_dir = tmp_path / "out"
        media_inputs, output_root = discover_inputs([videos], output_dir=out_dir)
        cfg = load_config(config_path=tmp_path / "missing.toml", env={})
        plan = build_plan(
            media_inputs,
            output_root=output_root,
            ffprobe_path=ffprobe_path,
            config=cfg,
            resolver=resolver,
            non_interactive=True,
        )
        by_name = {item.media_input.path.name: item for item in plan}

        # Trusted artist + title -> safe partial fallback to Unknown Album/ (never auto-applies!).
        hurt = by_name["Johnny Cash - Hurt.mp4"]
        assert hurt.match_decision == MatchDecision.AMBIGUOUS
        assert hurt.metadata is not None and hurt.metadata.match_status == MatchStatus.PARTIAL
        assert (
            hurt.output_path
            == output_root / "Johnny Cash" / "Unknown Album" / "Johnny Cash - Hurt.mp3"
        )
        assert len(hurt.top_candidates) >= 2
        assert any("ambiguous" in w.lower() for w in hurt.warnings)

        # Title-only -> safe unmatched fallback to _unmatched/!
        creep = by_name["Creep.mp4"]
        assert creep.match_decision == MatchDecision.AMBIGUOUS
        assert creep.metadata is not None and creep.metadata.match_status == MatchStatus.UNMATCHED
        assert creep.output_path == output_root / "_unmatched" / "Creep.mp3"

        summary = execute_plan(plan, ffmpeg_path=ffmpeg_path, output_root=output_root)
        report_path = tmp_path / "ambig_report.json"
        write_json_report(summary, report_path)
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        hurt_rep = next(t for t in rep["tracks"] if "Hurt" in t["source"])
        assert len(hurt_rep["top_candidates"]) >= 2
    finally:
        mb.fetcher.close()
        mb.art_fetcher.close()
        itunes.fetcher.close()
        itunes.art_fetcher.close()
        cache.close()


def test_interactive_picker_manual_search_custom_tags_skip_and_always_prompt_edition(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    make_video_with_audio(videos / "01_ambig_Creep.mp4")  # title-only -> ambiguous
    make_video_with_audio(videos / "02_VID_0042.mp4")  # hintless -> manual search
    make_video_with_audio(videos / "03_unknown_jam.mp4")  # unmatched -> custom tags
    make_video_with_audio(videos / "04_skip_me.mp4")  # unmatched -> skip

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "musicbrainz.org":
            if "/ws/2/recording/" in request.url.path:
                q = request.url.params.get("query", "")
                if "Creep" in q:
                    return httpx.Response(
                        200,
                        json={
                            "recordings": [
                                {
                                    "id": "rec-creep-1",
                                    "title": "Creep",
                                    "artist-credit": [
                                        {
                                            "name": "Radiohead",
                                            "artist": {"id": "a-rh", "name": "Radiohead"},
                                        }
                                    ],
                                    "releases": [
                                        {
                                            "id": "rel-rh",
                                            "title": "Pablo Honey",
                                            "status": "Official",
                                            "date": "1993",
                                            "release-group": {
                                                "id": "rg-rh",
                                                "title": "Pablo Honey",
                                                "primary-type": "Album",
                                            },
                                        }
                                    ],
                                },
                                {
                                    "id": "rec-creep-2",
                                    "title": "Creep",
                                    "artist-credit": [
                                        {"name": "TLC", "artist": {"id": "a-tlc", "name": "TLC"}}
                                    ],
                                    "releases": [
                                        {
                                            "id": "rel-tlc",
                                            "title": "CrazySexyCool",
                                            "status": "Official",
                                            "date": "1994",
                                            "release-group": {
                                                "id": "rg-tlc",
                                                "title": "CrazySexyCool",
                                                "primary-type": "Album",
                                            },
                                        }
                                    ],
                                },
                            ]
                        },
                    )
                if "One More Time" in q:
                    return httpx.Response(
                        200, json=_daft_punk_recording_search("One More Time", "rec-omt", 1)
                    )
                return httpx.Response(200, json={"recordings": []})
            if "/ws/2/release/rel-tlc" in request.url.path:
                return httpx.Response(
                    200,
                    json={
                        "id": "rel-tlc",
                        "title": "CrazySexyCool",
                        "date": "1994-11-15",
                        "artist-credit": [
                            {"name": "TLC", "artist": {"id": "a-tlc", "name": "TLC"}}
                        ],
                        "release-group": {
                            "id": "rg-tlc",
                            "title": "CrazySexyCool",
                            "first-release-date": "1994-11-15",
                        },
                        "media": [
                            {
                                "position": 1,
                                "track-count": 16,
                                "tracks": [
                                    {
                                        "position": 3,
                                        "number": "3",
                                        "title": "Creep",
                                        "recording": {"id": "rec-creep-2", "title": "Creep"},
                                    }
                                ],
                            }
                        ],
                    },
                )
            if "/ws/2/release/rel-disc-orig" in request.url.path:
                return httpx.Response(200, json=_daft_punk_release_detail())
        if request.url.host == "itunes.apple.com":
            return httpx.Response(200, json={"results": []})
        return httpx.Response(404)

    cache, mb, itunes, resolver, _art = _make_online_stack(_handler, tmp_path / "cache.sqlite3")
    try:
        # Rename files cleanly so 01 parses to Creep and 02 is strictly hintless VID_0042.mp4.
        (videos / "01_ambig_Creep.mp4").rename(videos / "Creep.mp4")
        (videos / "02_VID_0042.mp4").rename(videos / "VID_0042.mp4")
        (videos / "03_unknown_jam.mp4").rename(videos / "zz_jam.mp4")
        (videos / "04_skip_me.mp4").rename(videos / "zz_skip.mp4")

        # Deterministic sort order: Creep.mp4, VID_0042.mp4, zz_jam.mp4, zz_skip.mp4.
        answers = iter(
            [
                # 1. Creep.mp4 -> pick #2 (TLC)
                "2",
                # 2. VID_0042.mp4 -> manual search -> Daft Punk / One More Time / "" -> pick #1
                "s",
                "Daft Punk",
                "One More Time",
                "",
                "1",
                # 3. zz_jam.mp4 -> custom tags
                "m",
                "Indie Band",
                "Live Jam",
                "Bootleg",
                "3",
                # 4. zz_skip.mp4 -> skip
                "k",
            ]
        )
        console = Console(file=io.StringIO(), force_terminal=False)
        callbacks = make_rich_prompt_callbacks(console=console, input_fn=lambda _: next(answers))

        out_dir = tmp_path / "out"
        media_inputs, output_root = discover_inputs([videos], output_dir=out_dir)
        cfg = load_config(config_path=tmp_path / "missing.toml", env={})
        plan = build_plan(
            media_inputs,
            output_root=output_root,
            ffprobe_path=ffprobe_path,
            config=cfg,
            resolver=resolver,
            non_interactive=False,
            prompt_callbacks=callbacks,
        )
        summary = execute_plan(plan, ffmpeg_path=ffmpeg_path, output_root=output_root)
        assert summary.exit_code == 0
        assert summary.count(RunOutcome.SUCCEEDED) == 3
        assert summary.count(RunOutcome.SKIPPED) == 1
        assert (output_root / "TLC" / "CrazySexyCool" / "03 TLC - Creep.mp3").is_file()
        assert (
            output_root / "Daft Punk" / "Discovery" / "01 Daft Punk - One More Time.mp3"
        ).is_file()
        assert (output_root / "Indie Band" / "Bootleg" / "03 Indie Band - Live Jam.mp3").is_file()
    finally:
        mb.fetcher.close()
        mb.art_fetcher.close()
        itunes.fetcher.close()
        itunes.art_fetcher.close()
        cache.close()


def test_always_prompt_honors_user_selected_edition(tmp_path: Path, ffprobe_path: Path) -> None:
    clip = make_video_with_audio(tmp_path / "Daft Punk - One More Time.mp4")

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "musicbrainz.org":
            if "/ws/2/recording/" in request.url.path:
                return httpx.Response(
                    200,
                    json={
                        "recordings": [
                            {
                                "id": "rec-omt",
                                "title": "One More Time",
                                "artist-credit": [
                                    {
                                        "name": "Daft Punk",
                                        "artist": {"id": "art-dp", "name": "Daft Punk"},
                                    }
                                ],
                                "releases": [
                                    {
                                        "id": "rel-orig",
                                        "title": "Discovery",
                                        "status": "Official",
                                        "date": "2001-03-12",
                                        "country": "FR",
                                        "count": 1,
                                        "track-count": 14,
                                        "release-group": {
                                            "id": "rg-disc",
                                            "title": "Discovery",
                                            "primary-type": "Album",
                                        },
                                        "media": [
                                            {"position": 1, "format": "CD", "track-count": 14}
                                        ],
                                    },
                                    {
                                        "id": "rel-deluxe",
                                        "title": "Discovery",
                                        "disambiguation": "Deluxe Edition",
                                        "status": "Official",
                                        "date": "2011-05-01",
                                        "country": "US",
                                        "count": 2,
                                        "track-count": 20,
                                        "release-group": {
                                            "id": "rg-disc",
                                            "title": "Discovery",
                                            "primary-type": "Album",
                                        },
                                        "media": [
                                            {"position": 1, "format": "CD", "track-count": 14}
                                        ],
                                    },
                                ],
                            }
                        ]
                    },
                )
            if "/ws/2/release/rel-orig" in request.url.path:
                return httpx.Response(200, json=_daft_punk_release_detail())
            if "/ws/2/release/rel-deluxe" in request.url.path:
                return httpx.Response(
                    200,
                    json={
                        "id": "rel-deluxe",
                        "title": "Discovery",
                        "date": "2011-05-01",
                        "artist-credit": [
                            {"name": "Daft Punk", "artist": {"id": "art-dp", "name": "Daft Punk"}}
                        ],
                        "release-group": {
                            "id": "rg-disc",
                            "title": "Discovery",
                            "first-release-date": "2001-03-12",
                            "genres": [{"name": "french house", "count": 10}],
                        },
                        "media": [
                            {
                                "position": 1,
                                "track-count": 14,
                                "tracks": [
                                    {
                                        "position": 1,
                                        "number": "1",
                                        "title": "One More Time",
                                        "recording": {"id": "rec-omt", "title": "One More Time"},
                                    }
                                ],
                            },
                            {"position": 2, "track-count": 6, "tracks": []},
                        ],
                    },
                )
        return httpx.Response(404)

    cache, mb, itunes, resolver, _art = _make_online_stack(_handler, tmp_path / "cache.sqlite3")
    try:
        # Press Enter to confirm #1 candidate, then enter "2" to pick the 2-disc Deluxe edition.
        answers = iter(["", "2"])
        console = Console(file=io.StringIO(), force_terminal=False)
        callbacks = make_rich_prompt_callbacks(console=console, input_fn=lambda _: next(answers))

        out_dir = tmp_path / "out"
        media_inputs, output_root = discover_inputs([clip], output_dir=out_dir)
        cfg = load_config(config_path=tmp_path / "missing.toml", env={})
        plan = build_plan(
            media_inputs,
            output_root=output_root,
            ffprobe_path=ffprobe_path,
            config=cfg,
            resolver=resolver,
            always_prompt=True,
            non_interactive=False,
            prompt_callbacks=callbacks,
        )
        meta = plan[0].metadata
        assert meta is not None
        assert meta.mb_release_id == "rel-deluxe"
        assert meta.date == "2011-05-01"
        assert meta.original_year == "2001"
        assert meta.disc_total == 2
        assert plan[0].output_path == (
            output_root / "Daft Punk" / "Discovery" / "1-01 Daft Punk - One More Time.mp3"
        )
    finally:
        mb.fetcher.close()
        mb.art_fetcher.close()
        itunes.fetcher.close()
        itunes.art_fetcher.close()
        cache.close()


def test_interactive_multi_track_stream_selection_prompt(
    tmp_path: Path, ffprobe_path: Path
) -> None:
    mkv = make_multi_track_mkv(tmp_path / "Daft Punk - One More Time.mkv")
    answers = iter(["2"])  # Stream 1 = Japanese AAC (default), Stream 2 = English FLAC!
    console = Console(file=io.StringIO(), force_terminal=False)
    callbacks = make_rich_prompt_callbacks(console=console, input_fn=lambda _: next(answers))

    out_dir = tmp_path / "out"
    media_inputs, output_root = discover_inputs([mkv], output_dir=out_dir)
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    plan = build_offline_plan(
        media_inputs,
        output_root=output_root,
        ffprobe_path=ffprobe_path,
        config=cfg,
        prompt_callbacks=callbacks,
    )
    assert plan[0].selected_stream is not None
    assert plan[0].selected_stream.language == "eng"
    assert plan[0].selected_stream.codec_name == "flac"


def test_scoped_non_latin_alias_prompts_a_k_A_K_and_keep_native_missing(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    tracks = [
        ("01.mp4", "張學友", "吻別", "吻別", "1"),
        ("02.mp4", "張學友", "吻別", "一路上有你", "2"),
        ("03.mp4", "張學友", "吻別", "秋意濃", "3"),
        ("04.mp4", "王菲", "天空", "天空", "1"),
        ("05.mp4", "宇多田ヒカル", "First Love", "初恋", "1"),
    ]
    for fname, artist, album, title, trk in tracks:
        make_video_with_audio(
            videos / fname,
            metadata={
                "artist": artist,
                "album_artist": artist,
                "album": album,
                "title": title,
                "track": trk,
            },
        )

    # Scripted responses across all 5 tracks:
    # 1) Track 1 artist '張學友' -> [a] 'Jacky Cheung' (remembered for tracks 1-3 & album_artist!)
    # 2) Track 1 album '吻別' -> [a] 'The Goodbye Kiss' (remembered for tracks 1-3, and reused
    #    for the title track '吻別' without asking again!)
    # 3) Track 2 title '一路上有你' -> [A] keep native for album '吻別' (skips Track 3 title too!)
    # 4) Track 4 artist '王菲' -> [K] keep native for rest of run (skips Track 4 & Track 5!)
    scripted = iter(["a", "Jacky Cheung", "a", "The Goodbye Kiss", "A", "K"])
    prompts_seen: list[str] = []

    def _read(prompt: str) -> str:
        prompts_seen.append(prompt)
        return next(scripted)

    console = Console(file=io.StringIO(), force_terminal=False)
    callbacks = make_rich_prompt_callbacks(console=console, input_fn=_read)

    out_dir = tmp_path / "out"
    media_inputs, output_root = discover_inputs([videos], output_dir=out_dir)
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})

    plan = build_plan(
        media_inputs,
        output_root=output_root,
        ffprobe_path=ffprobe_path,
        config=cfg,
        resolver=None,
        non_interactive=False,
        keep_native_missing=False,
        prompt_callbacks=callbacks,
    )
    # Only 6 input reads total (4 choices + 2 English alias strings), zero redundant prompts!
    assert len(prompts_seen) == 6

    summary = execute_plan(plan, ffmpeg_path=ffmpeg_path, output_root=output_root, jobs=2)
    assert summary.exit_code == 0

    t1 = (
        output_root
        / "Jacky Cheung {張學友}"
        / "The Goodbye Kiss {吻別}"
        / "01 Jacky Cheung {張學友} - The Goodbye Kiss {吻別}.mp3"
    )
    t2 = (
        output_root
        / "Jacky Cheung {張學友}"
        / "The Goodbye Kiss {吻別}"
        / "02 Jacky Cheung {張學友} - 一路上有你.mp3"
    )
    t3 = (
        output_root
        / "Jacky Cheung {張學友}"
        / "The Goodbye Kiss {吻別}"
        / "03 Jacky Cheung {張學友} - 秋意濃.mp3"
    )
    t4 = output_root / "王菲" / "天空" / "01 王菲 - 天空.mp3"
    t5 = output_root / "宇多田ヒカル" / "First Love" / "01 宇多田ヒカル - 初恋.mp3"
    for expected in (t1, t2, t3, t4, t5):
        assert expected.is_file(), f"Missing expected output {expected}"

    # Verify ID3 tags match bilingual formatting consistently!
    t1_tags = ID3(str(t1))
    assert t1_tags["TPE1"].text == ["Jacky Cheung {張學友}"]
    assert t1_tags["TPE2"].text == ["Jacky Cheung {張學友}"]
    assert t1_tags["TALB"].text == ["The Goodbye Kiss {吻別}"]
    assert t1_tags["TIT2"].text == ["The Goodbye Kiss {吻別}"]

    # --keep-native-missing makes ZERO alias prompts and keeps native script as-is!
    called_native = 0

    def _fail_if_prompted(_prompt: str) -> str:
        nonlocal called_native
        called_native += 1
        raise AssertionError("Alias prompt must not fire under --keep-native-missing")

    out_native = tmp_path / "out_native"
    plan_native = build_plan(
        media_inputs,
        output_root=out_native,
        ffprobe_path=ffprobe_path,
        config=cfg,
        resolver=None,
        non_interactive=False,
        keep_native_missing=True,
        prompt_callbacks=make_rich_prompt_callbacks(console=console, input_fn=_fail_if_prompted),
    )
    assert called_native == 0
    assert plan_native[0].output_path == out_native / "張學友" / "吻別" / "01 張學友 - 吻別.mp3"


def test_folder_casing_canonicalization_and_long_output_root_warning(
    tmp_path: Path, ffprobe_path: Path
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    make_video_with_audio(
        videos / "a.mp4",
        metadata={
            "artist": "Daft Punk",
            "album": "Discovery",
            "title": "One More Time",
            "track": "1",
        },
    )
    make_video_with_audio(
        videos / "b.mp4",
        metadata={
            "artist": "daft punk",
            "album": "discovery",
            "title": "Aerodynamic",
            "track": "2",
        },
    )

    out_dir = tmp_path / "out"
    media_inputs, output_root = discover_inputs([videos], output_dir=out_dir)
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    plan = build_offline_plan(
        media_inputs, output_root=output_root, ffprobe_path=ffprobe_path, config=cfg
    )
    assert plan[0].output_path is not None and plan[1].output_path is not None
    # Both tracks canonicalize into the exact same folder casing ("Daft Punk/Discovery")!
    assert plan[0].output_path.parent == output_root / "Daft Punk" / "Discovery"
    assert plan[1].output_path.parent == output_root / "Daft Punk" / "Discovery"

    # Existing lowercase file on disk collides case-insensitively under skip & rename!
    existing = output_root / "daft punk" / "discovery" / "01 daft punk - one more time.mp3"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"existing")

    plan_skip = build_offline_plan(
        media_inputs,
        output_root=output_root,
        ffprobe_path=ffprobe_path,
        config=cfg,
        on_conflict=ConflictPolicy.SKIP,
    )
    assert plan_skip[0].action == PlanAction.SKIP_EXISTING
    assert plan_skip[1].action == PlanAction.EXTRACT

    plan_rename = build_offline_plan(
        media_inputs,
        output_root=output_root,
        ffprobe_path=ffprobe_path,
        config=cfg,
        on_conflict=ConflictPolicy.RENAME,
    )
    assert plan_rename[0].output_path is not None
    assert plan_rename[0].output_path.name == "01 Daft Punk - One More Time (2).mp3"

    # Long output_root (>= 180 UTF-16 units) triggers batch warning recommending shorter -o.
    long_root = tmp_path / ("very_long_segment_" * 11) / "out"
    batch_warnings: list[str] = []
    build_offline_plan(
        media_inputs,
        output_root=long_root,
        ffprobe_path=ffprobe_path,
        config=cfg,
        on_batch_warning=batch_warnings.append,
    )
    assert len(batch_warnings) == 1
    assert "-o/--output-dir" in batch_warnings[0]


def test_online_provider_outage_degrades_gracefully_with_warnings(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    make_video_with_audio(videos / "Daft Punk - One More Time.mp4")
    make_video_with_audio(videos / "Daft Punk - Aerodynamic.mp4")
    make_video_with_audio(videos / "Daft Punk - Digital Love.mp4")

    def _failing_handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "musicbrainz.org":
            return httpx.Response(503)
        raise httpx.ConnectError("network unreachable")

    cache, mb, itunes, resolver, _art = _make_online_stack(
        _failing_handler, tmp_path / "cache.sqlite3"
    )
    try:
        out_dir = tmp_path / "out"
        media_inputs, output_root = discover_inputs([videos], output_dir=out_dir)
        cfg = load_config(config_path=tmp_path / "missing.toml", env={})
        plan = build_plan(
            media_inputs,
            output_root=output_root,
            ffprobe_path=ffprobe_path,
            config=cfg,
            resolver=resolver,
            non_interactive=True,
        )
        summary = execute_plan(plan, ffmpeg_path=ffmpeg_path, output_root=output_root)
        assert summary.exit_code == 0
        assert summary.count(RunOutcome.SUCCEEDED) == 3
        all_warnings = [w for r in summary.results for w in r.warnings]
        assert any("unreachable" in w.lower() or "failed" in w.lower() for w in all_warnings)
    finally:
        mb.fetcher.close()
        mb.art_fetcher.close()
        itunes.fetcher.close()
        itunes.art_fetcher.close()
        cache.close()


def test_prompt_mb_contact_reprompts_on_placeholder_and_cli_saves_in_tty(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # 1. Unit test prompt_mb_contact rejects example.com then accepts valid relay.
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)
    inputs = iter(["user@example.com", "relay@duck.com"])
    got = prompt_mb_contact(console=console, input_fn=lambda _: next(inputs))
    assert got == "relay@duck.com"
    assert "Invalid contact" in buf.getvalue()

    # 2. Full CLI TTY run with missing contact -> prompts once -> saves config.toml -> --dry-run!
    clip = make_video_with_audio(tmp_path / "Daft Punk - One More Time.mp4")
    cfg_path = tmp_path / "cfg" / "config.toml"
    out_dir = tmp_path / "out"
    caa_hits = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal caa_hits
        if request.url.host == "coverartarchive.org":
            caa_hits += 1
            return httpx.Response(200, content=_make_jpeg_bytes())
        if request.url.host == "musicbrainz.org":
            if "/ws/2/recording/" in request.url.path:
                return httpx.Response(
                    200, json=_daft_punk_recording_search("One More Time", "rec-omt", 1)
                )
            if "/ws/2/release/rel-disc-orig" in request.url.path:
                return httpx.Response(200, json=_daft_punk_release_detail())
        return httpx.Response(404)

    def _mock_services(user_agent: str, cfg: Any, cache: TTLCache) -> Any:
        transport = httpx.MockTransport(_handler)
        mb = MusicBrainzClient(
            user_agent,
            fetcher=HttpFetcher(transport=transport, limiter=RateLimiter(0.0), max_retries=0),
            art_fetcher=HttpFetcher(transport=transport, limiter=RateLimiter(0.0), max_retries=0),
            cache=cache,
        )
        it = ITunesClient(
            fetcher=HttpFetcher(transport=transport, limiter=RateLimiter(0.0), max_retries=0),
            art_fetcher=HttpFetcher(transport=transport, limiter=RateLimiter(0.0), max_retries=0),
            cache=cache,
        )
        return mb, it, TrackResolver(mb, it, cfg.matching), ArtworkFetcher(mb, it, cache=cache)

    monkeypatch.delenv("AFV_MB_CONTACT", raising=False)
    monkeypatch.delenv("AFV_MB_USER_AGENT", raising=False)
    monkeypatch.setenv("AFV_CONFIG_PATH", str(cfg_path))
    monkeypatch.setattr(cli_main, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr(cli_main, "prompt_mb_contact", lambda **_kw: "relay@duck.com")
    monkeypatch.setattr(cli_main, "_create_online_services", _mock_services)
    monkeypatch.setattr(cli_main, "default_cache_dir", lambda: tmp_path / "cache")

    cli_runner = CliRunner()
    res = cli_runner.invoke(cli_main.app, ["--dry-run", "-o", str(out_dir), str(clip)])
    assert res.exit_code == 0, res.output
    assert cfg_path.is_file()
    assert load_config(config_path=cfg_path, env={}).musicbrainz.contact == "relay@duck.com"
    assert caa_hits == 0  # Zero artwork downloads in --dry-run!
    assert not out_dir.exists()


def test_cli_subprocess_normalize_report_and_plain_log_file(tmp_path: Path) -> None:
    clip = make_video_with_audio(
        tmp_path / "Daft Punk - One More Time.mp4",
        duration=0.8,
        metadata={
            "artist": "Daft Punk",
            "title": "One More Time",
            "album": "Discovery",
            "track": "1",
        },
    )
    out_dir = tmp_path / "out"
    report_path = tmp_path / "norm_report.json"
    log_path = tmp_path / "norm.log"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "audio_from_video.cli.main",
            "--offline",
            "--normalize",
            "-o",
            str(out_dir),
            "--report",
            str(report_path),
            "--log-file",
            str(log_path),
            str(clip),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    assert rep["exit_code"] == 0
    assert rep["counts"]["succeeded"] == 1
    norm = rep["tracks"][0]["normalization"]
    assert norm is not None
    assert norm["target_i"] == -16.0
    assert norm["normalization_type"] in ("linear", "linear_volume_fallback")
    assert norm["gain_db"] is not None

    log_text = log_path.read_text(encoding="utf-8")
    assert "[DEBUG]" in log_text
    assert not _ANSI_RE.search(log_text)
    assert list(out_dir.rglob("*.part")) == []


def test_normalize_interrupt_returns_130_and_cleans_up_part_files(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    clip = make_video_with_audio(
        tmp_path / "Daft Punk - One More Time.mp4",
        metadata={"artist": "Daft Punk", "title": "One More Time", "album": "Discovery"},
    )
    out_dir = tmp_path / "out"
    media_inputs, output_root = discover_inputs([clip], output_dir=out_dir)
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    plan = build_offline_plan(
        media_inputs,
        output_root=output_root,
        ffprobe_path=ffprobe_path,
        config=cfg,
        normalize=True,
    )
    registry = ProcessRegistry()
    registry.terminate_all(timeout=0.5)
    summary = execute_plan(
        plan,
        ffmpeg_path=ffmpeg_path,
        output_root=output_root,
        registry=registry,
    )
    assert summary.interrupted is True
    assert summary.exit_code == 130
    assert list(output_root.rglob("*.part")) == []


def test_live_subprocess_sigint_exits_130_and_removes_part_files(tmp_path: Path) -> None:
    """Send a real SIGINT to an active afv subprocess on POSIX (or verify live kill on Windows)."""
    if sys.platform == "win32":
        # Windows lacks POSIX SIGINT delivery to non-console children; verify active
        # ProcessRegistry termination directly.
        reg = ProcessRegistry()
        part = tmp_path / ".afv-1234567890ab.part"
        part.write_bytes(b"in-flight")
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert reg.register_process(child, part) is True
        reg.terminate_all(timeout=1.0)
        assert child.poll() is not None
        assert not part.exists()
        return

    # Create a batch of several videos so extraction is guaranteed to be running when SIGINT lands.
    videos = tmp_path / "batch"
    videos.mkdir()
    for i in range(1, 10):
        make_video_with_audio(
            videos / f"{i:02d} Artist - Track {i:02d}.mp4",
            duration=2.5,
            metadata={
                "artist": "Artist",
                "title": f"Track {i:02d}",
                "album": "Album",
                "track": str(i),
            },
        )
    out_dir = tmp_path / "out"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "audio_from_video.cli.main",
            "--offline",
            "--normalize",
            "-j",
            "1",
            "-o",
            str(out_dir),
            str(videos),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Wait until Stage 2 starts writing into out_dir, then send real SIGINT.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and proc.poll() is None:
            if out_dir.exists() and (list(out_dir.rglob("*.part")) or list(out_dir.rglob("*.mp3"))):
                break
            time.sleep(0.05)
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGINT)
        stdout, stderr = proc.communicate(timeout=15.0)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)

    assert proc.returncode == 130, (
        f"Expected exit code 130 on SIGINT, got {proc.returncode}: "
        f"{stderr.decode('utf-8', errors='replace')}\n{stdout.decode('utf-8', errors='replace')}"
    )
    assert list(out_dir.rglob("*.part")) == []
