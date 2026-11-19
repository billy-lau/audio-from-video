"""Offline end-to-end tests for planner, runner, conflicts, no-audio, and CLI."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from mutagen.id3 import ID3, ID3NoHeaderError
from mutagen.mp3 import MP3

from audio_from_video.config import load_config
from audio_from_video.discovery import discover_inputs
from audio_from_video.models import (
    ConflictPolicy,
    ManualOverrides,
    MatchStatus,
    PlanAction,
    RunOutcome,
)
from audio_from_video.planner import build_offline_plan
from audio_from_video.runner import execute_plan, write_json_report
from conftest import make_silent_video_no_audio, make_video_with_audio, make_video_with_chapters

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def test_offline_end_to_end_exact_partial_unmatched_and_downloader_safety(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()

    # 1. Exact embedded tags (Roxy Music positive control + chapters).
    make_video_with_audio(
        videos / "roxy.mp4",
        metadata={
            "artist": "Roxy Music",
            "album_artist": "Roxy Music",
            "title": "More Than This - 2012 Remaster",
            "album": "Avalon",
            "track": "1/10",
            "date": "1982",
            "genre": "Art Rock",
        },
    )

    # 2. Tagged classical positive control (must stay trusted, never split!).
    make_video_with_audio(
        videos / "classical.mp4",
        metadata={
            "artist": "Berliner Philharmoniker",
            "title": "Symphony No. 5 - Allegro con brio",
            "album": "Beethoven: Symphonies",
            "track": "1",
        },
    )

    # 3. Partial match from filename (no album -> Unknown Album/).
    make_video_with_audio(
        videos / "Daft Punk - Unreleased Demo (ft. Pharrell Williams) [Official Audio].mp4",
    )

    # 4. YouTube Music " - Topic" (stripped -> trusted artist!).
    make_video_with_audio(
        videos / "topic.mkv",
        audio_codec="libmp3lame",
        bitrate="192k",
        metadata={
            "artist": "Daft Punk - Topic",
            "title": "One More Time",
            "album": "Discovery",
            "track": "1",
            "purl": "https://www.youtube.com/watch?v=FGBhQbmPwH8",
        },
    )

    # 5. Untrusted VEVO / URL uploader (must NOT create DaftPunkVEVO/Unknown Album/!).
    make_video_with_audio(
        videos / "VID_9999.mp4",
        metadata={
            "artist": "DaftPunkVEVO",
            "title": "Behind the Scenes",
            "comment": "https://youtu.be/abc12345678",
        },
    )

    # 6. Completely hintless video (no tags -> _unmatched/VID_0042.mp3, zero ID3 frames).
    make_video_with_audio(videos / "VID_0042.mp4")

    # 7. Audio-less video inside folder scan -> skipped_no_audio (exit code 0!).
    make_silent_video_no_audio(videos / "silent_banner.mp4")

    # 8. Video with chapters -> stripped.
    make_video_with_chapters(videos / "AC-DC - Hells Bells.mkv")

    media_inputs, output_root = discover_inputs([videos])
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})

    plan_items = build_offline_plan(
        media_inputs,
        output_root=output_root,
        ffprobe_path=ffprobe_path,
        config=cfg,
    )
    summary = execute_plan(
        plan_items,
        ffmpeg_path=ffmpeg_path,
        output_root=output_root,
        jobs=2,
    )

    assert summary.exit_code == 0
    assert summary.count(RunOutcome.FAILED) == 0
    assert summary.count(RunOutcome.SKIPPED_NO_AUDIO) == 1
    assert summary.count(RunOutcome.UNMATCHED) == 2
    assert summary.count(RunOutcome.SUCCEEDED) == 5

    # Verify paths & ID3v2.3 tags.
    roxy_mp3 = (
        output_root / "Roxy Music" / "Avalon" / "01 Roxy Music - More Than This - 2012 Remaster.mp3"
    )
    assert roxy_mp3.is_file()
    roxy_tags = ID3(str(roxy_mp3), translate=False)
    assert roxy_tags.version == (2, 3, 0)
    assert roxy_tags["TPE1"].text == ["Roxy Music"]
    assert str(roxy_tags["TYER"]) == "1982"

    # Tagged classical positive control stays trusted in its Artist/Album folder (Phase C may
    # trim the filename stem when pytest's tmp_path is deep, while ID3 TIT2 remains verbatim).
    classical_res = next(
        r for r in summary.results if r.item.media_input.path.name == "classical.mp4"
    )
    assert classical_res.output_path is not None and classical_res.output_path.is_file()
    assert classical_res.output_path.parent == (
        output_root / "Berliner Philharmoniker" / "Beethoven - Symphonies"
    )
    assert classical_res.output_path.name.startswith("01 Berliner Philharmoniker - Symphony No. 5")
    classical_tags = ID3(str(classical_res.output_path))
    assert classical_tags["TPE1"].text == ["Berliner Philharmoniker"]
    assert classical_tags["TIT2"].text == ["Symphony No. 5 - Allegro con brio"]

    partial_mp3 = (
        output_root
        / "Daft Punk"
        / "Unknown Album"
        / "Daft Punk (ft. Pharrell Williams) - Unreleased Demo.mp3"
    )
    assert partial_mp3.is_file()
    p_tags = ID3(str(partial_mp3))
    assert p_tags["TPE1"].text == ["Daft Punk"]
    assert p_tags["TIT2"].text == ["Unreleased Demo (ft. Pharrell Williams)"]
    assert "TALB" not in p_tags

    topic_mp3 = output_root / "Daft Punk" / "Discovery" / "01 Daft Punk - One More Time.mp3"
    assert topic_mp3.is_file()
    # Stream-copied from 192k source (never re-encoded up to 320k).
    assert abs(MP3(str(topic_mp3)).info.bitrate - 192000) < 2000

    # Untrusted VEVO uploader must NOT create a DaftPunkVEVO folder!
    assert not (output_root / "DaftPunkVEVO").exists()
    assert (output_root / "_unmatched" / "VID_9999.mp3").is_file()

    # Hintless VID_0042 -> _unmatched/VID_0042.mp3 with zero stale ID3 tags.
    unmatched_mp3 = output_root / "_unmatched" / "VID_0042.mp3"
    assert unmatched_mp3.is_file()
    assert MP3(str(unmatched_mp3)).info.sample_rate > 0
    try:
        ID3(str(unmatched_mp3))
        raise AssertionError("Expected zero ID3 header on completely hintless unmatched track")
    except ID3NoHeaderError:
        pass

    # Verify JSON report generation.
    report_path = tmp_path / "report.json"
    write_json_report(summary, report_path)
    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_data["exit_code"] == 0
    assert report_data["counts"]["succeeded"] == 5
    assert report_data["counts"]["unmatched"] == 2
    assert report_data["counts"]["skipped_no_audio"] == 1

    # No stray .afv-*.part files left behind.
    assert list(output_root.rglob("*.part")) == []


def test_explicit_audio_less_video_fails_with_nonzero_exit(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    silent = make_silent_video_no_audio(tmp_path / "silent.mp4")
    media_inputs, output_root = discover_inputs([silent])
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})

    plan = build_offline_plan(
        media_inputs, output_root=output_root, ffprobe_path=ffprobe_path, config=cfg
    )
    assert plan[0].action == PlanAction.FAIL_NO_AUDIO

    summary = execute_plan(plan, ffmpeg_path=ffmpeg_path, output_root=output_root)
    assert summary.count(RunOutcome.FAILED) == 1
    assert summary.exit_code == 1


def test_deterministic_intra_batch_collision_and_on_conflict_policies(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    src = tmp_path / "batch"
    src.mkdir()
    # Two different video files with identical embedded metadata collide in-batch.
    for name in ("clip_b.mp4", "clip_a.mp4"):
        make_video_with_audio(
            src / name,
            metadata={
                "artist": "Artist",
                "title": "Same Song",
                "album": "Same Album",
                "track": "1",
            },
        )

    out_dir = tmp_path / "out"
    media_inputs, output_root = discover_inputs([src], output_dir=out_dir)
    assert [m.path.name for m in media_inputs] == ["clip_a.mp4", "clip_b.mp4"]

    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    plan_1 = build_offline_plan(
        media_inputs,
        output_root=output_root,
        ffprobe_path=ffprobe_path,
        config=cfg,
        on_conflict=ConflictPolicy.SKIP,
    )
    dests_1 = [p.output_path.name for p in plan_1 if p.output_path is not None]
    assert dests_1 == ["01 Artist - Same Song.mp3", "01 Artist - Same Song (2).mp3"]

    res_1 = execute_plan(plan_1, ffmpeg_path=ffmpeg_path, output_root=output_root)
    assert res_1.exit_code == 0
    assert (output_root / "Artist" / "Same Album" / "01 Artist - Same Song.mp3").is_file()
    assert (output_root / "Artist" / "Same Album" / "01 Artist - Same Song (2).mp3").is_file()

    # Re-run with --on-conflict skip -> both skipped in Stage 1 (even before --dry-run!).
    plan_skip = build_offline_plan(
        media_inputs,
        output_root=output_root,
        ffprobe_path=ffprobe_path,
        config=cfg,
        on_conflict=ConflictPolicy.SKIP,
    )
    assert [p.action for p in plan_skip] == [PlanAction.SKIP_EXISTING, PlanAction.SKIP_EXISTING]

    # Re-run with --on-conflict rename -> creates (3) and (4).
    plan_rename = build_offline_plan(
        media_inputs,
        output_root=output_root,
        ffprobe_path=ffprobe_path,
        config=cfg,
        on_conflict=ConflictPolicy.RENAME,
    )
    assert [p.output_path.name for p in plan_rename if p.output_path is not None] == [
        "01 Artist - Same Song (3).mp3",
        "01 Artist - Same Song (4).mp3",
    ]

    # Re-run with --on-conflict overwrite -> reuses exact (1) and (2) names.
    plan_over = build_offline_plan(
        media_inputs,
        output_root=output_root,
        ffprobe_path=ffprobe_path,
        config=cfg,
        on_conflict=ConflictPolicy.OVERWRITE,
    )
    assert [p.action for p in plan_over] == [PlanAction.EXTRACT, PlanAction.EXTRACT]
    assert [p.output_path.name for p in plan_over if p.output_path is not None] == [
        "01 Artist - Same Song.mp3",
        "01 Artist - Same Song (2).mp3",
    ]


def test_fail_fast_stops_after_first_failure(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    silent = make_silent_video_no_audio(tmp_path / "a_silent.mp4")
    good = make_video_with_audio(
        tmp_path / "b_good.mp4",
        metadata={"artist": "Daft Punk", "title": "One More Time", "album": "Discovery"},
    )
    media_inputs, output_root = discover_inputs([silent, good])
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    plan = build_offline_plan(
        media_inputs, output_root=output_root, ffprobe_path=ffprobe_path, config=cfg
    )
    summary = execute_plan(
        plan, ffmpeg_path=ffmpeg_path, output_root=output_root, jobs=2, fail_fast=True
    )
    assert summary.exit_code == 1
    assert len(summary.results) == 1
    assert summary.results[0].outcome == RunOutcome.FAILED
    assert not (output_root / "Daft Punk").exists()


def test_real_subprocess_cli_offline_end_to_end_and_plain_log_file(tmp_path: Path) -> None:
    clip = make_video_with_audio(
        tmp_path / "clip.mp4",
        audio_codec="aac",
        sample_rate=44100,
    )
    out_dir = tmp_path / "out"
    log_file = tmp_path / "run.log"
    report_file = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "audio_from_video.cli.main",
            "--offline",
            "--artist",
            "Daft Punk",
            "--album",
            "Discovery",
            "--title",
            "One More Time",
            "--track",
            "1",
            "--flat",
            "-o",
            str(out_dir),
            "--log-file",
            str(log_file),
            "--report",
            str(report_file),
            "-vv",
            str(clip),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout

    expected = out_dir / "01 Daft Punk - One More Time.mp3"
    assert expected.is_file()
    tags = ID3(str(expected))
    assert tags.version == (2, 3, 0)
    assert tags["TPE1"].text == ["Daft Punk"]
    assert tags["TIT2"].text == ["One More Time"]
    assert tags["TALB"].text == ["Discovery"]
    assert tags["TRCK"].text == ["1"]

    # Source video must remain strictly untouched.
    assert clip.is_file()

    # Plain UTF-8 log file with zero ANSI escape sequences.
    log_text = log_file.read_text(encoding="utf-8")
    assert "[DEBUG]" in log_text
    assert not _ANSI_RE.search(log_text)

    # JSON report matches.
    data = json.loads(report_file.read_text(encoding="utf-8"))
    assert data["counts"]["succeeded"] == 1
    assert data["exit_code"] == 0


def test_manual_overrides_pinned_verbatim_on_batch(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    src = tmp_path / "album_videos"
    src.mkdir()
    make_video_with_audio(src / "01 - Ignored Artist - Song One.mp4")
    make_video_with_audio(src / "02 - Ignored Artist - Song Two.mp4")

    media_inputs, output_root = discover_inputs([src])
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    plan = build_offline_plan(
        media_inputs,
        output_root=output_root,
        ffprobe_path=ffprobe_path,
        config=cfg,
        overrides=ManualOverrides(artist="Pinned Artist", album="Pinned Album"),
    )
    summary = execute_plan(plan, ffmpeg_path=ffmpeg_path, output_root=output_root, jobs=2)
    assert summary.exit_code == 0
    assert (
        output_root / "Pinned Artist" / "Pinned Album" / "01 Pinned Artist - Song One.mp3"
    ).is_file()
    assert (
        output_root / "Pinned Artist" / "Pinned Album" / "02 Pinned Artist - Song Two.mp3"
    ).is_file()


def test_runner_interrupt_returns_exit_code_130_and_leaves_no_part_files(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    from audio_from_video.extractor import ProcessRegistry

    clip = make_video_with_audio(
        tmp_path / "01 Daft Punk - One More Time.mp4",
        metadata={"artist": "Daft Punk", "title": "One More Time", "album": "Discovery"},
    )
    media_inputs, output_root = discover_inputs([clip])
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    plan = build_offline_plan(
        media_inputs, output_root=output_root, ffprobe_path=ffprobe_path, config=cfg
    )

    registry = ProcessRegistry()
    registry.terminate_all(timeout=0.5)

    summary = execute_plan(
        plan,
        ffmpeg_path=ffmpeg_path,
        output_root=output_root,
        jobs=1,
        registry=registry,
    )
    assert summary.interrupted is True
    assert summary.exit_code == 130
    assert list(output_root.rglob("*.part")) == []


def test_runner_toctou_recheck_handles_concurrent_destination_creation(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path, monkeypatch: object
) -> None:
    from audio_from_video import runner as runner_mod

    clip = make_video_with_audio(
        tmp_path / "01 Artist - Song.mp4",
        metadata={"artist": "Artist", "title": "Song", "album": "Album", "track": "1"},
    )
    out_skip = tmp_path / "out_skip"
    media_inputs, _ = discover_inputs([clip], output_dir=out_skip)
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    plan_skip = build_offline_plan(
        media_inputs,
        output_root=out_skip,
        ffprobe_path=ffprobe_path,
        config=cfg,
        on_conflict=ConflictPolicy.SKIP,
    )
    target_skip = plan_skip[0].output_path
    assert target_skip is not None

    real_write_tags = runner_mod.write_tags

    def _racy_write_tags(
        mp3_path: Path, meta: object, *, artwork_bytes: bytes | None = None
    ) -> tuple[bool, list[str]]:
        target_skip.parent.mkdir(parents=True, exist_ok=True)
        target_skip.write_bytes(b"preexisting-concurrent")
        return real_write_tags(mp3_path, meta, artwork_bytes=artwork_bytes)  # type: ignore[arg-type]

    monkeypatch.setattr(runner_mod, "write_tags", _racy_write_tags)  # type: ignore[attr-defined]
    summary_skip = execute_plan(
        plan_skip,
        ffmpeg_path=ffmpeg_path,
        output_root=out_skip,
        on_conflict=ConflictPolicy.SKIP,
    )
    assert summary_skip.count(RunOutcome.SKIPPED) == 1
    assert target_skip.read_bytes() == b"preexisting-concurrent"
    assert list(out_skip.rglob("*.part")) == []

    out_rename = tmp_path / "out_rename"
    plan_rename = build_offline_plan(
        media_inputs,
        output_root=out_rename,
        ffprobe_path=ffprobe_path,
        config=cfg,
        on_conflict=ConflictPolicy.RENAME,
    )
    target_rename = plan_rename[0].output_path
    assert target_rename is not None

    def _racy_write_tags_rename(
        mp3_path: Path, meta: object, *, artwork_bytes: bytes | None = None
    ) -> tuple[bool, list[str]]:
        target_rename.parent.mkdir(parents=True, exist_ok=True)
        target_rename.write_bytes(b"preexisting-concurrent")
        return real_write_tags(mp3_path, meta, artwork_bytes=artwork_bytes)  # type: ignore[arg-type]

    monkeypatch.setattr(runner_mod, "write_tags", _racy_write_tags_rename)  # type: ignore[attr-defined]
    summary_rename = execute_plan(
        plan_rename,
        ffmpeg_path=ffmpeg_path,
        output_root=out_rename,
        on_conflict=ConflictPolicy.RENAME,
    )
    assert summary_rename.count(RunOutcome.SUCCEEDED) == 1
    assert target_rename.read_bytes() == b"preexisting-concurrent"
    renamed = target_rename.with_name("01 Artist - Song (2).mp3")
    assert renamed.is_file()
    assert list(out_rename.rglob("*.part")) == []


def test_parent_folder_hints_used_for_untagged_artist_album_layout(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    # 1. Artist/Album/02 Aerodynamic.mp4 (zero container tags, no dash in filename).
    album_dir = tmp_path / "Daft Punk" / "Discovery"
    album_dir.mkdir(parents=True)
    make_video_with_audio(album_dir / "02 Aerodynamic.mp4")

    # 2. Artist-only folder under Music/ (must NOT turn "Daft Punk" into an album name!).
    artist_only = tmp_path / "Music" / "Daft Punk"
    artist_only.mkdir(parents=True)
    make_video_with_audio(artist_only / "Daft Punk - Loose Single.mp4")

    # 3. Self-titled album Artist/Artist/track.mp4 (both parent and grandparent present -> kept!).
    self_titled = tmp_path / "The Beatles" / "The Beatles"
    self_titled.mkdir(parents=True)
    make_video_with_audio(self_titled / "2-01 Birthday.mp4")

    out_dir = tmp_path / "out"
    media_inputs, output_root = discover_inputs(
        [album_dir, artist_only, self_titled], output_dir=out_dir
    )
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    plan = build_offline_plan(
        media_inputs, output_root=output_root, ffprobe_path=ffprobe_path, config=cfg
    )
    summary = execute_plan(plan, ffmpeg_path=ffmpeg_path, output_root=output_root, jobs=2)
    assert summary.exit_code == 0

    aero = output_root / "Daft Punk" / "Discovery" / "02 Daft Punk - Aerodynamic.mp3"
    assert aero.is_file()
    aero_tags = ID3(str(aero))
    assert aero_tags["TPE1"].text == ["Daft Punk"]
    assert aero_tags["TALB"].text == ["Discovery"]
    assert aero_tags["TIT2"].text == ["Aerodynamic"]
    assert aero_tags["TRCK"].text == ["2"]

    loose = output_root / "Daft Punk" / "Unknown Album" / "Daft Punk - Loose Single.mp3"
    assert loose.is_file()
    assert "TALB" not in ID3(str(loose))

    white_album = output_root / "The Beatles" / "The Beatles" / "2-01 The Beatles - Birthday.mp3"
    assert white_album.is_file()


def test_fail_fast_runs_jobs_concurrently_and_cancels_remaining_on_error(
    tmp_path: Path, monkeypatch: object
) -> None:
    import threading
    import time

    from audio_from_video import runner as runner_mod
    from audio_from_video.models import MediaInput, PlanItem, TrackResult

    out_dir = tmp_path / "out"
    items = [
        PlanItem(
            media_input=MediaInput(path=tmp_path / f"f{i}.mp4", from_directory_scan=False),
            action=PlanAction.EXTRACT,
        )
        for i in range(4)
    ]

    # 1. All 4 items succeed: barrier proves all 4 actually run concurrently under --fail-fast!
    barrier = threading.Barrier(4, timeout=3.0)
    peak_active = 0
    active = 0
    lock = threading.Lock()

    def _concurrent_ok(item: PlanItem, **_kwargs: object) -> TrackResult:
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        barrier.wait()
        with lock:
            active -= 1
        return TrackResult(item=item, outcome=RunOutcome.SUCCEEDED)

    monkeypatch.setattr(runner_mod, "_execute_single_item", _concurrent_ok)  # type: ignore[attr-defined]
    summary_ok = execute_plan(
        items, ffmpeg_path=Path("ffmpeg"), output_root=out_dir, jobs=4, fail_fast=True
    )
    assert summary_ok.exit_code == 0
    assert peak_active == 4

    # 2. 6-item batch with jobs=2 where item 1 fails mid-run: items 2..5 never execute!
    six_items = [
        PlanItem(
            media_input=MediaInput(path=tmp_path / f"g{i}.mp4", from_directory_scan=False),
            action=PlanAction.EXTRACT,
        )
        for i in range(6)
    ]
    executed: list[str] = []

    def _fail_on_second(item: PlanItem, **_kwargs: object) -> TrackResult:
        name = item.media_input.path.name
        with lock:
            executed.append(name)
        if name == "g1.mp4":
            return TrackResult(item=item, outcome=RunOutcome.FAILED, error="boom")
        time.sleep(0.15)
        return TrackResult(item=item, outcome=RunOutcome.SUCCEEDED)

    monkeypatch.setattr(runner_mod, "_execute_single_item", _fail_on_second)  # type: ignore[attr-defined]
    summary_fail = execute_plan(
        six_items, ffmpeg_path=Path("ffmpeg"), output_root=out_dir, jobs=2, fail_fast=True
    )
    assert summary_fail.interrupted is False
    assert summary_fail.exit_code == 1
    assert "g1.mp4" in executed
    assert len(executed) <= 2


def test_artist_override_rescues_vevo_and_hintless_files(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    vevo = make_video_with_audio(
        videos / "VID_8888.mp4",
        metadata={
            "artist": "DaftPunkVEVO",
            "title": "One More Time (Official Video)",
            "comment": "https://youtu.be/abc12345678",
        },
    )
    hintless = make_video_with_audio(videos / "VID_0042.mp4")

    out_dir = tmp_path / "out"
    media_inputs, output_root = discover_inputs([vevo, hintless], output_dir=out_dir)
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    plan = build_offline_plan(
        media_inputs,
        output_root=output_root,
        ffprobe_path=ffprobe_path,
        config=cfg,
        overrides=ManualOverrides(artist="Daft Punk"),
    )
    summary = execute_plan(plan, ffmpeg_path=ffmpeg_path, output_root=output_root, jobs=2)
    assert summary.exit_code == 0
    assert summary.count(RunOutcome.UNMATCHED) == 0
    assert (output_root / "Daft Punk" / "Unknown Album" / "Daft Punk - One More Time.mp3").is_file()
    assert (output_root / "Daft Punk" / "Unknown Album" / "Daft Punk - VID_0042.mp3").is_file()


def test_real_run_surfaces_bitrate_clamp_warning_and_clean_output_root_line(
    tmp_path: Path,
) -> None:
    clip = make_video_with_audio(
        tmp_path / "Artist - Low Rate.mkv",
        audio_codec="pcm_s16le",
        sample_rate=8000,
        channels=1,
    )
    out_dir = tmp_path / "very_long_output_directory_name_to_avoid_title_wrapping" / "extracted"
    log_file = tmp_path / "clamp.log"

    # 1. Dry-run shows warning ONCE (inside Notes column), not duplicated on stderr!
    dry = subprocess.run(
        [
            sys.executable,
            "-m",
            "audio_from_video.cli.main",
            "--offline",
            "--dry-run",
            "-o",
            str(out_dir),
            str(clip),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env={**os.environ, "COLUMNS": "220"},
        check=False,
    )
    assert dry.returncode == 0, dry.stderr + dry.stdout
    dry_combined = dry.stdout + "\n" + dry.stderr
    assert dry_combined.count("Clamped CBR bitrate from 320k to 64k") == 1

    # 2. Real run prints warning ONCE (in Warnings section), still logs [WARNING] to --log-file!
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "audio_from_video.cli.main",
            "--offline",
            "-o",
            str(out_dir),
            "--log-file",
            str(log_file),
            str(clip),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env={**os.environ, "COLUMNS": "220"},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    combined = completed.stdout + "\n" + completed.stderr
    assert combined.count("Clamped CBR bitrate from 320k to 64k") == 1
    assert "Warnings:" in completed.stdout
    assert "Output root:" in completed.stdout
    assert f"Run Summary ({out_dir})" not in completed.stdout
    # Log file still records the warning at [WARNING] level.
    assert "[WARNING]" in log_file.read_text(encoding="utf-8")


def test_loose_files_and_unnumbered_folders_never_fabricate_album_or_artist(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    # Problem 1: Loose Artist - Title files in ordinary folders stay Unknown Album + partial!
    fav = tmp_path / "Favorites"
    mvid = tmp_path / "Music Videos"
    short = tmp_path / "v"
    for folder in (fav, mvid, short):
        folder.mkdir(parents=True)

    f_fav = make_video_with_audio(fav / "Daft Punk - One More Time.mp4")
    f_mvid = make_video_with_audio(mvid / "Adele - Hello.mp4")
    f_short = make_video_with_audio(short / "50 Cent - In Da Club.mp4")

    # Problem 2: Unnumbered files with no artist in X/Y/ folders stay _unmatched/!
    xy1 = tmp_path / "r2" / "2024"
    xy2 = tmp_path / "p1" / "v"
    xy1.mkdir(parents=True)
    xy2.mkdir(parents=True)
    f_cats = make_video_with_audio(xy1 / "clip about cats.mp4")
    f_guest = make_video_with_audio(xy2 / "Song feat. Guest (Live).mp4")

    out_dir = tmp_path / "out"
    media_inputs, output_root = discover_inputs(
        [f_fav, f_mvid, f_short, f_cats, f_guest], output_dir=out_dir
    )
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    plan = build_offline_plan(
        media_inputs, output_root=output_root, ffprobe_path=ffprobe_path, config=cfg
    )
    by_name = {item.media_input.path.name: item for item in plan}

    for name in ("Daft Punk - One More Time.mp4", "Adele - Hello.mp4", "50 Cent - In Da Club.mp4"):
        meta = by_name[name].metadata
        assert meta is not None
        assert meta.album is None
        assert meta.match_status == MatchStatus.PARTIAL

    for name in ("clip about cats.mp4", "Song feat. Guest (Live).mp4"):
        meta = by_name[name].metadata
        assert meta is not None
        assert meta.match_status == MatchStatus.UNMATCHED

    summary = execute_plan(plan, ffmpeg_path=ffmpeg_path, output_root=output_root, jobs=2)
    assert summary.exit_code == 0
    assert (output_root / "Daft Punk" / "Unknown Album" / "Daft Punk - One More Time.mp3").is_file()
    assert (output_root / "Adele" / "Unknown Album" / "Adele - Hello.mp3").is_file()
    assert (output_root / "50 Cent" / "Unknown Album" / "50 Cent - In Da Club.mp3").is_file()
    assert (output_root / "_unmatched" / "clip about cats.mp3").is_file()
    assert (output_root / "_unmatched" / "Song feat. Guest (Live).mp3").is_file()
    assert ID3(str(output_root / "_unmatched" / "Song feat. Guest (Live).mp3"))["TIT2"].text == [
        "Song (Live) (ft. Guest)"
    ]
    # TALB must NOT be set to "Favorites", "Music Videos", or "v".
    assert "TALB" not in ID3(
        str(output_root / "Daft Punk" / "Unknown Album" / "Daft Punk - One More Time.mp3")
    )

    # Also verify --artist on v/VID_0042.mp4 stays Unknown Album (NOT album="v"!).
    f_vid = make_video_with_audio(short / "VID_0042.mp4")
    out_rescue = tmp_path / "out_rescue"
    mi_r, root_r = discover_inputs([f_vid], output_dir=out_rescue)
    plan_r = build_offline_plan(
        mi_r,
        output_root=root_r,
        ffprobe_path=ffprobe_path,
        config=cfg,
        overrides=ManualOverrides(artist="Daft Punk"),
    )
    assert plan_r[0].metadata is not None
    assert plan_r[0].metadata.album is None
    assert plan_r[0].metadata.match_status == MatchStatus.PARTIAL
    assert (
        plan_r[0].output_path == root_r / "Daft Punk" / "Unknown Album" / "Daft Punk - VID_0042.mp3"
    )


def test_folder_inferred_layout_reports_partial_and_parses_tracks_01_to_12(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    album_dir = tmp_path / "Daft Punk" / "Discovery"
    album_dir.mkdir(parents=True)

    # Tracks 01..11 with "NN Artist - Title", track 12 as "12 Something", plus loose 50 Cent.
    for i in range(1, 10):
        make_video_with_audio(album_dir / f"{i:02d} Daft Punk - Track{i:02d}.mp4")
    make_video_with_audio(album_dir / "10 Daft Punk - Track10.mp4")
    make_video_with_audio(album_dir / "11 Daft Punk - Track11.mp4")
    make_video_with_audio(album_dir / "12 Something.mp4")
    make_video_with_audio(album_dir / "50 Cent - In Da Club.mp4")

    out_dir = tmp_path / "out"
    media_inputs, output_root = discover_inputs([album_dir], output_dir=out_dir)
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    plan = build_offline_plan(
        media_inputs, output_root=output_root, ffprobe_path=ffprobe_path, config=cfg
    )
    by_name = {item.media_input.path.name: item for item in plan}

    # Every folder-inferred album track reports partial (never exact) and keeps artist="Daft Punk".
    for track_num in range(1, 12):
        fname = f"{track_num:02d} Daft Punk - Track{track_num:02d}.mp4"
        item = by_name[fname]
        assert item.metadata is not None
        assert item.metadata.artist == "Daft Punk"
        assert item.metadata.album == "Discovery"
        assert item.metadata.track_number == track_num
        assert item.metadata.match_status == MatchStatus.PARTIAL
        assert item.output_path == (
            output_root
            / "Daft Punk"
            / "Discovery"
            / f"{track_num:02d} Daft Punk - Track{track_num:02d}.mp3"
        )

    item_12 = by_name["12 Something.mp4"]
    assert item_12.metadata is not None
    assert item_12.metadata.artist == "Daft Punk"
    assert item_12.metadata.album == "Discovery"
    assert item_12.metadata.track_number == 12
    assert item_12.metadata.title == "Something"
    assert item_12.metadata.match_status == MatchStatus.PARTIAL
    assert (
        item_12.output_path
        == output_root / "Daft Punk" / "Discovery" / "12 Daft Punk - Something.mp3"
    )

    # 50 Cent in the exact same folder keeps artist="50 Cent", no track, and Unknown Album!
    item_50 = by_name["50 Cent - In Da Club.mp4"]
    assert item_50.metadata is not None
    assert item_50.metadata.artist == "50 Cent"
    assert item_50.metadata.track_number is None
    assert item_50.metadata.album is None
    assert (
        item_50.output_path
        == output_root / "50 Cent" / "Unknown Album" / "50 Cent - In Da Club.mp3"
    )

    summary = execute_plan(plan, ffmpeg_path=ffmpeg_path, output_root=output_root, jobs=4)
    assert summary.exit_code == 0
    assert (output_root / "Daft Punk" / "Discovery" / "10 Daft Punk - Track10.mp3").is_file()
    assert (output_root / "Daft Punk" / "Discovery" / "12 Daft Punk - Something.mp3").is_file()
    assert (output_root / "50 Cent" / "Unknown Album" / "50 Cent - In Da Club.mp3").is_file()


def test_consecutive_numeric_artists_and_gapped_unpadded_album_end_to_end(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    # 1. Loose folder with 2 Chainz, 3 Doors Down, and 4 Non Blondes must NOT become a track run!
    loose = tmp_path / "loose"
    loose.mkdir()
    make_video_with_audio(loose / "2 Chainz - Birthday Song.mp4")
    make_video_with_audio(loose / "3 Doors Down - Kryptonite.mp4")
    make_video_with_audio(loose / "4 Non Blondes - What's Up.mp4")

    # 2. Unpadded album with a gap (1, 2, 3, 10, 11) under Daft Punk/Homework/ resolves all 5!
    hw = tmp_path / "Daft Punk" / "Homework"
    hw.mkdir(parents=True)
    for fname in (
        "1 Daftendirekt.mp4",
        "2 WDPK 83.7 FM.mp4",
        "3 Revolution 909.mp4",
        "10 High Fidelity.mp4",
        "11 Rock'n Roll.mp4",
    ):
        make_video_with_audio(hw / fname)

    out_dir = tmp_path / "out"
    media_inputs, output_root = discover_inputs([loose, hw], output_dir=out_dir)
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    plan = build_offline_plan(
        media_inputs, output_root=output_root, ffprobe_path=ffprobe_path, config=cfg
    )
    summary = execute_plan(plan, ffmpeg_path=ffmpeg_path, output_root=output_root, jobs=4)
    assert summary.exit_code == 0
    assert summary.count(RunOutcome.UNMATCHED) == 0

    assert (output_root / "2 Chainz" / "Unknown Album" / "2 Chainz - Birthday Song.mp3").is_file()
    assert (
        output_root / "3 Doors Down" / "Unknown Album" / "3 Doors Down - Kryptonite.mp3"
    ).is_file()
    assert (
        output_root / "4 Non Blondes" / "Unknown Album" / "4 Non Blondes - What's Up.mp3"
    ).is_file()

    for expected_name in (
        "01 Daft Punk - Daftendirekt.mp3",
        "02 Daft Punk - WDPK 83.7 FM.mp3",
        "03 Daft Punk - Revolution 909.mp3",
        "10 Daft Punk - High Fidelity.mp3",
        "11 Daft Punk - Rock'n Roll.mp3",
    ):
        assert (output_root / "Daft Punk" / "Homework" / expected_name).is_file()


def test_explicit_file_subset_preserves_parent_folder_sibling_context(
    tmp_path: Path, ffprobe_path: Path
) -> None:
    album = tmp_path / "album"
    album.mkdir()
    for i in range(1, 13):
        make_video_with_audio(album / f"{i:02d} Daft Punk - T{i:02d}.mp4")
    f_50 = make_video_with_audio(album / "50 Cent - In Da Club.mp4")
    f_10y = make_video_with_audio(album / "10 Years - Wasteland.mp4")

    f10 = album / "10 Daft Punk - T10.mp4"
    f11 = album / "11 Daft Punk - T11.mp4"
    f12 = album / "12 Daft Punk - T12.mp4"
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    out_dir = tmp_path / "out"

    # 1. Passing 10 and 12 explicitly gives artist "Daft Punk" and tracks 10 & 12 ONLY.
    mi_pair, root = discover_inputs([f10, f12], output_dir=out_dir)
    plan_pair = build_offline_plan(mi_pair, output_root=root, ffprobe_path=ffprobe_path, config=cfg)
    assert [item.media_input.path.name for item in plan_pair] == [
        "10 Daft Punk - T10.mp4",
        "12 Daft Punk - T12.mp4",
    ]
    assert [
        (item.metadata.artist, item.metadata.track_number) for item in plan_pair if item.metadata
    ] == [
        ("Daft Punk", 10),
        ("Daft Punk", 12),
    ]

    # 2. Passing 10, 11, and 12 explicitly (e.g., shell glob album/1*.mp4) resolves all 3.
    mi_triple, _ = discover_inputs([f10, f11, f12], output_dir=out_dir)
    plan_triple = build_offline_plan(
        mi_triple, output_root=root, ffprobe_path=ffprobe_path, config=cfg
    )
    assert len(plan_triple) == 3
    assert [
        (item.metadata.artist, item.metadata.track_number) for item in plan_triple if item.metadata
    ] == [
        ("Daft Punk", 10),
        ("Daft Punk", 11),
        ("Daft Punk", 12),
    ]

    # 3. Passing 50 Cent and 10 Years explicitly from the same mixed folder keeps both intact.
    mi_mixed, _ = discover_inputs([f_50, f_10y], output_dir=out_dir)
    plan_mixed = build_offline_plan(
        mi_mixed, output_root=root, ffprobe_path=ffprobe_path, config=cfg
    )
    assert len(plan_mixed) == 2
    by_name = {item.media_input.path.name: item.metadata for item in plan_mixed}
    assert by_name["50 Cent - In Da Club.mp4"] is not None
    assert by_name["50 Cent - In Da Club.mp4"].artist == "50 Cent"
    assert by_name["50 Cent - In Da Club.mp4"].track_number is None
    assert by_name["10 Years - Wasteland.mp4"] is not None
    assert by_name["10 Years - Wasteland.mp4"].artist == "10 Years"
    assert by_name["10 Years - Wasteland.mp4"].track_number is None
