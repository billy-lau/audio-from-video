"""Tests for Typer DefaultGroup routing (path-first, option-first, and guardrails)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from audio_from_video.cli.main import app
from conftest import make_video_with_audio

runner = CliRunner()


def test_group_help_and_version() -> None:
    res_empty = runner.invoke(app, [])
    assert "Usage:" in res_empty.output

    res_h = runner.invoke(app, ["-h"])
    assert res_h.exit_code == 0
    assert "extract" in res_h.output

    res_ver = runner.invoke(app, ["--version"])
    assert res_ver.exit_code == 0
    assert "audio-from-video 0.1.0" in res_ver.output


def test_subcommands_route_directly() -> None:
    res_doc = runner.invoke(app, ["doctor"])
    assert res_doc.exit_code == 0
    assert "afv doctor" in res_doc.output
    assert "[musicbrainz]" in res_doc.output or "MusicBrainz UA" in res_doc.output

    res_cfg = runner.invoke(app, ["config", "path"])
    assert res_cfg.exit_code == 0
    assert "config.toml" in res_cfg.output

    res_show = runner.invoke(app, ["config", "show"])
    assert res_show.exit_code == 0
    assert "audio.quality" in res_show.output

    res_cache = runner.invoke(app, ["cache", "stats"])
    assert res_cache.exit_code == 0
    assert "Cache directory:" in res_cache.output


def test_option_first_and_path_first_route_to_extract(tmp_path: Path) -> None:
    clip = make_video_with_audio(
        tmp_path / "01 Daft Punk - One More Time.mp4",
        metadata={"artist": "Daft Punk", "title": "One More Time", "album": "Discovery"},
    )

    # Option-first: --dry-run BEFORE input path!
    res_opt_first = runner.invoke(
        app, ["--dry-run", "--offline", str(tmp_path)], env={"COLUMNS": "200"}
    )
    assert res_opt_first.exit_code == 0, res_opt_first.output
    assert "Dry-Run Plan" in res_opt_first.output
    assert "One More Time.mp3" in res_opt_first.output

    # Flag-first with -v:
    res_v_first = runner.invoke(app, ["-v", "--offline", "--dry-run", str(clip)])
    assert res_v_first.exit_code == 0, res_v_first.output

    # Path-first: input path BEFORE options!
    res_path_first = runner.invoke(app, [str(clip), "--offline", "--dry-run"])
    assert res_path_first.exit_code == 0, res_path_first.output


def test_literal_doctor_path_via_dot_slash_and_explicit_extract(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    doc_dir = tmp_path / "doctor"
    doc_dir.mkdir()
    make_video_with_audio(
        doc_dir / "Daft Punk - One More Time.mp4",
        metadata={"artist": "Daft Punk", "title": "One More Time"},
    )

    # './doctor' does not match subcommand name 'doctor' -> routes to extract.
    res_dot = runner.invoke(app, ["./doctor", "--offline", "--dry-run"])
    assert res_dot.exit_code == 0, res_dot.output
    assert "Dry-Run Plan" in res_dot.output

    # Explicit 'extract doctor' also works.
    res_explicit = runner.invoke(app, ["extract", "doctor", "--offline", "--dry-run"])
    assert res_explicit.exit_code == 0, res_explicit.output
    assert "Dry-Run Plan" in res_explicit.output


def test_cli_validation_guardrails_exit_code_2(tmp_path: Path, monkeypatch: object) -> None:
    f1 = make_video_with_audio(tmp_path / "a.mp4")
    f2 = make_video_with_audio(tmp_path / "b.mp4")

    # 1. Invalid -q/--quality
    res_q = runner.invoke(app, ["--offline", "-q", "999k", str(f1)])
    assert res_q.exit_code == 2

    # 2. Invalid -j/--jobs < 1
    res_j = runner.invoke(app, ["--offline", "-j", "0", str(f1)])
    assert res_j.exit_code == 2

    # 3. --normalize is enabled in Phase 3 (dry-run succeeds and displays +norm).
    res_norm = runner.invoke(app, ["--offline", "--normalize", "--dry-run", str(f1)])
    assert res_norm.exit_code == 0, res_norm.output
    assert "+norm" in res_norm.output

    # 4. --always-prompt on non-TTY / --non-interactive
    res_prompt = runner.invoke(app, ["--offline", "--always-prompt", "--non-interactive", str(f1)])
    assert res_prompt.exit_code == 2

    # 5. Online mode without MusicBrainz contact in non-TTY exits with code 2.
    monkeypatch.delenv("AFV_MB_CONTACT", raising=False)  # type: ignore[attr-defined]
    monkeypatch.delenv("AFV_MB_USER_AGENT", raising=False)  # type: ignore[attr-defined]
    monkeypatch.setenv("AFV_CONFIG_PATH", str(tmp_path / "empty.toml"))  # type: ignore[attr-defined]
    res_online = runner.invoke(app, [str(f1)])
    assert res_online.exit_code == 2
    assert "AFV_MB_CONTACT" in res_online.output
    assert "--offline" in res_online.output
    assert "afv config init" not in res_online.output

    # 6. --title and --track with > 1 resolved file -> exit code 2
    res_title = runner.invoke(app, ["--offline", "--title", "Only One", str(f1), str(f2)])
    assert res_title.exit_code == 2
    res_track = runner.invoke(app, ["--offline", "--track", "1", str(tmp_path)])
    assert res_track.exit_code == 2

    # 7. Inputs spanning multiple directories without -o -> exit code 2
    dir_a = tmp_path / "dir_a"
    dir_b = tmp_path / "dir_b"
    dir_a.mkdir()
    dir_b.mkdir()
    fa = make_video_with_audio(dir_a / "a.mp4")
    fb = make_video_with_audio(dir_b / "b.mp4")
    res_roots = runner.invoke(app, ["--offline", "--dry-run", str(fa), str(fb)])
    assert res_roots.exit_code == 2
