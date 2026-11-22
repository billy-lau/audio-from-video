"""Shared pytest helpers and real FFmpeg synthetic media generators."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_user_environment(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep every test away from the real user's config, cache, logs, and settings.

    Points ``AFV_CONFIG_PATH`` at a file that does not exist yet and clears every other
    ``AFV_*`` override. ``HOME`` and the XDG folders move to a temporary home, which
    redirects the platformdirs cache and log folders (and Qt settings) on macOS and
    Linux. Subprocesses inherit the same environment. Tests that need a specific value
    still set it themselves.
    """
    home = tmp_path_factory.mktemp("home")
    for name in [key for key in os.environ if key.startswith("AFV_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("AFV_CONFIG_PATH", str(home / "config.toml"))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    monkeypatch.setenv("XDG_STATE_HOME", str(home / ".local" / "state"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))


def _find_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not available on PATH")
    return ffmpeg


def _find_ffprobe() -> str:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        pytest.skip("ffprobe not available on PATH")
    return ffprobe


def run_ffmpeg_gen(args: list[str]) -> None:
    """Run ffmpeg to generate a tiny synthetic test fixture."""
    ffmpeg = _find_ffmpeg()
    completed = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Synthetic fixture generation failed: {completed.stderr}")


def make_video_with_audio(
    path: Path,
    *,
    audio_codec: str = "aac",
    sample_rate: int = 44100,
    channels: int = 2,
    duration: float = 0.6,
    bitrate: str | None = None,
    metadata: dict[str, str] | None = None,
    with_video: bool = True,
) -> Path:
    """Generate a tiny real video/audio file using FFmpeg lavfi sources."""
    path.parent.mkdir(parents=True, exist_ok=True)
    args: list[str] = []
    if with_video:
        args.extend(["-f", "lavfi", "-i", f"color=c=black:s=64x64:d={duration}"])
    layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(channels, "stereo")
    args.extend(
        [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate={sample_rate}:duration={duration}",
        ]
    )
    if with_video:
        args.extend(["-c:v", "mpeg4", "-pix_fmt", "yuv420p"])
    args.extend(["-ac", str(channels)])
    if channels == 6:
        args.extend(["-af", f"aformat=channel_layouts={layout}"])
    args.extend(["-c:a", audio_codec])
    if bitrate:
        args.extend(["-b:a", bitrate])
    if metadata:
        for key, value in metadata.items():
            args.extend(["-metadata", f"{key}={value}"])
    args.append(str(path))
    run_ffmpeg_gen(args)
    return path


def make_silent_video_no_audio(path: Path, *, duration: float = 0.4) -> Path:
    """Generate a real MP4 video containing ONLY a video stream (no audio)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg_gen(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:s=64x64:d={duration}",
            "-c:v",
            "mpeg4",
            "-an",
            str(path),
        ]
    )
    return path


def make_raw_mp3(path: Path, *, duration: float = 0.5, sample_rate: int = 44100) -> Path:
    """Generate a tiny real MP3 without any ID3 header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg_gen(
        [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate={sample_rate}:duration={duration}",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            "-id3v2_version",
            "0",
            "-write_id3v1",
            "0",
            str(path),
        ]
    )
    return path


def make_video_with_chapters(path: Path) -> Path:
    """Generate a real MKV video containing an audio track and two chapters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmeta = path.parent / f".{path.stem}.ffmeta"
    ffmeta.write_text(
        ";FFMETADATA1\n"
        "[CHAPTER]\n"
        "TIMEBASE=1/1000\n"
        "START=0\n"
        "END=300\n"
        "title=Intro\n"
        "[CHAPTER]\n"
        "TIMEBASE=1/1000\n"
        "START=300\n"
        "END=600\n"
        "title=Chorus\n",
        encoding="utf-8",
    )
    try:
        run_ffmpeg_gen(
            [
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:d=0.6",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=44100:duration=0.6",
                "-i",
                str(ffmeta),
                "-map_metadata",
                "2",
                "-map_chapters",
                "2",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "mpeg4",
                "-c:a",
                "aac",
                str(path),
            ]
        )
    finally:
        ffmeta.unlink(missing_ok=True)
    return path


def make_multi_track_mkv(path: Path) -> Path:
    """Generate an MKV with 3 audio tracks: Japanese default, English FLAC, and commentary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg_gen(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=0.5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=0.5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=660:sample_rate=44100:duration=0.5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=44100:duration=0.5",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map",
            "2:a:0",
            "-map",
            "3:a:0",
            "-c:v",
            "mpeg4",
            "-c:a:0",
            "aac",
            "-metadata:s:a:0",
            "language=jpn",
            "-metadata:s:a:0",
            "title=Japanese",
            "-disposition:a:0",
            "default",
            "-c:a:1",
            "flac",
            "-metadata:s:a:1",
            "language=eng",
            "-metadata:s:a:1",
            "title=English Lossless",
            "-disposition:a:1",
            "0",
            "-c:a:2",
            "aac",
            "-metadata:s:a:2",
            "language=eng",
            "-metadata:s:a:2",
            "title=Director Commentary",
            "-disposition:a:2",
            "comment",
            str(path),
        ]
    )
    return path


@pytest.fixture(scope="session")
def ffmpeg_path() -> Path:
    return Path(_find_ffmpeg())


@pytest.fixture(scope="session")
def ffprobe_path() -> Path:
    return Path(_find_ffprobe())
