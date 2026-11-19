"""Tests for FFmpeg extraction commands, bitrate clamping, stream-copy, and interrupts."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest
from mutagen.id3 import ID3, ID3NoHeaderError
from mutagen.mp3 import MP3

from audio_from_video.config import parse_quality
from audio_from_video.extractor import (
    ProcessRegistry,
    build_ffmpeg_command,
    build_loudnorm_measure_command,
    clamp_cbr_bitrate,
    detect_loudnorm_max_lra,
    make_part_path,
    parse_loudnorm_json,
    plan_extraction,
    run_ffmpeg,
    select_target_sample_rate,
)
from audio_from_video.models import AudioStreamInfo
from audio_from_video.probe import probe_media
from conftest import make_video_with_audio, make_video_with_chapters, run_ffmpeg_gen


@pytest.mark.parametrize(
    ("source_sr", "expected"),
    [
        (None, 44100),
        (0, 44100),
        (8000, 8000),
        (11025, 11025),
        (22050, 22050),
        (44100, 44100),
        (48000, 48000),
        (88200, 44100),
        (176400, 44100),
        (96000, 48000),
        (192000, 48000),
        (10000, 11025),
        (40000, 44100),
    ],
)
def test_select_target_sample_rate(source_sr: int | None, expected: int) -> None:
    assert select_target_sample_rate(source_sr) == expected


@pytest.mark.parametrize(
    ("target_kbps", "sample_rate", "expected_kbps", "clamped"),
    [
        (320, 44100, 320, False),
        (320, 48000, 320, False),
        (16, 44100, 32, True),  # Below MPEG-1 minimum 32k
        (320, 22050, 160, True),  # Above MPEG-2 ceiling 160k
        (160, 22050, 160, False),
        (320, 11025, 64, True),  # Above MPEG-2.5 ceiling 64k
        (64, 8000, 64, False),
        (144, 44100, 160, True),  # 144k is MPEG-2 only -> snaps to 160k for MPEG-1
    ],
)
def test_clamp_cbr_bitrate(
    target_kbps: int, sample_rate: int, expected_kbps: int, clamped: bool
) -> None:
    got, was_clamped = clamp_cbr_bitrate(target_kbps, sample_rate)
    assert got == expected_kbps
    assert was_clamped is clamped


def test_stream_copy_precedence_matrix() -> None:
    q320 = parse_quality("320k")
    mp3_stream = AudioStreamInfo(0, 1, "mp3", 44100, 2, bit_rate=128000)
    aac_stream = AudioStreamInfo(0, 1, "aac", 44100, 6, bit_rate=384000)

    # Default MP3 -> stream copy.
    p_copy = plan_extraction(mp3_stream, quality=q320, copy_mp3=True)
    assert p_copy.stream_copy is True

    # Non-MP3 -> re-encode and downmix 5.1 (6 ch) to stereo (2 ch).
    p_aac = plan_extraction(aac_stream, quality=q320, copy_mp3=True)
    assert p_aac.stream_copy is False
    assert p_aac.target_channels == 2
    assert p_aac.effective_bitrate_kbps == 320

    # --reencode forces re-encode on MP3.
    p_re = plan_extraction(mp3_stream, quality=q320, copy_mp3=True, reencode=True)
    assert p_re.stream_copy is False

    # copy_mp3=False in config forces re-encode on MP3.
    p_cfg = plan_extraction(mp3_stream, quality=q320, copy_mp3=False)
    assert p_cfg.stream_copy is False

    # Explicit CLI -q forces re-encode on MP3 and warns when exceeding source bitrate.
    p_cli = plan_extraction(mp3_stream, quality=q320, copy_mp3=True, explicit_cli_quality=True)
    assert p_cli.stream_copy is False
    assert any("exceeds source bitrate" in w for w in p_cli.warnings)


def test_build_ffmpeg_command_flags() -> None:
    part = make_part_path(Path("/tmp/out"))
    assert part.name.startswith(".afv-")
    assert part.name.endswith(".part")
    assert len(part.name) == len(".afv-") + 12 + len(".part")

    stream = AudioStreamInfo(1, 2, "aac", 22050, 6)
    plan = plan_extraction(stream, quality=parse_quality("320k"))
    cmd = build_ffmpeg_command(Path("ffmpeg"), Path("in.mkv"), part, plan)

    assert "-nostdin" in cmd
    assert "-y" in cmd
    assert "-vn" in cmd and "-sn" in cmd and "-dn" in cmd
    assert cmd[cmd.index("-map") + 1] == "0:a:1"
    assert cmd[cmd.index("-map_metadata") + 1] == "-1"
    assert cmd[cmd.index("-map_chapters") + 1] == "-1"
    assert cmd[cmd.index("-id3v2_version") + 1] == "0"
    assert cmd[cmd.index("-write_id3v1") + 1] == "0"
    assert cmd[cmd.index("-f") + 1] == "mp3"
    assert cmd[cmd.index("-ar") + 1] == "22050"
    assert cmd[cmd.index("-ac") + 1] == "2"
    assert cmd[cmd.index("-b:a") + 1] == "160k"  # Clamped from 320k!


def test_real_ffmpeg_extraction_aac_mp3_copy_low_sr_51_and_chapters(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    # 1. AAC 5.1 -> downmixed stereo 320k MP3 with NO FFmpeg ID3 header.
    src_51 = make_video_with_audio(
        tmp_path / "surround.mp4", audio_codec="aac", sample_rate=48000, channels=6
    )
    probe_51 = probe_media(ffprobe_path, src_51)
    plan_51 = plan_extraction(probe_51.audio_streams[0], quality=parse_quality("320k"))
    part_51 = make_part_path(tmp_path)
    run_ffmpeg(ffmpeg_path, src_51, part_51, plan_51)
    mp3_51 = MP3(str(part_51))
    assert mp3_51.info.channels == 2
    assert mp3_51.info.sample_rate == 48000
    assert mp3_51.info.bitrate == pytest.approx(320000, rel=0.02)
    with pytest.raises(ID3NoHeaderError):
        ID3(str(part_51))

    # 2. MP3 inside MKV -> stream-copy (0 generational loss).
    src_mp3 = make_video_with_audio(
        tmp_path / "already_mp3.mkv", audio_codec="libmp3lame", bitrate="192k"
    )
    probe_mp3 = probe_media(ffprobe_path, src_mp3)
    plan_copy = plan_extraction(probe_mp3.audio_streams[0], quality=parse_quality("320k"))
    assert plan_copy.stream_copy is True
    part_copy = make_part_path(tmp_path)
    run_ffmpeg(ffmpeg_path, src_mp3, part_copy, plan_copy)
    # Stream-copied 192k MP3 with Xing/Info header reports ~192 kbps (never 320k re-encode).
    assert MP3(str(part_copy)).info.bitrate == pytest.approx(192000, rel=0.02)

    # 3. Low sample rate (22050 Hz) clamps 320k -> 160k so libmp3lame succeeds cleanly.
    src_low = make_video_with_audio(
        tmp_path / "low_sr.mkv", audio_codec="pcm_s16le", sample_rate=22050, channels=1
    )
    probe_low = probe_media(ffprobe_path, src_low)
    plan_low = plan_extraction(probe_low.audio_streams[0], quality=parse_quality("320k"))
    assert plan_low.effective_bitrate_kbps == 160
    assert plan_low.clamped_from_kbps == 320
    part_low = make_part_path(tmp_path)
    run_ffmpeg(ffmpeg_path, src_low, part_low, plan_low)
    mp3_low = MP3(str(part_low))
    assert mp3_low.info.sample_rate == 22050
    assert mp3_low.info.channels == 1
    assert mp3_low.info.bitrate == pytest.approx(160000, rel=0.02)

    # 4. Chaptered MKV -> chapters are stripped from output MP3.
    src_chap = make_video_with_chapters(tmp_path / "chap.mkv")
    probe_chap = probe_media(ffprobe_path, src_chap)
    assert probe_chap.has_chapters is True
    part_chap = make_part_path(tmp_path)
    run_ffmpeg(
        ffmpeg_path,
        src_chap,
        part_chap,
        plan_extraction(probe_chap.audio_streams[0], quality=parse_quality("v2")),
    )
    out_probe = probe_media(ffprobe_path, part_chap)
    assert out_probe.has_chapters is False


def test_process_registry_terminate_all_kills_child_and_removes_part(tmp_path: Path) -> None:
    registry = ProcessRegistry()
    part = make_part_path(tmp_path)
    part.write_bytes(b"partial-data")

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert registry.register_process(proc, part) is True

    removed = registry.terminate_all(timeout=1.0)
    # Give process exit a tiny check; wait already ran inside terminate_all.
    assert proc.poll() is not None
    assert not part.exists()
    assert part in removed
    assert registry.is_interrupted is True

    # Further registrations after interrupt are refused.
    late = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.1)"])
    try:
        assert registry.register_process(late, part) is False
    finally:
        late.wait(timeout=2.0)
        time.sleep(0.01)


def _make_dynamic_audio_video(
    path: Path,
    *,
    channels: int = 2,
    sample_rate: int = 44100,
    peak_limited: bool = False,
) -> Path:
    """Generate a multi-segment real video with non-zero LRA for real FFmpeg loudnorm tests.

    Combines quiet and loud musical-rate tone segments (plus an optional high-crest peak
    impulse when peak_limited=True so true-peak headroom forces effective_i < target_i).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if peak_limited:
        # FFmpeg's sine source has ~-18 dBFS amplitude: combining quiet (-12 dB), medium (-4 dB),
        # and a brief peak (+16 dB) yields LRA > 0 with high crest so linear mode peak-limits
        # effective_i < target_i (-16.0 LUFS) rather than clipping or switching to dynamic mode.
        filter_expr = (
            f"sine=f=220:r={sample_rate}:d=1.5,volume=-12dB[a];"
            f"sine=f=330:r={sample_rate}:d=1.5,volume=-4dB[b];"
            f"sine=f=440:r={sample_rate}:d=0.05,volume=+16dB[c];"
            f"[a][b][c]concat=n=3:v=0:a=1"
        )
    else:
        # Three segments at -26 dB, -20 dB, and -16 dB -> genuine non-zero LRA with plenty of
        # true-peak headroom to reach -16.0 LUFS cleanly in linear mode.
        filter_expr = (
            f"sine=f=220:r={sample_rate}:d=1.2,volume=-26dB[a];"
            f"sine=f=330:r={sample_rate}:d=1.2,volume=-20dB[b];"
            f"sine=f=440:r={sample_rate}:d=1.2,volume=-16dB[c];"
            f"[a][b][c]concat=n=3:v=0:a=1"
        )

    if channels == 6:
        filter_expr += ",aformat=channel_layouts=5.1"
    elif channels == 2:
        filter_expr += ",aformat=channel_layouts=stereo"
    else:
        filter_expr += ",aformat=channel_layouts=mono"

    run_ffmpeg_gen(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=3.6",
            "-filter_complex",
            f"{filter_expr}[aout]",
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "mpeg4",
            "-c:a",
            "pcm_s16le",
            str(path),
        ]
    )
    return path


def test_normalize_forces_reencode_on_mp3_and_builds_channel_prep_commands() -> None:
    q320 = parse_quality("320k")
    mp3_stream = AudioStreamInfo(0, 1, "mp3", 44100, 2, bit_rate=192000)
    plan_mp3 = plan_extraction(mp3_stream, quality=q320, copy_mp3=True, normalize=True)
    assert plan_mp3.stream_copy is False
    assert plan_mp3.normalize is True

    # 5.1 surround -> pre-loudnorm stereo downmix inside -af!
    s51 = AudioStreamInfo(0, 1, "aac", 48000, 6)
    p51 = plan_extraction(s51, quality=q320, normalize=True)
    cmd51 = build_loudnorm_measure_command(Path("ffmpeg"), Path("in.mkv"), p51)
    af51 = cmd51[cmd51.index("-af") + 1]
    assert af51.startswith("aformat=channel_layouts=stereo,loudnorm=")
    assert "dual_mono=true" not in af51

    # Mono -> dual_mono=true inside -af!
    s1 = AudioStreamInfo(0, 1, "aac", 44100, 1)
    p1 = plan_extraction(s1, quality=q320, normalize=True)
    cmd1 = build_loudnorm_measure_command(Path("ffmpeg"), Path("in.mkv"), p1)
    af1 = cmd1[cmd1.index("-af") + 1]
    assert "aformat=channel_layouts=stereo" not in af1
    assert ":dual_mono=true:print_format=json" in af1

    # JSON parser extracts the last loudnorm object even with noisy stderr lines.
    noisy = '[Parsed_loudnorm_0 @ 0x1]\n{\n  "input_i": "-18.40",\n  "input_lra": "4.20"\n}\n'
    parsed = parse_loudnorm_json(noisy)
    assert parsed is not None
    assert parsed["input_i"] == "-18.40"


def test_detect_loudnorm_max_lra_real_ffmpeg(ffmpeg_path: Path) -> None:
    max_lra = detect_loudnorm_max_lra(ffmpeg_path)
    assert max_lra in (20.0, 50.0)


def test_real_ffmpeg_two_pass_loudnorm_51_mono_dynamic_peak_sine_and_silent(
    tmp_path: Path, ffmpeg_path: Path, ffprobe_path: Path
) -> None:
    q = parse_quality("192k")

    # 1. Dynamic 5.1 surround -> pre-downmixed stereo + linear two-pass loudnorm!
    src_51 = _make_dynamic_audio_video(tmp_path / "dyn_51.mkv", channels=6, sample_rate=48000)
    probe_51 = probe_media(ffprobe_path, src_51)
    assert probe_51.audio_streams[0].channels == 6
    plan_51 = plan_extraction(probe_51.audio_streams[0], quality=q, normalize=True)
    part_51 = make_part_path(tmp_path)
    warn_51: list[str] = []
    norm_51 = run_ffmpeg(ffmpeg_path, src_51, part_51, plan_51, warnings_out=warn_51)
    assert norm_51 is not None
    assert norm_51.normalization_type == "linear"
    assert norm_51.target_i == -16.0
    assert norm_51.effective_i is not None
    mp3_51 = MP3(str(part_51))
    assert mp3_51.info.channels == 2
    assert mp3_51.info.sample_rate == 48000

    # 2. Dynamic mono -> dual_mono=true + linear two-pass loudnorm!
    src_mono = _make_dynamic_audio_video(tmp_path / "dyn_mono.mkv", channels=1)
    probe_mono = probe_media(ffprobe_path, src_mono)
    plan_mono = plan_extraction(probe_mono.audio_streams[0], quality=q, normalize=True)
    part_mono = make_part_path(tmp_path)
    norm_mono = run_ffmpeg(ffmpeg_path, src_mono, part_mono, plan_mono)
    assert norm_mono is not None
    assert norm_mono.normalization_type == "linear"
    assert MP3(str(part_mono)).info.channels == 1

    # 3. High-crest peak-limited track -> effective_i < target_i, stays linear, emits notice!
    src_peak = _make_dynamic_audio_video(
        tmp_path / "peak_limited.mkv", channels=2, peak_limited=True
    )
    probe_peak = probe_media(ffprobe_path, src_peak)
    plan_peak = plan_extraction(probe_peak.audio_streams[0], quality=q, normalize=True)
    part_peak = make_part_path(tmp_path)
    warn_peak: list[str] = []
    norm_peak = run_ffmpeg(ffmpeg_path, src_peak, part_peak, plan_peak, warnings_out=warn_peak)
    assert norm_peak is not None
    assert norm_peak.input_lra is not None and norm_peak.input_lra > 0.0
    assert norm_peak.effective_i is not None and norm_peak.effective_i < norm_peak.target_i
    assert norm_peak.normalization_type == "linear"
    assert any("peak-limited" in w for w in warn_peak)
    assert any('loudnorm_mode = "dynamic"' in w for w in warn_peak)
    assert MP3(str(part_peak)).info.sample_rate == 44100

    # 4. Pure constant sine wave -> LRA == 0.0 sentinel triggers linear_volume_fallback!
    src_sine = make_video_with_audio(tmp_path / "pure_sine.mp4", duration=1.0)
    probe_sine = probe_media(ffprobe_path, src_sine)
    plan_sine = plan_extraction(probe_sine.audio_streams[0], quality=q, normalize=True)
    part_sine = make_part_path(tmp_path)
    norm_sine = run_ffmpeg(ffmpeg_path, src_sine, part_sine, plan_sine)
    assert norm_sine is not None
    assert norm_sine.normalization_type == "linear_volume_fallback"
    assert norm_sine.gain_db is not None
    assert MP3(str(part_sine)).info.sample_rate == 44100

    # 5. Completely silent track (input_i == -inf) -> warns & re-encodes without loudnorm!
    src_silent = tmp_path / "silent_audio.mkv"
    run_ffmpeg_gen(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=0.6",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100:d=0.6",
            "-t",
            "0.6",
            "-c:v",
            "mpeg4",
            "-c:a",
            "pcm_s16le",
            str(src_silent),
        ]
    )
    probe_sil = probe_media(ffprobe_path, src_silent)
    plan_sil = plan_extraction(probe_sil.audio_streams[0], quality=q, normalize=True)
    part_sil = make_part_path(tmp_path)
    warn_sil: list[str] = []
    norm_sil = run_ffmpeg(ffmpeg_path, src_silent, part_sil, plan_sil, warnings_out=warn_sil)
    assert norm_sil is not None
    assert norm_sil.normalization_type == "skipped_silent"
    assert any("silent" in w.lower() for w in warn_sil)
    assert part_sil.is_file() and part_sil.stat().st_size > 0
