"""Tests for ffprobe JSON parsing and multi-track audio stream selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_from_video.models import AudioStreamInfo
from audio_from_video.probe import (
    ProbeError,
    StreamSelectionError,
    needs_stream_prompt,
    parse_probe_json,
    probe_media,
    select_audio_stream,
)
from conftest import (
    make_multi_track_mkv,
    make_silent_video_no_audio,
    make_video_with_audio,
    make_video_with_chapters,
)


def test_parse_probe_json_merges_tags_and_detects_commentary() -> None:
    payload = {
        "format": {
            "format_name": "matroska,webm",
            "duration": "245.5",
            "tags": {"ARTIST": "Daft Punk", "TITLE": "One More Time"},
        },
        "chapters": [{"id": 0, "start_time": "0.0"}],
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264"},
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
                "bit_rate": "192000",
                "disposition": {"default": 0, "comment": 1},
                "tags": {"language": "eng", "title": "Director Commentary"},
            },
            {
                "index": 2,
                "codec_type": "audio",
                "codec_name": "flac",
                "sample_rate": "44100",
                "channels": 2,
                "disposition": {"default": 1},
                "tags": {"language": "jpn", "title": "Main Stereo"},
            },
        ],
    }
    res = parse_probe_json(Path("movie.mkv"), payload)
    assert res.duration_seconds == pytest.approx(245.5)
    assert res.has_chapters is True
    assert res.container_tags["artist"] == "Daft Punk"
    assert len(res.audio_streams) == 2
    assert res.audio_streams[0].audio_index == 0
    assert res.audio_streams[0].is_commentary is True
    assert res.audio_streams[1].audio_index == 1
    assert res.audio_streams[1].is_commentary is False

    picked = select_audio_stream(res.audio_streams)
    assert picked.audio_index == 1
    assert needs_stream_prompt(res.audio_streams) is False


def test_real_ffprobe_on_video_chapters_and_audio_less(tmp_path: Path, ffprobe_path: Path) -> None:
    normal = make_video_with_audio(
        tmp_path / "song.mp4",
        audio_codec="aac",
        sample_rate=44100,
        metadata={"artist": "Daft Punk", "title": "One More Time", "album": "Discovery"},
    )
    res = probe_media(ffprobe_path, normal)
    assert len(res.audio_streams) == 1
    assert res.audio_streams[0].codec_name == "aac"
    assert res.audio_streams[0].sample_rate == 44100
    assert res.container_tags["artist"] == "Daft Punk"
    assert res.container_tags["album"] == "Discovery"

    silent = make_silent_video_no_audio(tmp_path / "silent.mp4")
    silent_res = probe_media(ffprobe_path, silent)
    assert silent_res.audio_streams == []

    chaptered = make_video_with_chapters(tmp_path / "chapters.mkv")
    chap_res = probe_media(ffprobe_path, chaptered)
    assert chap_res.has_chapters is True
    assert len(chap_res.audio_streams) == 1


def test_real_multi_track_selection_lang_and_stream_override(
    tmp_path: Path, ffprobe_path: Path
) -> None:
    mkv = make_multi_track_mkv(tmp_path / "multi.mkv")
    res = probe_media(ffprobe_path, mkv)
    assert len(res.audio_streams) == 3
    assert res.audio_streams[2].is_commentary is True
    assert needs_stream_prompt(res.audio_streams) is True

    # Default auto-pick prefers the default non-commentary stream (audio_index=0, jpn)
    # and emits an informational warning naming the chosen stream and alternatives.
    auto_warnings: list[str] = []
    auto = select_audio_stream(res.audio_streams, warnings_out=auto_warnings)
    assert auto.audio_index == 0
    assert auto.language == "jpn"
    assert len(auto_warnings) == 1
    assert "jpn" in auto_warnings[0]
    assert "eng" in auto_warnings[0]
    assert "--lang" in auto_warnings[0]

    # --lang en matches ISO 639-2 'eng' non-commentary stream (audio_index=1, flac).
    eng = select_audio_stream(res.audio_streams, lang="en")
    assert eng.audio_index == 1
    assert eng.codec_name == "flac"

    # --stream 2 allows explicitly choosing commentary if requested.
    explicit = select_audio_stream(res.audio_streams, stream_index=2)
    assert explicit.audio_index == 2

    with pytest.raises(StreamSelectionError):
        select_audio_stream(res.audio_streams, stream_index=9)
    with pytest.raises(StreamSelectionError):
        select_audio_stream(res.audio_streams, lang="fra")


def test_select_stream_prefers_non_commentary_and_highest_quality() -> None:
    streams = [
        AudioStreamInfo(
            0, 1, "aac", 48000, 2, bit_rate=128000, is_default=False, is_commentary=False
        ),
        AudioStreamInfo(
            1, 2, "flac", 48000, 2, bit_rate=None, is_default=False, is_commentary=False
        ),
        AudioStreamInfo(
            2, 3, "aac", 48000, 2, bit_rate=256000, is_default=True, is_commentary=True
        ),
    ]
    picked = select_audio_stream(streams)
    assert picked.audio_index == 1


def test_lang_falls_back_gracefully_for_sole_untagged_stream() -> None:
    untagged = [AudioStreamInfo(0, 1, "aac", 44100, 2, language=None)]
    warnings_out: list[str] = []
    chosen = select_audio_stream(untagged, lang="eng", warnings_out=warnings_out)
    assert chosen.audio_index == 0
    assert len(warnings_out) == 1
    assert "falling back" in warnings_out[0]

    # Single stream WITH a different explicit language tag must still raise.
    jpn_only = [AudioStreamInfo(0, 1, "aac", 44100, 2, language="jpn")]
    with pytest.raises(StreamSelectionError):
        select_audio_stream(jpn_only, lang="eng")

    # Multiple untagged streams cannot be disambiguated automatically -> raise.
    two_untagged = [
        AudioStreamInfo(0, 1, "aac", 44100, 2, language=None),
        AudioStreamInfo(1, 2, "flac", 44100, 2, language=None),
    ]
    with pytest.raises(StreamSelectionError):
        select_audio_stream(two_untagged, lang="eng")


def test_probe_corrupt_file_raises_probe_error(tmp_path: Path, ffprobe_path: Path) -> None:
    bad = tmp_path / "corrupt.mp4"
    bad.write_bytes(b"not a valid media container at all")
    with pytest.raises(ProbeError) as exc_info:
        probe_media(ffprobe_path, bad)
    # With -v error, ffprobe includes the diagnostic reason from stderr.
    msg = str(exc_info.value)
    assert "corrupt.mp4" in msg
    assert ":" in msg
    assert len(msg.split(":", 1)[1].strip()) > 0
