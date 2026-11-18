"""Tests for fresh ID3v2.3 tag replacement, TORY, AC/DC, and Pillow artwork handling."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from mutagen.id3 import (
    CHAP,
    COMM,
    ID3,
    TIT2,
    Encoding,
    ID3NoHeaderError,
)
from PIL import Image

from audio_from_video.models import MatchStatus, TrackMetadata
from audio_from_video.tagger import (
    EXIF_ORIENTATION_TAG,
    VerificationError,
    _is_progressive_jpeg,
    normalize_id3_date,
    process_artwork,
    verify_tagged_mp3,
    write_tags,
)
from conftest import make_raw_mp3


def _make_jpeg_bytes(
    size: tuple[int, int] = (500, 500),
    *,
    mode: str = "RGB",
    color: tuple[int, ...] | int = (40, 120, 200),
    progressive: bool = False,
    exif_orientation: int | None = None,
) -> bytes:
    img = Image.new(mode, size, color=color)
    buf = io.BytesIO()
    save_kwargs: dict[str, object] = {"format": "JPEG", "quality": 90, "progressive": progressive}
    if exif_orientation is not None:
        exif = img.getexif()
        exif[EXIF_ORIENTATION_TAG] = exif_orientation
        save_kwargs["exif"] = exif.tobytes()
    img.save(buf, **save_kwargs)
    return buf.getvalue()


def test_fresh_id3v23_replaces_stale_v24_v1_and_chap_frames(tmp_path: Path) -> None:
    mp3_file = make_raw_mp3(tmp_path / "stale.mp3")

    # Seed file with a stale ID3v2.4 header containing CHAP + COMM plus an ID3v1 footer.
    stale = ID3()
    stale.add(TIT2(encoding=Encoding.UTF8, text=["Old Title"]))
    stale.add(COMM(encoding=Encoding.UTF8, lang="eng", desc="", text=["Old Comment"]))
    stale.add(
        CHAP(
            element_id="ch1",
            start_time=0,
            end_time=500,
            start_offset=0,
            end_offset=0,
            sub_frames=[TIT2(encoding=Encoding.UTF8, text=["Chapter 1"])],
        )
    )
    stale.save(str(mp3_file), v2_version=4, v1=2)
    assert ID3(str(mp3_file)).version == (2, 4, 0)
    assert mp3_file.read_bytes()[-128:-125] == b"TAG"

    meta = TrackMetadata(
        artist="AC/DC",
        album_artist="AC/DC",
        featuring=["Guest Artist"],
        title="Hells Bells: Live?",
        album="Back in Black",
        track_number=1,
        track_total=10,
        disc_number=1,
        disc_total=1,
        date="2003-04-15",
        original_year="1980",
        genre="Hard Rock",
        mb_recording_id="11111111-2222-3333-4444-555555555555",
        mb_release_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        mb_artist_id="bbbbbbbb-cccc-dddd-eeee-ffffffffffff",
        mb_release_group_id="cccccccc-dddd-eeee-ffff-000000000000",
        match_status=MatchStatus.EXACT,
    )
    art = _make_jpeg_bytes((400, 400))

    wrote, warnings = write_tags(mp3_file, meta, artwork_bytes=art)
    assert wrote is True
    assert warnings == []
    verify_tagged_mp3(mp3_file, expect_id3=True)

    # Read raw bytes to prove ID3v1 footer was completely removed.
    assert mp3_file.read_bytes()[-128:-125] != b"TAG"

    loaded = ID3(str(mp3_file))
    assert loaded.version == (2, 3, 0)
    assert not loaded.getall("CHAP")
    assert not loaded.getall("COMM")

    # Verify AC/DC preserved verbatim as a single element (not split on '/').
    assert loaded["TPE1"].text == ["AC/DC"]
    assert loaded["TPE2"].text == ["AC/DC"]

    # Verify iTunes-style featuring in TIT2.
    assert loaded["TIT2"].text == ["Hells Bells: Live? (ft. Guest Artist)"]
    assert loaded["TALB"].text == ["Back in Black"]
    assert loaded["TRCK"].text == ["1/10"]
    assert loaded["TPOS"].text == ["1/1"]
    assert loaded["TCON"].text == ["Hard Rock"]

    # Verify default translated view (mutagen upgrades TYER/TDAT -> TDRC and TORY -> TDOR on read).
    assert str(loaded["TDRC"]) == "2003-04-15"
    assert str(loaded["TDOR"]) == "1980"

    # Verify ACTUAL raw on-disk ID3v2.3 binary frames (translate=False) are TYER, TDAT, and TORY.
    raw_v23 = ID3(str(mp3_file), translate=False)
    assert raw_v23.version == (2, 3, 0)
    assert str(raw_v23["TYER"]) == "2003"
    assert str(raw_v23["TDAT"]) == "1504"
    assert str(raw_v23["TORY"]) == "1980"

    # Verify MusicBrainz IDs.
    ufids = loaded.getall("UFID")
    assert len(ufids) == 1
    assert ufids[0].owner == "http://musicbrainz.org"
    assert ufids[0].data == b"11111111-2222-3333-4444-555555555555"

    txxx_map = {frame.desc: frame.text for frame in loaded.getall("TXXX")}
    assert txxx_map["MusicBrainz Album Id"] == ["aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"]
    assert txxx_map["MusicBrainz Artist Id"] == ["bbbbbbbb-cccc-dddd-eeee-ffffffffffff"]
    assert txxx_map["MusicBrainz Release Group Id"] == ["cccccccc-dddd-eeee-ffff-000000000000"]

    # Verify APIC pass-through embeds exact original compliant JPEG bytes!
    apics = loaded.getall("APIC")
    assert len(apics) == 1
    assert apics[0].data == art


def test_zero_frame_unmatched_track_strips_prior_tags(tmp_path: Path) -> None:
    mp3_file = make_raw_mp3(tmp_path / "empty.mp3")
    stale = ID3()
    stale.add(TIT2(encoding=Encoding.UTF8, text=["Stale"]))
    stale.save(str(mp3_file), v2_version=4, v1=2)

    wrote, warnings = write_tags(mp3_file, TrackMetadata(match_status=MatchStatus.UNMATCHED))
    assert wrote is False
    assert warnings == []
    assert mp3_file.read_bytes()[-128:-125] != b"TAG"
    with pytest.raises(ID3NoHeaderError):
        ID3(str(mp3_file))

    verify_tagged_mp3(mp3_file, expect_id3=False)


def test_artwork_compliant_baseline_jpeg_passes_through_untouched() -> None:
    raw = _make_jpeg_bytes((800, 800), progressive=False)
    res = process_artwork(raw)
    assert res.warning is None
    assert res.passthrough is True
    assert res.data == raw  # Exact byte-for-byte identity!


def test_artwork_progressive_jpeg_reencoded_to_baseline() -> None:
    raw = _make_jpeg_bytes((600, 600), progressive=True)
    assert _is_progressive_jpeg(raw) is True
    res = process_artwork(raw)
    assert res.warning is None
    assert res.passthrough is False
    assert res.data is not None
    assert _is_progressive_jpeg(res.data) is False


def test_artwork_oversized_resized_without_upscaling() -> None:
    raw = _make_jpeg_bytes((1600, 1200))
    res = process_artwork(raw)
    assert res.warning is None
    assert res.passthrough is False
    assert res.data is not None
    with Image.open(io.BytesIO(res.data)) as out_img:
        assert out_img.format == "JPEG"
        assert out_img.size == (1000, 750)


def test_artwork_exif_orientation_transposed() -> None:
    # 400x200 image with EXIF Orientation=6 (rotate 90 CW -> 200x400).
    raw = _make_jpeg_bytes((400, 200), exif_orientation=6)
    res = process_artwork(raw)
    assert res.warning is None
    assert res.passthrough is False
    assert res.data is not None
    with Image.open(io.BytesIO(res.data)) as out_img:
        assert out_img.size == (200, 400)


def test_artwork_png_alpha_composited_on_white() -> None:
    # Fully transparent PNG -> should composite onto solid white (255, 255, 255).
    img = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    res = process_artwork(buf.getvalue())
    assert res.warning is None
    assert res.passthrough is False
    assert res.data is not None
    with Image.open(io.BytesIO(res.data)) as out_img:
        assert out_img.mode == "RGB"
        pixel = out_img.getpixel((150, 150))
        assert isinstance(pixel, tuple)
        assert all(channel >= 250 for channel in pixel)


def test_artwork_cmyk_converted_to_rgb() -> None:
    raw = _make_jpeg_bytes((300, 300), mode="CMYK", color=(0, 100, 100, 0))
    res = process_artwork(raw)
    assert res.warning is None
    assert res.passthrough is False
    assert res.data is not None
    with Image.open(io.BytesIO(res.data)) as out_img:
        assert out_img.mode == "RGB"


def test_corrupt_truncated_and_empty_artwork_are_non_fatal(tmp_path: Path) -> None:
    mp3_file = make_raw_mp3(tmp_path / "resilient.mp3")
    meta = TrackMetadata(artist="Daft Punk", title="One More Time", album="Discovery")

    # 1. Completely corrupt bytes.
    wrote, warnings = write_tags(mp3_file, meta, artwork_bytes=b"not an image at all")
    assert wrote is True
    assert len(warnings) == 1
    assert "without APIC" in warnings[0]
    loaded = ID3(str(mp3_file))
    assert loaded["TPE1"].text == ["Daft Punk"]
    assert loaded.getall("APIC") == []

    # 2. Truncated JPEG (first 40% of a real JPEG stream).
    valid_jpeg = _make_jpeg_bytes((600, 600))
    truncated = valid_jpeg[: len(valid_jpeg) // 3]
    res_trunc = process_artwork(truncated)
    assert res_trunc.data is None
    assert res_trunc.warning is not None

    # 3. Empty bytes.
    res_empty = process_artwork(b"")
    assert res_empty.data is None
    assert res_empty.warning is not None


def test_verify_tagged_mp3_rejects_empty_and_corrupt(tmp_path: Path) -> None:
    empty = tmp_path / "zero.mp3"
    empty.write_bytes(b"")
    with pytest.raises(VerificationError):
        verify_tagged_mp3(empty)

    garbage = tmp_path / "garbage.mp3"
    garbage.write_bytes(b"not an mp3 file" * 20)
    with pytest.raises(VerificationError):
        verify_tagged_mp3(garbage)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("20010312", "2001-03-12"),
        ("2001-03-12", "2001-03-12"),
        ("2001-03-12T08:00:00Z", "2001-03-12"),
        ("2001/3/5", "2001-03-05"),
        ("200103", "2001-03"),
        ("2001-03", "2001-03"),
        ("2001", "2001"),
        ("March 2001", "2001-03"),
        ("Mar 12, 2001", "2001-03-12"),
        ("12 March 2001", "2001-03-12"),
        ("20019999", None),
        ("2001-02-30", None),
        ("sometime in the 90s", None),
        ("", None),
    ],
)
def test_normalize_id3_date(raw: str, expected: str | None) -> None:
    assert normalize_id3_date(raw) == expected


def test_compact_ytdlp_date_and_unparseable_date_tagging(tmp_path: Path) -> None:
    # Compact YYYYMMDD date writes valid TYER + TDAT and round-trips through mutagen.
    mp3_ok = make_raw_mp3(tmp_path / "compact_date.mp3")
    meta_ok = TrackMetadata(
        artist="Daft Punk",
        title="One More Time",
        album="Discovery",
        date="20010312",
        original_year="20010312",
    )
    wrote, warnings = write_tags(mp3_ok, meta_ok)
    assert wrote is True
    assert warnings == []

    raw_v23 = ID3(str(mp3_ok), translate=False)
    assert str(raw_v23["TYER"]) == "2001"
    assert str(raw_v23["TDAT"]) == "1203"
    assert str(raw_v23["TORY"]) == "2001"

    translated = ID3(str(mp3_ok))
    assert str(translated["TDRC"]) == "2001-03-12"
    assert str(translated["TDOR"]) == "2001"

    # Unparseable date is dropped with a warning instead of writing corrupt TYER.
    mp3_bad = make_raw_mp3(tmp_path / "bad_date.mp3")
    meta_bad = TrackMetadata(artist="Daft Punk", title="One More Time", date="sometime in 2001?")
    wrote_bad, bad_warnings = write_tags(mp3_bad, meta_bad)
    assert wrote_bad is True
    assert any("Unparseable date tag" in w for w in bad_warnings)
    raw_bad = ID3(str(mp3_bad), translate=False)
    assert "TYER" not in raw_bad
    assert "TDAT" not in raw_bad


def test_decompression_bomb_artwork_is_non_fatal(tmp_path: Path) -> None:
    # Generate a real 14000x14000 1-bit PNG (~24 KB compressed) exceeding Pillow's
    # DecompressionBombError threshold (196M pixels > 178.9M pixels).
    bomb = Image.new("1", (14000, 14000), 0)
    buf = io.BytesIO()
    bomb.save(buf, format="PNG")
    bomb_bytes = buf.getvalue()

    res = process_artwork(bomb_bytes)
    assert res.data is None
    assert res.warning is not None
    assert "without APIC" in res.warning

    mp3_file = make_raw_mp3(tmp_path / "bomb_safe.mp3")
    meta = TrackMetadata(artist="Daft Punk", title="One More Time", album="Discovery")
    wrote, warnings = write_tags(mp3_file, meta, artwork_bytes=bomb_bytes)
    assert wrote is True
    assert len(warnings) == 1
    assert "without APIC" in warnings[0]
    loaded = ID3(str(mp3_file))
    assert loaded["TPE1"].text == ["Daft Punk"]
    assert loaded.getall("APIC") == []
