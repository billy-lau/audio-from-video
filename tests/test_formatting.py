"""Tests for formatting, Section 3 concrete examples, track numbers, and featuring."""

from __future__ import annotations

from pathlib import Path

from audio_from_video.formatting import (
    build_id3_title,
    build_relative_path,
    format_bilingual,
    format_track_prefix,
    has_non_latin,
    join_featuring,
    normalize_featuring,
)
from audio_from_video.models import MatchStatus, TrackMetadata


def test_has_non_latin_ignores_hawaiian_okina_and_micro_sign() -> None:
    assert has_non_latin("Israel Kamakawiwo\u02bbole") is False
    assert has_non_latin("\u00b5-Ziq") is False
    assert has_non_latin("Beyonc\u00e9") is False
    assert has_non_latin("張學友") is True
    assert has_non_latin("방탄소년단") is True
    assert has_non_latin("Кино") is True


def test_section3_concrete_examples() -> None:
    """Verify all 12 concrete examples from design_spec Section 3 verbatim."""
    cases = [
        # 1. Standard track
        (
            TrackMetadata(
                artist="Daft Punk",
                title="One More Time",
                album="Discovery",
                album_artist="Daft Punk",
                track_number=1,
                match_status=MatchStatus.EXACT,
            ),
            "orig",
            "Daft Punk/Discovery/01 Daft Punk - One More Time.mp3",
            "One More Time",
        ),
        # 2. Single featuring artist
        (
            TrackMetadata(
                artist="Daft Punk",
                featuring=["Pharrell Williams"],
                title="Lose Yourself to Dance",
                album="Random Access Memories",
                album_artist="Daft Punk",
                track_number=6,
                match_status=MatchStatus.EXACT,
            ),
            "orig",
            (
                "Daft Punk/Random Access Memories/"
                "06 Daft Punk (ft. Pharrell Williams) - Lose Yourself to Dance.mp3"
            ),
            "Lose Yourself to Dance (ft. Pharrell Williams)",
        ),
        # 3. Two featuring artists
        (
            TrackMetadata(
                artist="Daft Punk",
                featuring=["Pharrell Williams", "Nile Rodgers"],
                title="Get Lucky",
                album="Random Access Memories",
                album_artist="Daft Punk",
                track_number=8,
                match_status=MatchStatus.EXACT,
            ),
            "orig",
            (
                "Daft Punk/Random Access Memories/"
                "08 Daft Punk (ft. Pharrell Williams & Nile Rodgers) - Get Lucky.mp3"
            ),
            "Get Lucky (ft. Pharrell Williams & Nile Rodgers)",
        ),
        # 4. Three featuring artists (Oxford comma!)
        (
            TrackMetadata(
                artist="Artist",
                featuring=["A", "B", "C"],
                title="Song",
                album="Album",
                album_artist="Artist",
                track_number=3,
                match_status=MatchStatus.EXACT,
            ),
            "orig",
            "Artist/Album/03 Artist (ft. A, B, and C) - Song.mp3",
            "Song (ft. A, B, and C)",
        ),
        # 5. Non-Latin w/ English alias
        (
            TrackMetadata(
                artist=format_bilingual("Jacky Cheung", "張學友"),
                title=format_bilingual("The Goodbye Kiss", "吻別"),
                album=format_bilingual("The Goodbye Kiss", "吻別"),
                album_artist=format_bilingual("Jacky Cheung", "張學友"),
                track_number=1,
                match_status=MatchStatus.EXACT,
            ),
            "orig",
            (
                "Jacky Cheung {張學友}/The Goodbye Kiss {吻別}/"
                "01 Jacky Cheung {張學友} - The Goodbye Kiss {吻別}.mp3"
            ),
            "The Goodbye Kiss {吻別}",
        ),
        # 6. Accented Latin name (omit {orig})
        (
            TrackMetadata(
                artist=format_bilingual("Beyonce", "Beyoncé"),
                featuring=["Jack White"],
                title="Don't Hurt Yourself",
                album="Lemonade",
                album_artist=format_bilingual("Beyonce", "Beyoncé"),
                track_number=3,
                match_status=MatchStatus.EXACT,
            ),
            "orig",
            "Beyoncé/Lemonade/03 Beyoncé (ft. Jack White) - Don't Hurt Yourself.mp3",
            "Don't Hurt Yourself (ft. Jack White)",
        ),
        # 7. Multi-disc album (disc 2)
        (
            TrackMetadata(
                artist="The Beatles",
                title="Birthday",
                album="The Beatles",
                album_artist="The Beatles",
                track_number=1,
                disc_number=2,
                disc_total=2,
                match_status=MatchStatus.EXACT,
            ),
            "orig",
            "The Beatles/The Beatles/2-01 The Beatles - Birthday.mp3",
            "Birthday",
        ),
        # 8. Illegal chars in artist/title
        (
            TrackMetadata(
                artist="AC/DC",
                title="Hells Bells: Live?",
                album="Back in Black",
                album_artist="AC/DC",
                track_number=1,
                match_status=MatchStatus.EXACT,
            ),
            "orig",
            "AC-DC/Back in Black/01 AC-DC - Hells Bells - Live.mp3",
            "Hells Bells: Live?",
        ),
        # 9. Leading dots in album name
        (
            TrackMetadata(
                artist="Metallica",
                title="...And Justice for All",
                album="...And Justice for All",
                album_artist="Metallica",
                track_number=2,
                match_status=MatchStatus.EXACT,
            ),
            "orig",
            "Metallica/And Justice for All/02 Metallica - ...And Justice for All.mp3",
            "...And Justice for All",
        ),
        # 10. All-illegal album name
        (
            TrackMetadata(
                artist="XXXTENTACION",
                title="SAD!",
                album="?",
                album_artist="XXXTENTACION",
                track_number=4,
                match_status=MatchStatus.EXACT,
            ),
            "orig",
            "XXXTENTACION/_/04 XXXTENTACION - SAD!.mp3",
            "SAD!",
        ),
        # 11. Partial match (no album)
        (
            TrackMetadata(
                artist="Daft Punk",
                title="Unreleased Demo",
                album=None,
                album_artist="Daft Punk",
                track_number=None,
                match_status=MatchStatus.PARTIAL,
            ),
            "orig",
            "Daft Punk/Unknown Album/Daft Punk - Unreleased Demo.mp3",
            "Unreleased Demo",
        ),
        # 12. Unmatched track
        (
            TrackMetadata(match_status=MatchStatus.UNMATCHED),
            "VID_0042",
            "_unmatched/VID_0042.mp3",
            "",
        ),
    ]

    for meta, stem, expected_rel, expected_tit2 in cases:
        rel = build_relative_path(meta, original_stem=stem)
        assert rel == Path(expected_rel), f"Got {rel.as_posix()!r}, expected {expected_rel!r}"
        if meta.title:
            assert build_id3_title(meta.title, meta.featuring) == expected_tit2


def test_format_track_prefix() -> None:
    assert format_track_prefix(None) is None
    assert format_track_prefix(0) is None
    assert format_track_prefix(1) == "01"
    assert format_track_prefix(12) == "12"
    assert format_track_prefix(100) == "100"
    assert format_track_prefix(1, disc_number=1, disc_total=1) == "01"
    assert format_track_prefix(1, disc_number=1, disc_total=2) == "1-01"
    assert format_track_prefix(1, disc_number=2, disc_total=2) == "2-01"
    assert format_track_prefix(5, disc_number=2, disc_total=None) == "2-05"


def test_featuring_extraction_dedup_and_coheadliners() -> None:
    # Keep co-headliners ('&', 'and', 'vs.') in artist.
    artist, title, feat = normalize_featuring("Simon & Garfunkel", "Bridge Over Troubled Water")
    assert artist == "Simon & Garfunkel"
    assert title == "Bridge Over Troubled Water"
    assert feat == []

    artist, title, feat = normalize_featuring("Hall and Oates", "Maneater")
    assert artist == "Hall and Oates"
    assert feat == []

    artist, title, feat = normalize_featuring("Daft Punk vs. Justice", "Mashup")
    assert artist == "Daft Punk vs. Justice"
    assert feat == []

    # Deduplicate across artist and title without destructively splitting '&'.
    artist, title, feat = normalize_featuring(
        "Daft Punk feat. Pharrell Williams",
        "Get Lucky (ft. Pharrell Williams & Nile Rodgers)",
    )
    assert artist == "Daft Punk"
    assert title == "Get Lucky"
    assert feat == ["Pharrell Williams & Nile Rodgers"]
    assert join_featuring(feat) == "Pharrell Williams & Nile Rodgers"

    # Structured featuring lists (e.g., from MusicBrainz artist credits) join with Oxford comma.
    _, _, structured = normalize_featuring(
        "Artist", "Song", extra_featuring=["Alice", "Bob", "Carol"]
    )
    assert structured == ["Alice", "Bob", "Carol"]
    assert join_featuring(structured) == "Alice, Bob, and Carol"

    artist, title, feat = normalize_featuring("Artist", "Song f/ Guest")
    assert title == "Song"
    assert feat == ["Guest"]


def test_inline_feat_stops_before_brackets_and_edition_tails() -> None:
    artist, title, feat = normalize_featuring(
        "Artist", "One More Time feat. Romanthony (Radio Edit)"
    )
    assert artist == "Artist"
    assert title == "One More Time (Radio Edit)"
    assert feat == ["Romanthony"]

    meta = TrackMetadata(
        artist=artist,
        featuring=feat,
        title=title,
        album="Discovery",
        album_artist=artist,
        match_status=MatchStatus.EXACT,
    )
    assert build_relative_path(meta, original_stem="orig") == Path(
        "Artist/Discovery/Artist (ft. Romanthony) - One More Time (Radio Edit).mp3"
    )
    assert build_id3_title(title or "", feat) == "One More Time (Radio Edit) (ft. Romanthony)"

    # Hyphenated names like Jay-Z stay intact, while spaced " - " edition tails are preserved.
    _, t_jayz, f_jayz = normalize_featuring("Artist", "Song feat. Jay-Z (Radio Edit)")
    assert t_jayz == "Song (Radio Edit)"
    assert f_jayz == ["Jay-Z"]

    _, t_rem, f_rem = normalize_featuring(
        "Artist", "One More Time feat. Romanthony - 2012 Remaster"
    )
    assert t_rem == "One More Time - 2012 Remaster"
    assert f_rem == ["Romanthony"]


def test_featuring_names_with_commas_ampersands_and_and_stay_intact() -> None:
    for guest in ("Earth, Wind & Fire", "Tyler, the Creator", "Simon and Garfunkel"):
        artist, title, feat = normalize_featuring("Artist", f"Song (feat. {guest})")
        assert artist == "Artist"
        assert title == "Song"
        assert feat == [guest]
        assert join_featuring(feat) == guest
        assert build_id3_title(title or "", feat) == f"Song (ft. {guest})"

    # Structured extra_featuring entries are also never re-split.
    _, _, multi = normalize_featuring(
        "Artist",
        "Song",
        extra_featuring=["Earth, Wind & Fire", "Tyler, the Creator"],
    )
    assert multi == ["Earth, Wind & Fire", "Tyler, the Creator"]
    assert join_featuring(multi) == "Earth, Wind & Fire & Tyler, the Creator"


def test_bilingual_and_script_detection() -> None:
    assert has_non_latin("張學友") is True
    assert has_non_latin("Beyoncé") is False
    assert has_non_latin("Кино") is True

    assert format_bilingual("Beyonce", "Beyoncé") == "Beyoncé"
    assert format_bilingual("DAFT PUNK", "Daft Punk") == "Daft Punk"
    assert format_bilingual("Jacky Cheung", "張學友") == "Jacky Cheung {張學友}"
    assert format_bilingual("Jacky Cheung", "Jacky Cheung {張學友}") == "Jacky Cheung {張學友}"
    assert format_bilingual(None, "張學友") == "張學友"


def test_bilingual_split_trim_simplified_and_folder_grouping() -> None:
    from audio_from_video.formatting import (
        folder_grouping_key,
        split_bilingual,
        to_simplified_chinese,
        trim_bilingual_context,
    )

    assert split_bilingual("Zhao Chuan {趙傳}") == ("Zhao Chuan", "趙傳")
    assert split_bilingual("{吻別}") == (None, "吻別")
    assert split_bilingual("Daft Punk") is None
    assert split_bilingual("Song {Acoustic}") is None
    assert split_bilingual("Artist {UK}") is None
    assert trim_bilingual_context("Song {Live} at Wembley") == "Song {Live} at Wembley"
    assert trim_bilingual_context("Love {愛} in the Rain") == "Love {愛} in the Rain"
    assert trim_bilingual_context("Song {歌} at Wembley") == "Song {歌}"
    assert trim_bilingual_context(
        "Hai Kuo Tian Kong {海阔天空} at I Am A Singer 3 {我是歌手 3}"
    ) == ("Hai Kuo Tian Kong {海阔天空}")
    assert trim_bilingual_context("Song {歌} (Live)") == "Song {歌} (Live)"
    assert to_simplified_chinese("我是一隻小小鳥 Daft Punk") == "我是一只小小鸟 Daft Punk"
    assert to_simplified_chinese("Beyoncé") == "Beyoncé"
    assert folder_grouping_key("Zhao Chuan {赵传}") == folder_grouping_key("zhao chuan {趙傳}")
