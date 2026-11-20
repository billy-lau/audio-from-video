"""Tests for parser, high-precision yt-dlp downloader-tag detection, and controls."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_from_video.models import ProbeResult
from audio_from_video.parser import (
    detect_downloader_signals,
    infer_unpadded_track_paths,
    is_edition_suffix,
    is_hintless_stem,
    parse_hints,
)


def _probe(path: str | Path, tags: dict[str, str] | None = None) -> ProbeResult:
    return ProbeResult(
        path=Path(path),
        duration_seconds=200.0,
        format_name="mov,mp4",
        container_tags=tags or {},
        audio_streams=[],
    )


@pytest.mark.parametrize(
    ("artist", "title", "extra"),
    [
        ("Roxy Music", "More Than This", {}),
        ("Roxy Music", "More Than This - 2012 Remaster", {}),
        ("Roxy Music", "Love Is the Drug - Radio Edit", {}),
        ("TV on the Radio", "Wolf Like Me", {}),
        ("Channel Tres", "Controller", {}),
        ("Official Hige Dandism", "Pretender", {}),
        ("Records & Tapes", "Live at Home - Live", {}),
        (
            "Berliner Philharmoniker",
            "Symphony No. 5 - Allegro con brio",
            {"album": "Beethoven: Symphonies Nos. 5 & 7", "track": "1"},
        ),
        (
            "Berliner Philharmoniker",
            "Symphony No. 5 - Allegro con brio",
            {"album_artist": "Ludwig van Beethoven"},
        ),
        (
            "London Symphony Orchestra",
            "Piano Concerto No. 2 - Adagio sostenuto",
            {"track": "2/3"},
        ),
    ],
)
def test_positive_controls_remain_trusted(artist: str, title: str, extra: dict[str, str]) -> None:
    tags = {"artist": artist, "title": title, **extra}
    signals, split_pair = detect_downloader_signals(tags)
    assert signals == []
    assert split_pair is None

    hints = parse_hints(_probe("/music/Artist/Album/track.mp4", tags))
    assert hints.downloader_signals == []
    assert hints.trusted_artist is True
    assert hints.trusted_title is True
    assert hints.artist == artist
    assert hints.title == title


def test_signal1_video_platform_url() -> None:
    tags = {
        "artist": "SomeChannelName",
        "title": "Cool Video Without Dash",
        "purl": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    }
    signals, _ = detect_downloader_signals(tags)
    assert "url" in signals

    hints = parse_hints(_probe("/tmp/video.mp4", tags))
    assert hints.trusted_artist is False
    assert hints.trusted_title is False
    assert hints.album is None


def test_signal2_topic_suffix_yields_trusted_artist() -> None:
    tags = {
        "artist": "Daft Punk - Topic",
        "album_artist": "Daft Punk - Topic",
        "title": "One More Time",
        "album": "Discovery",
        "purl": "https://www.youtube.com/watch?v=abc12345678",
    }
    hints = parse_hints(_probe("/downloads/track.mp4", tags))
    assert "topic_suffix" in hints.downloader_signals
    assert "url" in hints.downloader_signals
    assert hints.artist == "Daft Punk"
    assert hints.album_artist == "Daft Punk"
    assert hints.album == "Discovery"
    assert hints.trusted_artist is True
    assert hints.trusted_title is True


def test_signal2_vevo_suffix_stays_untrusted() -> None:
    tags = {
        "artist": "DaftPunkVEVO",
        "title": "One More Time (Official Video)",
    }
    hints = parse_hints(_probe("/downloads/clip.mp4", tags))
    assert "vevo_suffix" in hints.downloader_signals
    assert hints.trusted_artist is False
    assert hints.album is None


def test_signal3_title_mismatch_and_edition_tail_peeling() -> None:
    # Stripped-URL yt-dlp file (no album, album_artist, or track).
    tags = {
        "artist": "RandomUploader99",
        "title": "Daft Punk - One More Time - 2012 Remaster",
    }
    signals, split_pair = detect_downloader_signals(tags)
    assert signals == ["title_mismatch"]
    assert split_pair == ("Daft Punk", "One More Time")

    hints = parse_hints(_probe("/downloads/VID_0042.mp4", tags))
    assert hints.artist == "Daft Punk"
    assert hints.title == "One More Time"
    assert hints.swapped_candidate == ("One More Time", "Daft Punk")
    assert hints.original_embedded_pair == (
        "RandomUploader99",
        "Daft Punk - One More Time - 2012 Remaster",
    )
    assert hints.trusted_artist is False


def test_signal3_loose_classical_clip_preserves_original_pair_for_fallback() -> None:
    # Loose classical clip with NO album/album_artist/track tags triggers Signal 3,
    # keeping original_embedded_pair so online search can still retry it safely,
    # while never promoting bogus artist folder offline.
    tags = {
        "artist": "Berliner Philharmoniker",
        "title": "Symphony No. 5 - Allegro con brio",
    }
    hints = parse_hints(_probe("/downloads/clip.mp4", tags))
    assert "title_mismatch" in hints.downloader_signals
    assert hints.original_embedded_pair == (
        "Berliner Philharmoniker",
        "Symphony No. 5 - Allegro con brio",
    )
    assert hints.trusted_artist is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2012 Remaster", True),
        ("Remastered 2009", True),
        ("Radio Edit", True),
        ("Live at Budokan", True),
        ("Acoustic Version", True),
        ("Part 2", True),
        ("Pt. 1", True),
        ("Instrumental", True),
        ("Bonus Track", True),
        ("Allegro con brio", False),
        ("One More Time", False),
    ],
)
def test_edition_suffix_detection(text: str, expected: bool) -> None:
    assert is_edition_suffix(text) is expected


def test_filename_cleanup_track_numbers_and_featuring() -> None:
    probe = _probe(
        "/library/Daft Punk/Random Access Memories/"
        "08 - Daft Punk - Get Lucky (feat. Pharrell Williams & Nile Rodgers) "
        "[Official Audio] [dQw4w9WgXcQ].mp4"
    )
    hints = parse_hints(probe)
    assert hints.track_number == 8
    assert hints.artist == "Daft Punk"
    assert hints.title == "Get Lucky"
    assert hints.featuring == ["Pharrell Williams & Nile Rodgers"]
    assert hints.parent_album_hint == "Random Access Memories"
    assert hints.parent_artist_hint == "Daft Punk"
    assert hints.trusted_artist is True
    assert hints.trusted_title is True
    assert hints.is_hintless is False


@pytest.mark.parametrize(
    ("filename", "expected_artist", "expected_title", "expected_track"),
    [
        ("50 Cent - In Da Club.mp4", "50 Cent", "In Da Club", None),
        ("3 Doors Down - Kryptonite.mp4", "3 Doors Down", "Kryptonite", None),
        ("7 Rings - Ariana Grande.mp4", "7 Rings", "Ariana Grande", None),
        ("311 - Amber.mp4", "311", "Amber", None),
        ("21 Savage - Bank Account.mp4", "21 Savage", "Bank Account", None),
        ("100 gecs - money machine.mp4", "100 gecs", "money machine", None),
        ("070 Shake - Guilty Conscience.mp4", "070 Shake", "Guilty Conscience", None),
        ("01 Daft Punk - One More Time.mp4", "Daft Punk", "One More Time", 1),
        ("02 Aerodynamic.mp4", None, "Aerodynamic", 2),
        ("1. Daft Punk - One More Time.mp4", "Daft Punk", "One More Time", 1),
        ("12. Too Long.mp4", None, "Too Long", 12),
        ("1 - Daft Punk - One More Time.mp4", "Daft Punk", "One More Time", 1),
        ("1 - Aerodynamic.mp4", None, "Aerodynamic", 1),
        ("01-Aerodynamic.mp4", None, "Aerodynamic", 1),
    ],
)
def test_numeric_leading_artists_vs_explicit_track_prefixes(
    filename: str,
    expected_artist: str | None,
    expected_title: str,
    expected_track: int | None,
) -> None:
    hints = parse_hints(_probe(f"/downloads/{filename}"))
    assert hints.artist == expected_artist
    assert hints.title == expected_title
    assert hints.track_number == expected_track


def test_embedded_titles_cleaned_and_split_with_and_without_artist_tag() -> None:
    # 1. Filename + raw video-style embedded title without artist tag: no duplicate artist!
    h1 = parse_hints(
        _probe(
            "/downloads/Daft Punk - Da Funk.mp4",
            {"title": "Daft Punk - Da Funk (Official Video)"},
        )
    )
    assert h1.artist == "Daft Punk"
    assert h1.title == "Da Funk"
    assert h1.trusted_artist is True
    assert h1.trusted_title is True

    # 2. Cryptic filename + embedded title "Artist - Title (Official Video)" without artist tag.
    h2 = parse_hints(
        _probe(
            "/downloads/titleonly.mp4",
            {"title": "Daft Punk - Around the World (Official Video)"},
        )
    )
    assert h2.artist == "Daft Punk"
    assert h2.title == "Around the World"
    assert h2.trusted_artist is True
    assert h2.trusted_title is True

    # 3. Embedded artist AND redundant "<artist> - <title> (Official Video)" in title tag.
    h3 = parse_hints(
        _probe(
            "/downloads/clip.mp4",
            {"artist": "Daft Punk", "title": "Daft Punk - Da Funk (Official Video)"},
        )
    )
    assert h3.artist == "Daft Punk"
    assert h3.title == "Da Funk"

    # 4. Embedded artist + video noise on title tag.
    h4 = parse_hints(
        _probe(
            "/downloads/clip.mp4",
            {"artist": "Daft Punk", "title": "One More Time [Official Audio]"},
        )
    )
    assert h4.artist == "Daft Punk"
    assert h4.title == "One More Time"

    # 5. Embedded title with edition tail and NO artist tag must NOT split as artist!
    h5 = parse_hints(
        _probe(
            "/downloads/Roxy Music - More Than This.mp4",
            {"title": "More Than This - 2012 Remaster"},
        )
    )
    assert h5.artist == "Roxy Music"
    assert h5.title == "More Than This - 2012 Remaster"

    # 6. Embedded title containing " - " must NOT replace an artist already parsed from filename!
    h6 = parse_hints(
        _probe(
            "/downloads/Beethoven - Fifth.mp4",
            {"title": "Symphony No. 5 - Allegro con brio"},
        )
    )
    assert h6.artist == "Beethoven"
    assert h6.title == "Symphony No. 5 - Allegro con brio"
    assert h6.trusted_artist is True
    assert h6.trusted_title is True


def test_infer_unpadded_track_paths_from_sibling_context() -> None:
    # 1. Full 01..12 album folder alongside numeric-leading artists in the same folder:
    # tracks 10..12 are confirmed, while 3, 7, and 50 stay untouched!
    p10 = Path("/album/10 Daft Punk - Track10.mp4")
    p11 = Path("/album/11 Daft Punk - Track11.mp4")
    p12 = Path("/album/12 Something.mp4")
    p50 = Path("/album/50 Cent - In Da Club.mp4")
    album_plus_loose = [
        *[Path(f"/album/{i:02d} Daft Punk - Track{i:02d}.mp4") for i in range(1, 10)],
        p10,
        p11,
        p12,
        p50,
        Path("/album/3 Doors Down - Kryptonite.mp4"),
        Path("/album/7 Rings - Ariana Grande.mp4"),
    ]
    confirmed = infer_unpadded_track_paths(album_plus_loose)
    assert confirmed == {p10, p11, p12}

    h10 = parse_hints(_probe(p10), allow_unpadded_space_track=p10 in confirmed)
    assert h10.track_number == 10
    assert h10.artist == "Daft Punk"
    assert h10.title == "Track10"

    h12 = parse_hints(_probe(p12), allow_unpadded_space_track=p12 in confirmed)
    assert h12.track_number == 12
    assert h12.artist is None
    assert h12.title == "Something"

    h50 = parse_hints(_probe(p50), allow_unpadded_space_track=p50 in confirmed)
    assert h50.track_number is None
    assert h50.artist == "50 Cent"
    assert h50.title == "In Da Club"

    # 2. Three consecutive numeric-leading artists must NOT be treated as a track run!
    assert (
        infer_unpadded_track_paths(
            [
                Path("/loose/2 Chainz - Birthday Song.mp4"),
                Path("/loose/3 Doors Down - Kryptonite.mp4"),
                Path("/loose/4 Non Blondes - What's Up.mp4"),
            ]
        )
        == set()
    )

    # 3. Unpadded undashed album with a gap (1, 2, 3, 10, 11) confirms all 5 undashed tracks
    # while leaving a colliding dashed artist ("2 Chainz - Birthday Song.mp4") untouched!
    hw_tracks = [
        Path("/hw/1 Daftendirekt.mp4"),
        Path("/hw/2 WDPK 83.7 FM.mp4"),
        Path("/hw/3 Revolution 909.mp4"),
        Path("/hw/10 High Fidelity.mp4"),
        Path("/hw/11 Rock'n Roll.mp4"),
    ]
    two_chainz = Path("/hw/2 Chainz - Birthday Song.mp4")
    assert infer_unpadded_track_paths([*hw_tracks, two_chainz]) == set(hw_tracks)

    # 4. Same-artist unpadded dashed album with a gap (1, 2, 3, 10) confirms all 4 of that
    # artist while leaving a different artist ("4 Non Blondes") untouched!
    same_art = [
        Path("/dp/1 Daft Punk - One More Time.mp4"),
        Path("/dp/2 Daft Punk - Aerodynamic.mp4"),
        Path("/dp/3 Daft Punk - Digital Love.mp4"),
        Path("/dp/10 Daft Punk - Voyager.mp4"),
    ]
    four_non = Path("/dp/4 Non Blondes - What's Up.mp4")
    assert infer_unpadded_track_paths([*same_art, four_non]) == set(same_art)

    # 5. Zero-padded VA compilation continues smoothly past track 9 while protecting 50 Cent.
    va_10 = Path("/va/10 Artist J - Song 10.mp4")
    va_50 = Path("/va/50 Cent - In Da Club.mp4")
    va_paths = [
        *[Path(f"/va/{i:02d} Artist {chr(64 + i)} - Song {i}.mp4") for i in range(1, 10)],
        va_10,
        va_50,
    ]
    assert infer_unpadded_track_paths(va_paths) == {va_10}

    # 6. Sparse same-artist tracks (01 + 14) confirm 14 while protecting 50 Cent.
    p14 = Path("/mix/14 Daft Punk - Too Long.mp4")
    assert infer_unpadded_track_paths(
        [
            Path("/mix/01 Daft Punk - One More Time.mp4"),
            p14,
            Path("/mix/50 Cent - In Da Club.mp4"),
        ]
    ) == {p14}


def test_numeric_title_before_dash_keeps_swapped_candidate_for_phase2() -> None:
    h = parse_hints(_probe("/downloads/22 - Taylor Swift.mp4"))
    assert h.track_number == 22
    assert h.swapped_candidate == ("Taylor Swift", "22")


def test_fullwidth_cjk_noise_brackets_stripped() -> None:
    h1 = parse_hints(_probe("/downloads/張學友 - 吻別【MV】.mp4"))
    assert h1.artist == "張學友"
    assert h1.title == "吻別"

    h2 = parse_hints(_probe("/downloads/張學友 - 吻別\uff08Official Video\uff09.mp4"))
    assert h2.artist == "張學友"
    assert h2.title == "吻別"

    h3 = parse_hints(_probe("/downloads/周杰倫 - 晴天【官方MV】.mp4"))
    assert h3.artist == "周杰倫"
    assert h3.title == "晴天"

    h4 = parse_hints(_probe("/downloads/방탄소년단 - Dynamite (공식 뮤직비디오).mp4"))
    assert h4.artist == "방탄소년단"
    assert h4.title == "Dynamite"

    h5 = parse_hints(_probe("/downloads/아이유 - 밤편지 [공식 MV].mp4"))
    assert h5.artist == "아이유"
    assert h5.title == "밤편지"

    h6 = parse_hints(_probe("/downloads/a-ha - Take On Me (Official Video) [Remastered in 4K].mp4"))
    assert h6.artist == "a-ha"
    assert h6.title == "Take On Me"
    assert "remaster" in h6.edition_flags

    h7 = parse_hints(
        _probe(
            "/downloads/Linkin Park - Numb (Official Music Video) [4K UPGRADE] [kXYiU_JCYtU].mp4"
        )
    )
    assert h7.artist == "Linkin Park"
    assert h7.title == "Numb"


def test_parent_hints_skip_user_home_directory() -> None:
    hints = parse_hints(_probe("/Users/alice/my_clips/song.mp4"))
    assert hints.parent_album_hint == "my_clips"
    assert hints.parent_artist_hint is None


def test_multi_disc_filename_prefix_and_generic_folder_stop_list() -> None:
    probe = _probe("/Users/alice/Downloads/2-01 The Beatles - Birthday (Live).mkv")
    hints = parse_hints(probe)
    assert hints.disc_number == 2
    assert hints.track_number == 1
    assert hints.artist == "The Beatles"
    assert hints.title == "Birthday (Live)"
    assert "live" in hints.edition_flags
    assert hints.parent_album_hint is None
    assert hints.parent_artist_hint is None


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("VID_0042", True),
        ("IMG_1234", True),
        ("videoplayback", True),
        ("videoplayback (2)", True),
        ("clip_01", True),
        ("12345678", True),
        ("550e8400-e29b-41d4-a716-446655440000", True),
        ("Daft Punk - One More Time", False),
        ("1999", True),
    ],
)
def test_hintless_stems(stem: str, expected: bool) -> None:
    assert is_hintless_stem(stem) is expected


def test_hintless_file_without_tags_marked_hintless() -> None:
    hints = parse_hints(_probe("/Downloads/VID_0042.mp4"))
    assert hints.is_hintless is True
    assert hints.trusted_artist is False
    assert hints.trusted_title is False


def test_title_only_downloader_tags_prefer_the_filename_title() -> None:
    """Real-world regression: 4K Video Downloader+ stores only the raw video title."""
    elevation = parse_hints(
        _probe(
            "/x/Christian/Elevation Worship - Praise (feat. Brandon Lake, Chris Brown & "
            "Chandler Moore).mp4",
            {"title": "Praise | Elevation Worship"},
        )
    )
    assert elevation.artist == "Elevation Worship"
    assert elevation.title == "Praise"
    assert elevation.trusted_artist is True

    zhao = parse_hints(
        _probe(
            "/x/Zhao Chuan {趙傳} - At Last I Have Lost You {我終於失去了妳}"
            "Official Music Video.mp4",
            {
                "title": "趙傳 Chao Chuan【我終於失去了妳 At Last I Have Lost You】"
                "Official Music Video"
            },
        )
    )
    assert zhao.artist == "Zhao Chuan {趙傳}"
    assert zhao.title == "At Last I Have Lost You {我終於失去了妳}"

    # A clean classical title tag that does not repeat the artist is still preferred.
    classical = parse_hints(
        _probe("/x/Beethoven - Fifth.mp4", {"title": "Symphony No. 5 - Allegro con brio"})
    )
    assert classical.title == "Symphony No. 5 - Allegro con brio"


def test_bilingual_context_trimmed_lyric_video_noise_and_tv_show_artist() -> None:
    han = parse_hints(
        _probe(
            "/x/Han Hong {韩红} - Hai Kuo Tian Kong {海阔天空} at I Am A Singer 3 EP 5 "
            "{我是歌手 3}.mp4"
        )
    )
    assert han.title == "Hai Kuo Tian Kong {海阔天空}"

    getty = parse_hints(
        _probe("/x/Kristyn Getty - Before the Throne of God Above Official Lyric Video.mp4")
    )
    assert getty.title == "Before the Throne of God Above"

    show = parse_hints(
        _probe("/x/Come Sing With Me S02 - Chief Zhao's I'm ugly but gentle Ep.3 Single.mp4")
    )
    assert show.trusted_artist is False


def test_embedded_title_kept_when_only_prefixed_or_word_matched() -> None:
    """Review regressions: sanitized filenames and short artist names keep the tag title."""
    cases = [
        (
            "/x/4 Non Blondes - What's Up_.mp4",
            "4 Non Blondes - What's Up? (Official Music Video)",
            "What's Up?",
        ),
        (
            "/x/The Black Eyed Peas - Where Is The Love_ (Official Music Video).mp4",
            "The Black Eyed Peas - Where Is The Love? (Official Music Video)",
            "Where Is The Love?",
        ),
        ("/x/Queen - KQ.mp4", "Killer Queen", "Killer Queen"),
        ("/x/Yes - Roundabout live cut.mp4", "Yesterday and Today", "Yesterday and Today"),
        # Reversed "Title - Artist" tags drop the artist suffix (any name length).
        ("/x/Michael Jackson - Who Is It_.mp4", "Who Is It? - Michael Jackson", "Who Is It?"),
        ("/x/M - Pop Muzik.mp4", "Pop Muzik - M", "Pop Muzik"),
        # A hyphen without spaces is part of a word, not a separator.
        ("/x/Yes - Roundabout.mp4", "Yes-Man Blues", "Yes-Man Blues"),
        # Raw video titles still lose to the cleaner filename title.
        (
            "/x/Panic! At The Disco - I Write Sins Not Tragedies.mp4",
            "Panic! At The Disco: I Write Sins Not Tragedies [OFFICIAL VIDEO]",
            "I Write Sins Not Tragedies",
        ),
        ("/x/Prince - 1999.mp4", "Prince - 1999 (Official Music Video)", "1999"),
        ("/x/周杰倫 - 晴天.mp4", "周杰倫 Jay Chou【晴天 Sunny Day】Official MV", "晴天"),
    ]
    for path, tag, expected in cases:
        assert parse_hints(_probe(path, {"title": tag})).title == expected, path


def test_latin_braces_are_not_bilingual_and_tv_rule_needs_a_show_name() -> None:
    acoustic = parse_hints(_probe("/x/Artist - Song {Live} at Wembley.mp4"))
    assert acoustic.title == "Song {Live} at Wembley"

    for name in ("S4", "Season 5", "EP 7"):
        assert parse_hints(_probe(f"/x/{name} - Song.mp4")).trusted_artist is True, name
    assert parse_hints(_probe("/x/Come Sing With Me S02 - Song.mp4")).trusted_artist is False


def test_edition_flags_come_only_from_qualifiers_not_names() -> None:
    """Review regression: a bare 'Live' or 'Acoustic' in a name never requests a version."""
    no_flags = [
        "/x/Oasis - Live Forever.mp4",
        "/x/Wings - Live and Let Die.mp4",
        "/x/T.I. - Live Your Life.mp4",
        "/x/Live - Lightning Crashes.mp4",
        "/x/Acoustic Alchemy - Mr. Chow.mp4",
        "/x/The Remix Kings - Song.mp4",
    ]
    for path in no_flags:
        assert parse_hints(_probe(path)).edition_flags == [], path

    expected = {
        "/x/Oasis - Live Forever (Live).mp4": ["live"],
        "/x/Artist - Song live at Wembley.mp4": ["live"],
        "/x/Artist - Song - Live 1985.mp4": ["live"],
        "/x/Artist - Song (Acoustic).mp4": ["acoustic"],
        "/x/Fleetwood Mac - Dreams (2004 Remaster).mp4": ["remaster"],
        "/x/Mr. Probz - Waves - Robin Schulz Remix.mp4": ["remix"],
        # Noise after the qualifier, follow-on words, two flag words, and curly braces.
        "/x/Artist - Song Live HD.mp4": ["live"],
        "/x/Artist - Song Live Official Video.mp4": ["live"],
        "/x/Artist - Song DJ X Remix Official Video.mp4": ["remix"],
        "/x/Artist - Song Live Version.mp4": ["live"],
        "/x/Artist - Song Acoustic Version.mp4": ["acoustic"],
        "/x/Artist - Song Instrumental Version.mp4": ["instrumental"],
        "/x/Artist - Song Live Acoustic.mp4": ["live", "acoustic"],
        "/x/Artist - Song {Live}.mp4": ["live"],
        "/x/Artist - Song {Acoustic}.mp4": ["acoustic"],
    }
    for path, flags in expected.items():
        assert parse_hints(_probe(path)).edition_flags == flags, path
