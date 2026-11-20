"""Tests for pure candidate ranking, identity margins, and edition tie-breaking."""

from __future__ import annotations

import pytest

from audio_from_video.config import MatchingConfig
from audio_from_video.models import Candidate, MatchDecision, ReleaseRef
from audio_from_video.ranking import (
    TITLE_ONLY_CAP,
    Query,
    choose_edition,
    choose_release_group,
    classify,
    duration_score,
    edition_sort_key,
    evaluate_candidates,
    find_markers,
    identity_key,
    normalize_text,
    rank_candidates,
    score_candidate,
    strip_qualifiers,
    title_similarity,
    type_rank,
)

CFG = MatchingConfig()


def cand(
    title: str = "One More Time",
    artist: str = "Daft Punk",
    *,
    album: str | None = "Discovery",
    rg: str = "rg-1",
    primary: str | None = "Album",
    secondary: list[str] | None = None,
    duration: float | None = None,
    date: str | None = "2001-03-12",
    releases: list[ReleaseRef] | None = None,
    source: str = "musicbrainz",
    aliases: list[str] | None = None,
) -> Candidate:
    return Candidate(
        source=source,
        title=title,
        artist=artist,
        lead_artist=artist,
        artist_credit=artist,
        artist_aliases=aliases or [],
        album=album,
        primary_type=primary,
        secondary_types=secondary or [],
        duration_seconds=duration,
        release_group_id=rg,
        release_group_date=date,
        recording_id=f"rec-{rg}-{title}",
        releases=releases or [],
    )


def rel(rid: str, **kw: object) -> ReleaseRef:
    base: dict[str, object] = {
        "release_id": rid,
        "title": "Album",
        "status": "Official",
        "date": "2001-03-12",
        "country": "XW",
        "disc_count": 1,
        "track_count": 12,
        "formats": ("CD",),
        "track_number": 3,
        "disc_number": 1,
        "medium_track_count": 12,
    }
    base.update(kw)
    return ReleaseRef(**base)  # type: ignore[arg-type]


# -- Normalization --------------------------------------------------------------------


def test_normalize_text_quotes_dashes_ellipsis_and_punctuation() -> None:
    assert normalize_text("Don\u2019t Stop \u2014 Me\u2026") == normalize_text("Dont Stop - Me...")
    assert normalize_text("\uff21\uff22\uff23") == "abc"  # NFKC full-width.
    assert normalize_text("AC/DC") == "ac dc"
    assert normalize_text("Simon & Garfunkel") == "simon and garfunkel"
    assert normalize_text("Beyoncé") == "beyonce"
    # Traditional and Simplified Chinese compare equal (normalized to Simplified).
    assert normalize_text("吻別!") == "吻别"
    assert normalize_text("我是一隻小小鳥") == normalize_text("我是一只小小鸟")
    # Punctuation inside CJK text does not split it into separate words.
    assert normalize_text("我很醜\uff0c可是我很溫柔") == normalize_text("我很醜可是我很溫柔")
    assert normalize_text(None) == ""


def test_strip_qualifiers() -> None:
    assert strip_qualifiers("More Than This - 2012 Remaster") == "More Than This"
    assert strip_qualifiers("One More Time (Radio Edit)") == "One More Time"
    assert strip_qualifiers("Take On Me [Remastered in 4K]") == "Take On Me"
    assert strip_qualifiers("Live and Let Die") == "Live and Let Die"
    assert strip_qualifiers("(Remix)") == "(Remix)"


def test_find_markers() -> None:
    assert find_markers("Hello (Karaoke Version)") == {"karaoke"}
    assert "nightcore" in find_markers("Faded (Nightcore)")
    assert "piano" in find_markers("Shake It Off (Piano Version)")
    assert {"slowed", "reverb"} <= find_markers("Shape of You (Slowed + Reverb)")
    assert "8-bit" in find_markers("Africa (8-Bit Version)")
    assert find_markers("Bohemian Rhapsody") == frozenset()


def test_title_similarity_tolerates_qualifiers() -> None:
    assert title_similarity("One More Time", "One More Time") == 1.0
    assert title_similarity("One More Time", "One More Time (Radio Edit)") >= 0.95
    assert title_similarity("Hello", "Hello, Goodbye") < 0.6
    assert title_similarity("The Goodbye Kiss {吻別}", "吻別") == 1.0


# -- Scoring --------------------------------------------------------------------------


def test_duration_tolerance_and_unknown_redistribution() -> None:
    assert duration_score(320, 322) == 1.0
    assert duration_score(320, 335) == pytest.approx(0.7)
    assert duration_score(320, 400) == 0.0
    assert duration_score(None, 320) is None
    q_none = Query(title="One More Time", artist="Daft Punk")
    q_dur = Query(title="One More Time", artist="Daft Punk", duration_seconds=320)
    c = cand(duration=320, primary="Single")
    assert score_candidate(q_none, c) == pytest.approx(1.0)
    assert score_candidate(q_dur, c) == pytest.approx(1.0)
    far = cand(duration=420, primary="Single")
    assert score_candidate(q_dur, far) < score_candidate(q_none, far)


def test_type_rank_and_live_remix_flip() -> None:
    assert type_rank("Album", []) == 0
    assert type_rank("EP", []) == 0
    assert type_rank("Single", []) == 1
    assert type_rank("Album", ["Compilation"]) == 2
    assert type_rank("Album", ["Soundtrack"]) == 2
    assert type_rank("Album", ["Live"]) == 2
    assert type_rank("Album", ["Remix"]) == 2
    assert type_rank("Album", ["Live"], ["live"]) == 0
    assert type_rank("Album", ["Remix"], ["remix"]) == 0
    assert type_rank("Album", [], ["live"]) == 1
    assert type_rank(None, []) == 2


def test_title_only_query_is_capped_and_never_auto_accepted() -> None:
    q = Query(title="Yesterday")
    c = cand(title="Yesterday", artist="The Beatles", album="Help!")
    assert score_candidate(q, c) <= TITLE_ONLY_CAP
    outcome = evaluate_candidates(q, [c], CFG)
    assert outcome.decision != MatchDecision.AUTO_ACCEPT
    assert classify(0.99, 0.5, has_artist=False, config=CFG) == MatchDecision.AMBIGUOUS


def test_adversarial_marker_penalty_both_directions() -> None:
    clean_q = Query(title="Hello", artist="Adele")
    karaoke = cand(title="Hello (Karaoke Version)", artist="Adele", album="Karaoke Hits")
    original = cand(title="Hello", artist="Adele", album="25", rg="rg-25")
    assert score_candidate(clean_q, karaoke) < 0.7
    assert score_candidate(clean_q, original) == pytest.approx(1.0)
    karaoke_q = Query(title="Hello", artist="Adele", markers=frozenset({"karaoke"}))
    assert score_candidate(karaoke_q, original) < 0.7
    outcome = evaluate_candidates(karaoke_q, [original], CFG)
    assert outcome.decision != MatchDecision.AUTO_ACCEPT


def test_live_qualifier_penalized_without_flag() -> None:
    q = Query(title="One More Time", artist="Daft Punk")
    live = cand(title="One More Time (Live)", primary="Album", secondary=["Live"], rg="rg-live")
    assert score_candidate(q, live) < 0.9
    q_live = Query(title="One More Time", artist="Daft Punk", edition_flags=frozenset({"live"}))
    assert score_candidate(q_live, live) > score_candidate(q, live)


def test_folder_album_boost() -> None:
    q = Query(title="Song", artist="Artist X", album_hint="Second Album")
    a = cand(title="Song", artist="Artist X", album="Second Album", rg="rg-a")
    b = cand(title="Song", artist="Artist X", album="First Album", rg="rg-b")
    q_no = Query(title="Song", artist="Artist Y")
    assert score_candidate(q_no, a) == score_candidate(q_no, b)
    best = choose_release_group(q, rank_candidates(q, [a, b])[0].candidates)
    assert best.release_group_id == "rg-a"


def test_artist_alias_matching() -> None:
    q = Query(title="吻別", artist="Jacky Cheung")
    c = cand(title="吻別", artist="張學友", album="吻別", aliases=["Jacky Cheung"])
    assert score_candidate(q, c) >= 0.95


# -- Identities and margins ------------------------------------------------------------


def test_identity_groups_ignore_parenthetical_qualifiers() -> None:
    a = cand(title="Shake It Off", artist="Taylor Swift")
    b = cand(title="Shake It Off (Taylor's Version)", artist="Taylor Swift")
    assert identity_key(a) == identity_key(b)
    assert identity_key(a) != identity_key(cand(title="Shake It Off", artist="Other Artist"))


def test_margin_is_measured_between_identities() -> None:
    q = Query(title="One More Time", artist="Daft Punk")
    album = cand(rg="rg-album")
    single = cand(rg="rg-single", primary="Single", album="One More Time")
    other = cand(title="One More Time", artist="Britney Spears", rg="rg-b", album="Circus")
    outcome = evaluate_candidates(q, [single, album, other], CFG)
    assert outcome.decision == MatchDecision.AUTO_ACCEPT
    assert outcome.best is not None and outcome.best.release_group_id == "rg-album"
    assert outcome.margin >= 0.25
    assert [c.artist for c in outcome.alternatives] == ["Britney Spears"]


def test_close_identities_are_ambiguous() -> None:
    q = Query(title="Hurt", artist="Cash")
    a = cand(title="Hurt", artist="Johnny Cash", rg="rg-a")
    b = cand(title="Hurt", artist="Rosanne Cash", rg="rg-b")
    outcome = evaluate_candidates(q, [a, b], CFG)
    assert outcome.margin < CFG.min_margin
    assert outcome.decision == MatchDecision.AMBIGUOUS


def test_low_scores_are_unmatched_and_empty_is_unmatched() -> None:
    q = Query(title="Completely Different", artist="Nobody")
    outcome = evaluate_candidates(q, [cand()], CFG)
    assert outcome.decision == MatchDecision.UNMATCHED
    empty = evaluate_candidates(q, [], CFG)
    assert empty.decision == MatchDecision.UNMATCHED
    assert empty.best is None


def test_alternatives_capped_at_five() -> None:
    q = Query(title="Song", artist="Artist")
    cands = [cand(title="Song", artist=f"Artist {i}", rg=f"rg-{i}") for i in range(9)]
    assert len(evaluate_candidates(q, cands, CFG).alternatives) == 5


# -- Release-group and edition choice -----------------------------------------------------


def test_release_group_type_rank_then_earliest_date_then_preferred() -> None:
    q = Query(title="Song", artist="Artist")
    comp = cand(
        title="Song",
        artist="Artist",
        rg="rg-comp",
        secondary=["Compilation"],
        date="1990-01-01",
        album="Greatest Hits",
    )
    single = cand(
        title="Song",
        artist="Artist",
        rg="rg-single",
        primary="Single",
        date="1984-01-01",
        album="Song",
    )
    first = cand(title="Song", artist="Artist", rg="rg-first", date="1985-06-01", album="First")
    later = cand(title="Song", artist="Artist", rg="rg-later", date="1999-06-01", album="Later")
    group = rank_candidates(q, [comp, single, later, first])[0].candidates
    assert choose_release_group(q, group).release_group_id == "rg-first"
    pref = Query(title="Song", artist="Artist", preferred_release_group_ids=frozenset({"rg-later"}))
    group_pref = rank_candidates(pref, [comp, single, later, first])[0].candidates
    assert choose_release_group(pref, group_pref).release_group_id == "rg-later"


def test_edition_tiebreak_prefers_fewer_tracks() -> None:
    standard = rel("std", track_count=12, medium_track_count=12)
    expanded = rel("exp", track_count=18, medium_track_count=18)
    chosen = choose_edition(cand(releases=[expanded, standard]))
    assert chosen.release_id == "std"
    assert chosen.track_total == 12


def test_edition_tiebreak_prefers_original_over_later_rerelease() -> None:
    original = rel("orig", date="1975-11-21", country="GB", formats=('12" Vinyl',))
    reissue = rel("re", date="2011-09-05", country="XW", formats=("CD",))
    assert choose_edition(cand(releases=[reissue, original])).release_id == "orig"


def test_edition_tiebreak_full_ordering() -> None:
    bootleg = rel("boot", status="Bootleg", date="1970-01-01")
    deluxe = rel("dlx", date="1999-01-01", disambiguation="deluxe edition")
    deluxe_title = rel("dlx2", date="1999-01-01", title="Album (Deluxe)")
    two_disc = rel("2cd", date="2000-01-01", disc_count=2)
    vinyl = rel("vinyl", date="2001-03-12", formats=("Vinyl",))
    jp = rel("jp", date="2001-03-12", country="JP")
    best = rel("best", date="2001-03-12")
    ordered = sorted(
        [bootleg, deluxe, deluxe_title, two_disc, vinyl, jp, best], key=edition_sort_key
    )
    assert ordered[0].release_id == "best"
    assert ordered[-1].release_id == "boot"
    assert [r.release_id for r in ordered[:3]] == ["best", "jp", "vinyl"]
    chosen = choose_edition(cand(releases=[deluxe, two_disc, best]))
    assert chosen.track_number == 3
    assert chosen.disc_number == 1
    assert chosen.release_date == "2001-03-12"


def test_choose_edition_without_releases_is_noop() -> None:
    c = cand(releases=[])
    assert choose_edition(c) is c


# -- Review regressions ---------------------------------------------------------------


def test_studio_album_names_with_marker_words_are_not_penalized() -> None:
    qotsa = Query(title="Little Sister", artist="Queens of the Stone Age")
    album = cand(
        title="Little Sister",
        artist="Queens of the Stone Age",
        album="Lullabies to Paralyze",
        rg="rg-ltp",
        date="2005-03-21",
    )
    single = cand(
        title="Little Sister",
        artist="Queens of the Stone Age",
        album="Little Sister",
        rg="rg-single",
        primary="Single",
        date="2005-01-01",
    )
    assert score_candidate(qotsa, album) == pytest.approx(1.0)
    outcome = evaluate_candidates(qotsa, [album, single], CFG)
    assert outcome.decision == MatchDecision.AUTO_ACCEPT
    assert outcome.best is not None and outcome.best.album == "Lullabies to Paralyze"
    only_album = evaluate_candidates(qotsa, [album], CFG)
    assert only_album.decision == MatchDecision.AUTO_ACCEPT

    newman = Query(title="Love Me Again", artist="John Newman")
    tribute = cand(title="Love Me Again", artist="John Newman", album="Tribute", rg="rg-t")
    assert score_candidate(newman, tribute) == pytest.approx(1.0)
    assert evaluate_candidates(newman, [tribute], CFG).decision == MatchDecision.AUTO_ACCEPT


def test_unambiguous_album_phrases_still_penalize_other_artists() -> None:
    q = Query(title="Love Me Again", artist="John Newman")
    tribute_album = cand(
        title="Love Me Again",
        artist="Studio Session Players",
        album="A Tribute to John Newman",
        rg="rg-x",
    )
    karaoke_album = cand(
        title="Love Me Again", artist="Party Singers", album="Karaoke Hits Vol. 3", rg="rg-y"
    )
    plain = cand(title="Love Me Again", artist="Studio Session Players", album="Covers", rg="rg-z")
    assert score_candidate(q, tribute_album) < score_candidate(q, plain) - 0.3
    assert score_candidate(q, karaoke_album) < 0.4


def test_music_video_extra_length_is_not_penalized() -> None:
    # Exact matches (<= 3s) still beat +60/+90s clips, while MVs up to +90s stay >= 0.88
    # and ultra-long narrative videos settle to a 0.40 floor so exact artist+title matches
    # keep the canonical studio recording in play.
    assert duration_score(320 + 60, 320) >= 0.90
    assert duration_score(320 + 90, 320) == pytest.approx(0.88)
    assert duration_score(320 + 2, 320) > duration_score(320 + 75, 320)
    assert duration_score(320 + 165, 320) == pytest.approx(0.64)
    assert duration_score(320 + 400, 320) == pytest.approx(0.40)
    assert duration_score(320 - 30, 320) < 0.7
    q = Query(title="One More Time", artist="Daft Punk", duration_seconds=320 + 75)
    outcome = evaluate_candidates(q, [cand(duration=320)], CFG)
    assert outcome.decision == MatchDecision.AUTO_ACCEPT
    long_mv = Query(title="One More Time", artist="Daft Punk", duration_seconds=320 + 450)
    assert evaluate_candidates(long_mv, [cand(duration=320)], CFG).decision == (
        MatchDecision.AUTO_ACCEPT
    )


def test_various_artists_karaoke_and_tribute_bands_are_demoted() -> None:
    # Various Artists compilation without secondary-types loses to studio album.
    q = Query(title="Karma Police", artist="Radiohead")
    va = cand(
        title="Karma Police",
        artist="Radiohead",
        album="Human Rights Concerts",
        rg="rg-va",
        date="1996-01-01",
    )
    va.album_artist = "Various Artists"
    studio = cand(
        title="Karma Police",
        artist="Radiohead",
        album="OK Computer",
        rg="rg-okc",
        date="1997-05-21",
    )
    studio.album_artist = "Radiohead"
    outcome = evaluate_candidates(q, [va, studio], CFG)
    assert outcome.best is not None and outcome.best.album == "OK Computer"

    # Karaoke album credited to the real artist is still heavily penalized.
    q_bey = Query(title="Crazy in Love", artist="Beyoncé")
    karaoke = cand(
        title="Crazy in Love",
        artist="Beyoncé",
        album="Smash Hits! Karaoke: 12 Tracks",
        rg="rg-k",
    )
    real = cand(title="Crazy in Love", artist="Beyoncé", album="Dangerously in Love", rg="rg-dil")
    assert score_candidate(q_bey, karaoke) < 0.7
    assert evaluate_candidates(q_bey, [karaoke, real], CFG).best.album == "Dangerously in Love"

    # Tribute band with extra tokens cannot nearly tie the exact artist.
    q_abba = Query(title="Dancing Queen", artist="ABBA", duration_seconds=231)
    abba = cand(title="Dancing Queen", artist="ABBA", album="Arrival", rg="rg-arr", duration=231)
    tribute = cand(
        title="Dancing Queen",
        artist="The Real ABBA Gold",
        album="Janus",
        rg="rg-trib",
        duration=230,
    )
    abba_outcome = evaluate_candidates(q_abba, [abba, tribute], CFG)
    assert abba_outcome.decision == MatchDecision.AUTO_ACCEPT
    assert abba_outcome.margin >= CFG.min_margin


def test_remix_qualifier_prefers_requested_remixer_over_plain_studio_track() -> None:
    q = Query(
        title="I Follow Rivers",
        artist="Lykke Li",
        raw_title="I Follow Rivers (The Magician Remix)",
        edition_flags=frozenset({"remix"}),
        duration_seconds=280,
    )
    plain = cand(
        title="I Follow Rivers",
        artist="Lykke Li",
        album="I Follow Rivers (remixes)",
        primary="Single",
        secondary=["Remix"],
        duration=228,
        rg="rg-plain",
    )
    remix = cand(
        title="I Follow Rivers (The Magician remix)",
        artist="Lykke Li",
        album="I Follow Rivers (The Magician Remix)",
        primary="Single",
        secondary=["Remix"],
        duration=279,
        rg="rg-mag",
    )
    outcome = evaluate_candidates(q, [plain, remix], CFG)
    assert outcome.decision == MatchDecision.AUTO_ACCEPT
    assert outcome.best is not None and "magician" in outcome.best.title.lower()


def test_edition_sort_prefers_audio_matching_rg_title_and_disc_hint() -> None:
    cassette = rel("cas", date="1979", formats=("Cassette",), disc_count=1, disc_number=1)
    cd2 = rel("cd2", date="1985", formats=("CD",), disc_count=2, disc_number=2, track_number=6)
    chosen = choose_edition(cand(album="The Wall", releases=[cassette, cd2]), disc_hint=2)
    assert chosen.release_id == "cd2"

    jp_local = rel("jp", title="チャン・ヨウ", date="1998-10-21", country="JP")
    canonical = rel("tw", title="唱遊", date="1998-10", country="TW")
    chosen_nl = choose_edition(cand(album="唱遊", releases=[jp_local, canonical]))
    assert chosen_nl.release_id == "tw"


def test_part_number_and_taylors_version_are_preserved_in_ranking() -> None:
    p1 = cand(
        title="Another Brick in the Wall (Part 1)",
        artist="Pink Floyd",
        album="The Wall",
        duration=191,
        rg="rg-wall",
    )
    p2 = cand(
        title="Another Brick in the Wall (Part 2)",
        artist="Pink Floyd",
        album="The Wall",
        duration=239,
        rg="rg-wall",
    )
    assert identity_key(p1) != identity_key(p2)
    q_p2 = Query(
        title="Another Brick in the Wall",
        raw_title="Another Brick in the Wall (Part 2)",
        artist="Pink Floyd",
        duration_seconds=239,
    )
    out_p2 = evaluate_candidates(q_p2, [p1, p2], CFG)
    assert out_p2.decision == MatchDecision.AUTO_ACCEPT
    assert out_p2.best is not None and "Part 2" in out_p2.best.title

    orig = cand(
        title="Love Story",
        artist="Taylor Swift",
        album="Fearless",
        date="2008-11-11",
        rg="rg-orig",
    )
    tv = cand(
        title="Love Story (Taylor's Version)",
        artist="Taylor Swift",
        album="Fearless (Taylor's Version)",
        date="2021-04-09",
        rg="rg-tv",
    )
    q_tv = Query(
        title="Love Story",
        raw_title="Love Story (Taylor's Version)",
        artist="Taylor Swift",
    )
    out_tv = evaluate_candidates(q_tv, [orig, tv], CFG)
    assert out_tv.decision == MatchDecision.AUTO_ACCEPT
    assert out_tv.best is not None and out_tv.best.release_group_id == "rg-tv"


def test_query_artist_aliases_match_cross_script_names() -> None:
    c = cand(title="吻別", artist="Jacky Cheung", album="吻別", source="itunes")
    assert score_candidate(Query(title="吻別", artist="張學友"), c) < 0.8
    q = Query(title="吻別", artist="張學友", artist_aliases=("Jacky Cheung",))
    assert score_candidate(q, c) >= 0.95


def test_unknown_disc_count_uses_medium_position_as_lower_bound() -> None:
    on_disc_two = rel("d2", disc_count=None, disc_number=2, date="2000-01-01")
    single_disc = rel("d1", disc_count=None, disc_number=1, date="2001-01-01")
    assert choose_edition(cand(releases=[on_disc_two, single_disc])).release_id == "d1"


def test_title_only_ranking_uses_duration_and_popularity_without_clipping() -> None:
    from dataclasses import replace

    q = Query(title="Hallelujah", artist=None, duration_seconds=413.0)
    obscure_alpha = cand(
        title="Hallelujah",
        artist="Amaury Vassili",
        album="Vincerò",
        duration=372.0,
        releases=[rel("r1")],
        rg="rg-av",
    )
    famous_match = replace(
        cand(
            title="Hallelujah",
            artist="Jeff Buckley",
            album="Grace",
            duration=413.8,
            source="itunes",
            rg="rg-jb",
        ),
        search_rank=1,
    )
    out = evaluate_candidates(q, [obscure_alpha, famous_match], CFG)
    assert out.decision == MatchDecision.AMBIGUOUS
    assert out.best is not None and out.best.artist == "Jeff Buckley"
    assert out.score < 0.75 and out.margin > 0.05


def test_live_albums_beat_live_compilations_and_short_snippets_get_no_mv_allowance() -> None:
    assert duration_score(162.0, 36.7) == 0.0
    assert type_rank("Album", ["Live"], ["live"]) == 0
    assert type_rank("Album", ["Compilation", "Live"], ["live"]) == 2
    q = Query(
        title="Folsom Prison Blues",
        raw_title="Folsom Prison Blues (Live)",
        artist="Johnny Cash",
        duration_seconds=162.0,
        edition_flags=frozenset({"live"}),
    )
    canonical = cand(
        title="Folsom Prison Blues",
        artist="Johnny Cash",
        album="At Folsom Prison",
        secondary=["Live"],
        date="1968",
        duration=164.0,
        rg="rg-afp",
    )
    obscure_live = cand(
        title="Folsom Prison Blues (live)",
        artist="Johnny Cash",
        album="Live: Longing for Old Virginia",
        secondary=["Live"],
        date="2020-01-01",
        duration=161.6,
        rg="rg-va",
    )
    out = evaluate_candidates(q, [obscure_live, canonical], CFG)
    assert out.best is not None and out.best.album == "At Folsom Prison"


def test_adversarial_query_markers_never_auto_accept_even_on_karaoke_releases() -> None:
    q = Query(
        title="Don't Stop Me Now",
        artist="Queen",
        duration_seconds=209.0,
        markers=frozenset({"karaoke"}),
    )
    karaoke_rel = cand(
        title="Don't Stop Me Now",
        artist="Queen",
        album="Greatest Karaoke Hits",
        secondary=["Compilation"],
        duration=211.0,
        rg="rg-karaoke",
    )
    out = evaluate_candidates(q, [karaoke_rel], CFG)
    assert out.decision != MatchDecision.AUTO_ACCEPT


def test_same_artist_mbid_merges_spelling_variants_without_merging_collaborations() -> None:
    from dataclasses import replace

    q = Query(title="Hotel California", artist="Eagles", duration_seconds=391.0)
    studio = replace(
        cand(
            title="Hotel California",
            artist="Eagles",
            album="Hotel California",
            duration=390.3,
            date="1976-12-08",
            rg="rg-hc",
        ),
        artist_id="mbid-eagles",
    )
    the_eagles = replace(
        cand(
            title="Hotel California",
            artist="The Eagles",
            album="In Concert",
            secondary=["Live"],
            duration=None,
            date="1991",
            rg="rg-live",
        ),
        artist_id="mbid-eagles",
    )
    collab = replace(
        cand(
            title="Hotel California",
            artist="The Eagles & Friends",
            album="Duets",
            duration=390.0,
            date="2005",
            rg="rg-duets",
        ),
        artist_id="mbid-eagles",
        lead_artist="The Eagles",
    )
    identities = rank_candidates(q, [studio, the_eagles, collab])
    assert len(identities) == 2
    assert len(identities[0].candidates) == 2
    assert evaluate_candidates(q, [studio, the_eagles], CFG).decision == MatchDecision.AUTO_ACCEPT


def test_artist_studio_soundtrack_and_variant_vs_subtitle_qualifiers() -> None:
    from dataclasses import replace

    # Artist-owned studio soundtrack beats a later reissue; Various Artists soundtrack does not.
    q_pr = Query(title="Purple Rain", artist="Prince", duration_seconds=521.0)
    ost = replace(
        cand(
            title="Purple Rain",
            artist="Prince and The Revolution",
            album="Purple Rain",
            secondary=["Soundtrack"],
            date="1984-06-25",
            duration=521.5,
            rg="rg-pr",
        ),
        lead_artist="Prince",
        album_artist="Prince and The Revolution",
    )
    vault = cand(
        title="Purple Rain",
        artist="Prince",
        album="Sign o' the Times",
        date="2020-09-25",
        duration=340.0,
        rg="rg-sott",
    )
    vault.album_artist = "Prince"
    va_ost = cand(
        title="Purple Rain",
        artist="Prince",
        album="80s Movie Classics",
        secondary=["Soundtrack"],
        date="1983-01-01",
        duration=521.0,
        rg="rg-va",
    )
    va_ost.album_artist = "Various Artists"
    out_pr = evaluate_candidates(q_pr, [va_ost, vault, ost], CFG)
    assert out_pr.best is not None and out_pr.best.album == "Purple Rain"

    # Explicit variant ("... version") loses to plain studio title, while subtitles stay intact.
    q_hb = Query(title="Hyperballad", artist="Björk", duration_seconds=321.0)
    plain = cand(title="Hyperballad", artist="Björk", album="Post", date="1995-06-13", rg="rg-p1")
    variant = cand(
        title="Hyperballad (Brodsky Quartet version)",
        artist="Björk",
        album="Post",
        date="1994-01-01",
        rg="rg-p2",
    )
    assert evaluate_candidates(q_hb, [variant, plain], CFG).best.title == "Hyperballad"

    q_rm = Query(title="Rocket Man", artist="Elton John", duration_seconds=281.0)
    subtitle = cand(
        title="Rocket Man (I Think It's Going to Be a Long, Long Time)",
        artist="Elton John",
        album="Honky Château",
        date="1972-05-19",
        rg="rg-hc",
    )
    later = cand(
        title="Rocket Man",
        artist="Elton John",
        album="Goodbye Yellow Brick Road",
        date="1973-10-05",
        rg="rg-gybr",
    )
    assert evaluate_candidates(q_rm, [later, subtitle], CFG).best.album == "Honky Château"


def test_title_only_duration_disambiguates_exact_length_over_shorter_covers() -> None:
    from dataclasses import replace

    q = Query(title="Mad World", artist=None, duration_seconds=215.0)
    shorter_cover = replace(
        cand(
            title="Mad World",
            artist="Gary Jules",
            album="Mad World",
            primary="Single",
            duration=187.8,
            source="itunes",
            rg="rg-gj",
        ),
        search_rank=1,
    )
    no_duration = cand(
        title="Mad World",
        artist="Unknown Band",
        album="Album",
        duration=None,
        releases=[rel("r1")],
        rg="rg-unk",
    )
    exact_original = replace(
        cand(
            title="Mad World",
            artist="Tears for Fears",
            album="The Hurting",
            duration=215.5,
            source="itunes",
            rg="rg-tff",
        ),
        search_rank=13,
    )
    out = evaluate_candidates(q, [shorter_cover, no_duration, exact_original], CFG)
    assert out.decision == MatchDecision.AMBIGUOUS
    assert out.best is not None and out.best.artist == "Tears for Fears"


def test_video_recordings_and_dualdisc_dvd_formats_are_demoted_and_never_auto_accepted() -> None:
    from dataclasses import replace

    # Gotye case: video recording (244.5 s) vs. audio recording (245.0 s) on Making Mirrors.
    q_gotye = Query(
        title="Somebody That I Used to Know",
        artist="Gotye",
        duration_seconds=244.0,
    )
    video_rec = replace(
        cand(
            title="Somebody That I Used to Know",
            artist="Gotye",
            album="Video Mirrors",
            duration=244.457,
            date="2013-01-01",
            releases=[rel("rel-vid", formats=("DVD-Video",), date="2013-01-01")],
            rg="rg-mm",
        ),
        is_video=True,
    )
    audio_rec = cand(
        title="Somebody That I Used to Know",
        artist="Gotye",
        album="Making Mirrors",
        duration=244.973,
        date="2011-08-19",
        releases=[
            rel("rel-dvd", formats=("DualDisc (DVD-Video side)",), date="2011-08-19"),
            rel("rel-cd", formats=("CD",), date="2011-08-19"),
        ],
        rg="rg-mm",
    )
    out = evaluate_candidates(q_gotye, [video_rec, audio_rec], CFG)
    assert out.decision == MatchDecision.AUTO_ACCEPT
    assert out.best is not None
    assert not out.best.is_video
    assert out.best.release_id == "rel-cd"

    # Video-only match (like Björk's DVD-only video recording) never auto-accepts.
    q_bjork = Query(title="Hyperballad", artist="Björk", duration_seconds=321.0)
    video_only = replace(
        cand(
            title="Hyperballad",
            artist="Björk",
            album="Post",
            duration=235.0,
            date="2006-07-10",
            releases=[rel("rel-dualdisc", formats=("DualDisc (DVD-Video side)",), date="2006")],
            rg="rg-post",
        ),
        is_video=True,
    )
    out_vid = evaluate_candidates(q_bjork, [video_only], CFG)
    assert out_vid.decision == MatchDecision.AMBIGUOUS


def test_unrecognized_qualifiers_and_extreme_durations_block_auto_accept() -> None:
    studio = cand(
        title="Thriller",
        artist="Michael Jackson",
        album="Thriller",
        duration=357.0,
        date="1982-11-30",
        rg="rg-thriller",
    )
    for bad_title, dur in (
        ("Thriller (Full Album)", 2540.0),
        ("Thriller (Guitar Tutorial)", 720.0),
        ("Thriller (Interview)", 1260.0),
        ("Thriller (10 Hours)", 36000.0),
        ("Thriller (Bass Boosted)", 357.0),
        ("Thriller (Vocals Only)", 357.0),
        ("Thriller (Violin Version)", 357.0),
        ("Thriller (Orchestral Version)", 357.0),
        ("Thriller (Lofi Remake)", 357.0),
        ("Thriller (Mashup)", 357.0),
        ("Thriller (Fan Edit)", 357.0),
    ):
        q = Query(
            title="Thriller",
            raw_title=bad_title,
            artist="Michael Jackson",
            duration_seconds=dur,
        )
        out = evaluate_candidates(q, [studio], CFG)
        assert out.decision != MatchDecision.AUTO_ACCEPT, bad_title

    # Extreme duration excess blocks even without a descriptor in the title.
    q_long_album = Query(title="Thriller", artist="Michael Jackson", duration_seconds=2540.0)
    assert evaluate_candidates(q_long_album, [studio], CFG).decision != MatchDecision.AUTO_ACCEPT
    # Legitimate 13:42 Thriller official video still auto-accepts.
    q_mv = Query(title="Thriller", artist="Michael Jackson", duration_seconds=822.0)
    assert evaluate_candidates(q_mv, [studio], CFG).decision == MatchDecision.AUTO_ACCEPT


def test_real_titles_with_marker_words_auto_accept_while_qualifier_markers_block() -> None:
    for artist, title in (
        ("The Cure", "Lullaby"),
        ("Tenacious D", "Tribute"),
        ("Bruce Springsteen", "Cover Me"),
        ("The Strokes", "Under Cover of Darkness"),
        ("Diana Ross", "Chain Reaction"),
        ("Mariah Carey", "Music Box"),
        ("Drake", "Karaoke"),
        ("100 gecs", "ringtone"),
        ("The Cover Girls", "Show Me"),
    ):
        c = cand(title=title, artist=artist, album="Studio LP", duration=240.0, rg="rg-1")
        q = Query(
            title=title,
            raw_title=title,
            artist=artist,
            duration_seconds=240.0,
            markers=find_markers(title, artist),
            qualifier_markers=frozenset(),
        )
        out = evaluate_candidates(q, [c], CFG)
        assert out.decision == MatchDecision.AUTO_ACCEPT, f"{artist} - {title}"

    # Qualifier marker "(Nightcore)" still blocks even when a nightcore release matches.
    nc = cand(title="Nobody", artist="Mitski", album="nightcore BOPS", duration=190.0, rg="rg-nc")
    q_nc = Query(
        title="Nobody",
        raw_title="Nobody (Nightcore)",
        artist="Mitski",
        duration_seconds=190.0,
        markers=frozenset({"nightcore"}),
        qualifier_markers=frozenset({"nightcore"}),
    )
    assert evaluate_candidates(q_nc, [nc], CFG).decision != MatchDecision.AUTO_ACCEPT


def test_base_title_similarity_floor_prevents_short_or_wrong_title_auto_accept() -> None:
    for q_title, c_title in (
        ("I Want You Back", "I Want You"),
        ("Hold On", "Hold On Tight"),
        ("Stay", "Stay Away"),
        ("Intro", "Outro"),
    ):
        q = Query(title=q_title, artist="Artist", duration_seconds=200.0)
        c = cand(title=c_title, artist="Artist", album="Album", duration=200.0, rg="rg-1")
        out = evaluate_candidates(q, [c], CFG)
        assert out.decision != MatchDecision.AUTO_ACCEPT, f"{q_title} -> {c_title}"


def test_unknown_candidate_duration_scores_below_exact_duration_match() -> None:
    q = Query(title="Song", artist="Artist", duration_seconds=240.0)
    exact = cand(title="Song", artist="Artist", album="LP", duration=240.0, rg="rg-exact")
    missing_len = cand(title="Song", artist="Artist", album="LP2", duration=None, rg="rg-none")
    assert score_candidate(q, missing_len) < score_candidate(q, exact)
    # Still auto-accepts when it is the only clean match.
    assert evaluate_candidates(q, [missing_len], CFG).decision == MatchDecision.AUTO_ACCEPT


def test_same_script_substring_edition_title_is_not_penalized() -> None:
    jp_orig = rel("rel-1978", title="At Budokan", date="1978-10", country="JP")
    us_reissue = rel("rel-1991", title="Cheap Trick at Budokan", date="1991-01-01", country="US")
    chosen = choose_edition(cand(album="Cheap Trick at Budokan", releases=[jp_orig, us_reissue]))
    assert chosen.release_id == "rel-1978"


def test_unrequested_live_recording_never_auto_accepts() -> None:
    """Live-run regression: 'Dreams (2004 Remaster)' must not auto-accept a live recording."""
    query = Query(
        title="Dreams",
        artist="Fleetwood Mac",
        duration_seconds=257,
        edition_flags=frozenset({"remaster"}),
        raw_title="Dreams (2004 Remaster)",
    )
    live_title = cand(
        "Dreams (live at Wembley, 6/20/80)",
        "Fleetwood Mac",
        album="Tusk",
        rg="rg-tusk",
        duration=254,
    )
    live_album = cand(
        "Dreams",
        "Fleetwood Mac",
        album="Live in Boston",
        rg="rg-boston",
        secondary=["Live"],
        duration=257,
    )
    for only in (live_title, live_album):
        outcome = evaluate_candidates(query, [only], CFG)
        assert outcome.decision != MatchDecision.AUTO_ACCEPT

    # An explicitly requested live version may still auto-accept.
    live_query = Query(
        title="Dreams",
        artist="Fleetwood Mac",
        duration_seconds=257,
        edition_flags=frozenset({"live"}),
        raw_title="Dreams (Live)",
    )
    assert evaluate_candidates(live_query, [live_album], CFG).decision == (
        MatchDecision.AUTO_ACCEPT
    )

    # With the studio recording present, it wins and auto-accepts.
    studio = cand("Dreams", "Fleetwood Mac", album="Rumours", rg="rg-rumours", duration=257)
    outcome = evaluate_candidates(query, [live_title, live_album, studio], CFG)
    assert outcome.best is not None and outcome.best.album == "Rumours"
    assert outcome.decision == MatchDecision.AUTO_ACCEPT


def test_various_artists_compilation_and_contradicted_unknown_duration_never_auto_accept() -> None:
    """Real-world regressions L1 and L2 (Chinese compilations beating the original album)."""
    query = Query(title="我很醜可是我很溫柔", artist="趙傳", duration_seconds=235)
    va = cand("我很醜可是我很溫柔", "趙傳", album="10年朋友6", rg="rg-va", duration=231)
    va.album_artist = "Various Artists"
    va.secondary_types = ["Compilation"]
    assert evaluate_candidates(query, [va], CFG).decision != MatchDecision.AUTO_ACCEPT

    original = cand(
        "我很醜\uff0c可是我很溫柔",
        "趙傳",
        album="我很醜\uff0c可是我很溫柔",
        rg="rg-orig",
        duration=235,
    )
    original.album_artist = "趙傳"
    outcome = evaluate_candidates(query, [va, original], CFG)
    assert outcome.best is not None and outcome.best.release_group_id == "rg-orig"
    assert outcome.decision == MatchDecision.AUTO_ACCEPT

    # A 133 s clip vs. a box-set recording with no length, while timed recordings are 280 s.
    clip = Query(title="我是一只小小鸟", artist="赵传", duration_seconds=133)
    untimed = cand("我是一只小小鸟", "赵传", album="东方之珠", rg="rg-box", duration=None)
    timed = cand("我是一隻小小鳥", "赵传", album="我是一隻小小鳥", rg="rg-orig2", duration=280)
    assert evaluate_candidates(clip, [untimed, timed], CFG).decision != MatchDecision.AUTO_ACCEPT


def test_requested_acoustic_instrumental_or_demo_version_wins_or_prompts() -> None:
    """Review regression: '(Acoustic)', '(Instrumental)', or '(Demo)' never gets the studio cut."""
    studio = cand("Dreams", "Fleetwood Mac", album="Rumours", rg="rg-rumours", duration=257)
    for qualifier, length in (("Acoustic", 240), ("Instrumental", 250), ("Demo", 230)):
        variant = cand(
            f"Dreams ({qualifier})",
            "Fleetwood Mac",
            album=f"{qualifier} Sessions",
            rg=f"rg-{qualifier.lower()}",
            duration=length,
        )
        query = Query(
            title="Dreams",
            artist="Fleetwood Mac",
            duration_seconds=length,
            raw_title=f"Dreams ({qualifier})",
            edition_flags=frozenset({qualifier.lower()}) - {"demo"},
        )
        outcome = evaluate_candidates(query, [studio, variant], CFG)
        assert outcome.best is not None
        assert outcome.best.release_group_id == f"rg-{qualifier.lower()}", qualifier

        # With no matching version at all, the studio recording is never auto-accepted.
        only_studio = evaluate_candidates(query, [studio], CFG)
        assert only_studio.decision != MatchDecision.AUTO_ACCEPT, qualifier

    # A plain request still gets the studio recording.
    plain = Query(title="Dreams", artist="Fleetwood Mac", duration_seconds=257)
    acoustic = cand("Dreams (Acoustic)", "Fleetwood Mac", album="X", rg="rg-x", duration=257)
    outcome = evaluate_candidates(plain, [studio, acoustic], CFG)
    assert outcome.best is not None and outcome.best.release_group_id == "rg-rumours"
    assert outcome.decision == MatchDecision.AUTO_ACCEPT


def test_requested_variant_prefers_exact_length_and_album_or_live_recognition() -> None:
    """Review regressions: exact-length versions win, and album or live context counts."""
    query = Query(
        title="Dreams",
        artist="Fleetwood Mac",
        duration_seconds=240,
        raw_title="Dreams (Acoustic)",
        edition_flags=frozenset({"acoustic"}),
    )
    single = cand(
        "Dreams (Acoustic)",
        "Fleetwood Mac",
        album="Dreams (Acoustic)",
        rg="rg-single",
        primary="Single",
        duration=240,
        date="2021-01-01",
    )
    for album_length in (270, 280):
        album = cand(
            "Dreams (Acoustic)",
            "Fleetwood Mac",
            album="Acoustic Collection",
            rg="rg-album",
            duration=album_length,
            date="2010-01-01",
        )
        outcome = evaluate_candidates(query, [single, album], CFG)
        assert outcome.best is not None and outcome.best.release_group_id == "rg-single"
        assert outcome.decision == MatchDecision.AUTO_ACCEPT, album_length

    # A plainly titled track on an acoustic album counts as the acoustic version.
    studio = cand("Dreams", "Fleetwood Mac", album="Rumours", rg="rg-rumours", duration=257)
    sessions = cand(
        "Dreams", "Fleetwood Mac", album="The Acoustic Sessions", rg="rg-sessions", duration=240
    )
    outcome = evaluate_candidates(query, [studio, sessions], CFG)
    assert outcome.best is not None and outcome.best.release_group_id == "rg-sessions"
    assert outcome.decision == MatchDecision.AUTO_ACCEPT

    # "(Live Acoustic)": a live recording satisfies the acoustic request.
    live_acoustic = Query(
        title="Dreams",
        artist="Fleetwood Mac",
        duration_seconds=240,
        raw_title="Dreams (Live Acoustic)",
        edition_flags=frozenset({"live", "acoustic"}),
    )
    live = cand(
        "Dreams (Live)",
        "Fleetwood Mac",
        album="Live in Boston",
        rg="rg-live",
        secondary=["Live"],
        duration=240,
    )
    outcome = evaluate_candidates(live_acoustic, [studio, live], CFG)
    assert outcome.best is not None and outcome.best.release_group_id == "rg-live"
    assert outcome.decision == MatchDecision.AUTO_ACCEPT
